/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ./ops/sdca_ops.cc.

#define EIGEN_USE_THREADS

#include <stddef.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/linear_optimizer/kernels/hinge-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/logistic-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/smooth-hinge-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/squared-loss.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {

using UnalignedFloatVector = TTypes<const float>::UnalignedConstVec;
using UnalignedInt64Vector = TTypes<const int64>::UnalignedConstVec;

// Statistics computed with input (ModelWeights, Example).
struct ExampleStatistics {
  // feature_weights dot feature_values for the example.
  double wx = 0;
  // dot product using the previous weights
  double prev_wx = 0;
  // sum of squared feature values occurring in the example divided by
  // L2 * sum(example_weights).
  double normalized_squared_norm = 0;
};

class Regularizations {
 public:
  Regularizations(){};

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelConstruction* const context) {
    TF_RETURN_IF_ERROR(context->GetAttr("l1", &symmetric_l1_));
    TF_RETURN_IF_ERROR(context->GetAttr("l2", &symmetric_l2_));
    shrinkage_ = symmetric_l1_ / symmetric_l2_;
    return Status::OK();
  }

  // Proximal SDCA shrinking for L1 regularization.
  double Shrink(const double weight) const {
    const double shrinked = std::max(std::abs(weight) - shrinkage_, 0.0);
    if (shrinked > 0.0) {
      return std::copysign(shrinked, weight);
    }
    return 0.0;
  }

  // Vectorized float variant of the above.
  Eigen::Tensor<float, 1, Eigen::RowMajor> EigenShrink(
      const Eigen::Tensor<float, 1, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }

  float symmetric_l2() const { return symmetric_l2_; }

 private:
  float symmetric_l1_ = 0;
  float symmetric_l2_ = 0;

  // L1 divided by L2, pre-computed for use during weight shrinking.
  double shrinkage_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Regularizations);
};

class ModelWeights;

// Struct describing a single example.
class Example {
 public:
  // Compute dot product between weights, and example feature values. This
  // method also computes the normalized example norm used in SDCA update.
  const ExampleStatistics ComputeWxAndWeightedExampleNorm(
      const int num_loss_partitions, const ModelWeights& model_weights,
      const Regularizations& regularization) const;

  float example_label() const { return example_label_; }

  float example_weight() const { return example_weight_; }

  double squared_norm() const { return squared_norm_; }

 private:
  // Sparse features associated with the example.
  // Indices and Values are the associated feature index, and values. Values
  // can be optionally absent, in which we case we implicitly assume a value of
  // 1.0f.
  struct SparseFeatures {
    std::unique_ptr<UnalignedInt64Vector> indices;
    std::unique_ptr<UnalignedFloatVector> values;  // nullptr encodes optional.
  };
  std::vector<SparseFeatures> sparse_features_;

  // A dense vector which is a row-slice of the underlying matrix.
  struct DenseVector {
    // Returns a row slice from the matrix.
    Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> row()
        const {
      // TensorMap to a row slice of the matrix.
      return Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1),
          data_matrix.dimension(1));
    }

    const TTypes<float>::ConstMatrix data_matrix;
    const int64 row_index;
  };
  std::vector<std::unique_ptr<DenseVector>> dense_vectors_;

  float example_label_ = 0;
  float example_weight_ = 0;
  double squared_norm_ = 0;  // sum squared norm of the features.

  // Examples fills Example in a multi-threaded way.
  friend class Examples;

  // ModelWeights use each example for model update w += \alpha * x_{i};
  friend class ModelWeights;
};

// Weights related to features. For example, say you have two sets of sparse
// features i.e. age bracket and country, then FeatureWeights hold the
// parameters for it. We keep track of the original weight passed in and the
// delta weight which the optimizer learns in each call to the optimizer.
struct FeatureWeights {
  // The nominal value of the weight for a feature (indexed by its id).
  TTypes<const float>::Vec nominals;

  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Vec deltas;
};

// Weights in the model, wraps both current weights, and the delta weights
// for both sparse and dense features.
class ModelWeights {
 public:
  ModelWeights() {}

  // Go through all the features present in the example, and update the
  // weights based on the dual delta.
  void UpdateDeltaWeights(const Eigen::ThreadPoolDevice& device,
                          const Example& example,
                          const double normalized_bounded_dual_delta) {
    // Sparse weights.
    for (size_t j = 0; j < sparse_weights_.size(); ++j) {
      const Example::SparseFeatures& sparse_features =
          example.sparse_features_[j];
      FeatureWeights* const feature_weights = &sparse_weights_[j];
      for (int64 k = 0; k < sparse_features.indices->size(); ++k) {
        const double feature_value = sparse_features.values == nullptr
                                         ? 1.0
                                         : (*sparse_features.values)(k);
        feature_weights->deltas((*sparse_features.indices)(k)) +=
            feature_value * normalized_bounded_dual_delta;
      }
    }

    // Dense weights.
    for (size_t j = 0; j < dense_weights_.size(); ++j) {
      const Example::DenseVector& dense_vector = *example.dense_vectors_[j];
      TTypes<float>::Vec deltas = dense_weights_[j].deltas;
      deltas.device(device) =
          deltas +
          dense_vector.row() * deltas.constant(normalized_bounded_dual_delta);
    }
  }

  Status Initialize(OpKernelContext* const context) {
    OpInputList sparse_weights_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("sparse_weights", &sparse_weights_inputs));
    OpInputList dense_weights_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("dense_weights", &dense_weights_inputs));

    OpOutputList sparse_weights_outputs;
    TF_RETURN_IF_ERROR(context->output_list("out_delta_sparse_weights",
                                            &sparse_weights_outputs));

    OpOutputList dense_weights_outputs;
    TF_RETURN_IF_ERROR(context->output_list("out_delta_dense_weights",
                                            &dense_weights_outputs));

    // Reads in the weights, and allocates and initializes the delta weights.
    const auto intialize_weights = [&](
        const OpInputList& weight_inputs, OpOutputList* const weight_outputs,
        std::vector<FeatureWeights>* const feature_weights) {
      for (int i = 0; i < weight_inputs.size(); ++i) {
        Tensor* delta_t;
        weight_outputs->allocate(i, weight_inputs[i].shape(), &delta_t);
        auto deltas = delta_t->flat<float>();
        deltas.setZero();
        feature_weights->emplace_back(
            FeatureWeights{weight_inputs[i].flat<float>(), deltas});
      }
    };

    intialize_weights(sparse_weights_inputs, &sparse_weights_outputs,
                      &sparse_weights_);
    intialize_weights(dense_weights_inputs, &dense_weights_outputs,
                      &dense_weights_);

    return Status::OK();
  }

  const std::vector<FeatureWeights>& sparse_weights() const {
    return sparse_weights_;
  }

  const std::vector<FeatureWeights>& dense_weights() const {
    return dense_weights_;
  }

 private:
  // TODO(sibyl-Aix6ihai): Refactor this to support both small-batch mode, and large
  // batch mode, where we use sparse storage (hashmap) vs dense storage
  // (vectors).
  std::vector<FeatureWeights> sparse_weights_;
  std::vector<FeatureWeights> dense_weights_;

  TF_DISALLOW_COPY_AND_ASSIGN(ModelWeights);
};

// Computes the example statistics for given example, and model. Defined here
// as we need definition of ModelWeights and Regularizations.
const ExampleStatistics Example::ComputeWxAndWeightedExampleNorm(
    const int num_loss_partitions, const ModelWeights& model_weights,
    const Regularizations& regularization) const {
  ExampleStatistics result;

  result.normalized_squared_norm =
      squared_norm_ / regularization.symmetric_l2();

  // Compute the w \dot x and prev_w \dot x.

  // Sparse features contribution.
  for (size_t j = 0; j < sparse_features_.size(); ++j) {
    const Example::SparseFeatures& sparse_features = sparse_features_[j];
    const FeatureWeights& sparse_weights = model_weights.sparse_weights()[j];

    for (int64 k = 0; k < sparse_features.indices->size(); ++k) {
      const int64 feature_index = (*sparse_features.indices)(k);
      const double feature_value = sparse_features.values == nullptr
                                       ? 1.0
                                       : (*sparse_features.values)(k);
      const double feature_weight =
          sparse_weights.nominals(feature_index) +
          sparse_weights.deltas(feature_index) * num_loss_partitions;
      result.prev_wx +=
          feature_value *
          regularization.Shrink(sparse_weights.nominals(feature_index));
      result.wx += feature_value * regularization.Shrink(feature_weight);
    }
  }

  // Dense features contribution.
  for (size_t j = 0; j < dense_vectors_.size(); ++j) {
    const Example::DenseVector& dense_vector = *dense_vectors_[j];
    const FeatureWeights& dense_weights = model_weights.dense_weights()[j];

    const Eigen::Tensor<float, 1, Eigen::RowMajor> feature_weights =
        dense_weights.nominals +
        dense_weights.deltas *
            dense_weights.deltas.constant(num_loss_partitions);
    const Eigen::Tensor<float, 0, Eigen::RowMajor> prev_prediction =
        (dense_vector.row() *
         regularization.EigenShrink(dense_weights.nominals))
            .sum();
    const Eigen::Tensor<float, 0, Eigen::RowMajor> prediction =
        (dense_vector.row() * regularization.EigenShrink(feature_weights))
            .sum();
    result.prev_wx += prev_prediction();
    result.wx += prediction();
  }

  return result;
}

// Examples contains all the training examples that SDCA uses for a mini-batch.
class Examples {
 public:
  Examples() {}

  // Returns the Example at |example_index|.
  const Example& example(const int example_index) const {
    return examples_.at(example_index);
  }

  int sampled_index(const int id, const bool adaptative) const {
    if (adaptative) return sampled_index_[id];
    return id;
  }

  void SampleAdaptativeProbabilities(
      const int num_partitions, const Regularizations& regularization,
      const ModelWeights& model_weights,
      const TTypes<float>::Matrix example_state_data,
      const std::unique_ptr<DualLossUpdater>& loss_updater) {
    // Compute the probabilities
    for (int example_id = 0; example_id < num_examples(); ++example_id) {
      const Example& example = examples_[example_id];
      const double example_weight = example.example_weight();
      float label = example.example_label();
      const Status conversion_status = loss_updater->ConvertLabel(&label);
      const ExampleStatistics example_statistics =
          example.ComputeWxAndWeightedExampleNorm(num_partitions, model_weights,
                                                  regularization);
      const double kappa = example_state_data(example_id, 0) +
                           loss_updater->PrimalLossDerivative(
                               example_statistics.wx, label, example_weight);
      probabilities_[example_id] =
          example_weight * sqrt(examples_[example_id].squared_norm_ +
                                regularization.symmetric_l2() *
                                    loss_updater->SmoothnessConstant()) *
          std::abs(kappa);
    }

    // Sample the index
    random::DistributionSampler sampler(probabilities_);
    GuardedPhiloxRandom generator;
    generator.Init(0, 0);
    auto local_gen = generator.ReserveSamples32(num_examples());
    random::SimplePhilox random(&local_gen);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // We use a decay of 10: the probability of an example is divided by 10
    // once that example is picked. A good approximation of that is to only
    // keep a picked example with probability (1 / 10) ^ k where k is the
    // number of times we already picked that example. We add a num_retries
    // to avoid taking too long to sample. We then fill the sampled_index with
    // unseen examples sorted by probabilities.
    int id = 0;
    int num_retries = 0;
    while (id < num_examples() && num_retries < num_examples()) {
      int picked_id = sampler.Sample(&random);
      if (dis(gen) > std::pow(0.1, sampled_count_[picked_id])) {
        num_retries++;
        continue;
      }
      sampled_count_[picked_id]++;
      sampled_index_[id++] = picked_id;
    }

    std::vector<std::pair<int, float>> examples_not_seen;
    examples_not_seen.reserve(num_examples());
    for (int i = 0; i < num_examples(); ++i) {
      if (sampled_count_[i] == 0)
        examples_not_seen.emplace_back(sampled_index_[i], probabilities_[i]);
    }
    std::sort(
        examples_not_seen.begin(), examples_not_seen.end(),
        [](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
          return lhs.second > rhs.second;
        });
    for (int i = id; i < num_examples(); ++i) {
      sampled_count_[i] = examples_not_seen[i - id].first;
    }
  }

  int num_examples() const { return examples_.size(); }

  int num_features() const { return num_features_; }

  // Initialize() must be called immediately after construction.
  // TODO(sibyl-Aix6ihai): Refactor/shorten this function.
  Status Initialize(OpKernelContext* const context, const ModelWeights& weights,
                    int num_sparse_features,
                    int num_sparse_features_with_values,
                    int num_dense_features);

 private:
  // Reads the input tensors, and builds the internal representation for sparse
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateSparseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, const ModelWeights& weights,
      const OpInputList& sparse_example_indices_inputs,
      const OpInputList& sparse_feature_indices_inputs,
      const OpInputList& sparse_feature_values_inputs,
      std::vector<Example>* const examples);

  // Reads the input tensors, and builds the internal representation for dense
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateDenseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_dense_features, const ModelWeights& weights,
      const OpInputList& dense_features_inputs,
      std::vector<Example>* const examples);

  // Computes squared example norm per example i.e |x|^2. This function modifies
  // the |examples| passed in and adds the squared norm per example.
  static void ComputeSquaredNormPerExample(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, int num_dense_features,
      std::vector<Example>* const examples);

  // All examples in the batch.
  std::vector<Example> examples_;

  // Adaptative sampling variables
  std::vector<float> probabilities_;
  std::vector<int> sampled_index_;
  std::vector<int> sampled_count_;

  int num_features_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Examples);
};

Status Examples::Initialize(OpKernelContext* const context,
                            const ModelWeights& weights,
                            const int num_sparse_features,
                            const int num_sparse_features_with_values,
                            const int num_dense_features) {
  num_features_ = num_sparse_features + num_dense_features;

  OpInputList sparse_example_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_example_indices",
                                         &sparse_example_indices_inputs));
  OpInputList sparse_feature_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_feature_indices",
                                         &sparse_feature_indices_inputs));
  OpInputList sparse_feature_values_inputs;
  if (num_sparse_features_with_values > 0) {
    TF_RETURN_IF_ERROR(context->input_list("sparse_feature_values",
                                           &sparse_feature_values_inputs));
  }

  const Tensor* example_weights_t;
  TF_RETURN_IF_ERROR(context->input("example_weights", &example_weights_t));
  auto example_weights = example_weights_t->flat<float>();

  if (example_weights.size() >= std::numeric_limits<int>::max()) {
    return errors::InvalidArgument(strings::Printf(
        "Too many examples in a mini-batch: %ld > %d", example_weights.size(),
        std::numeric_limits<int>::max()));
  }

  // The static_cast here is safe since num_examples can be at max an int.
  const int num_examples = static_cast<int>(example_weights.size());
  const Tensor* example_labels_t;
  TF_RETURN_IF_ERROR(context->input("example_labels", &example_labels_t));
  auto example_labels = example_labels_t->flat<float>();

  OpInputList dense_features_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("dense_features", &dense_features_inputs));

  examples_.clear();
  examples_.resize(num_examples);
  probabilities_.resize(num_examples);
  sampled_index_.resize(num_examples);
  sampled_count_.resize(num_examples);
  for (int example_id = 0; example_id < num_examples; ++example_id) {
    Example* const example = &examples_[example_id];
    example->sparse_features_.resize(num_sparse_features);
    example->dense_vectors_.resize(num_dense_features);
    example->example_weight_ = example_weights(example_id);
    example->example_label_ = example_labels(example_id);
  }
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();
  TF_RETURN_IF_ERROR(CreateSparseFeatureRepresentation(
      worker_threads, num_examples, num_sparse_features, weights,
      sparse_example_indices_inputs, sparse_feature_indices_inputs,
      sparse_feature_values_inputs, &examples_));
  TF_RETURN_IF_ERROR(CreateDenseFeatureRepresentation(
      worker_threads, num_examples, num_dense_features, weights,
      dense_features_inputs, &examples_));
  ComputeSquaredNormPerExample(worker_threads, num_examples,
                               num_sparse_features, num_dense_features,
                               &examples_);
  return Status::OK();
}

Status Examples::CreateSparseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_sparse_features, const ModelWeights& weights,
    const OpInputList& sparse_example_indices_inputs,
    const OpInputList& sparse_feature_indices_inputs,
    const OpInputList& sparse_feature_values_inputs,
    std::vector<Example>* const examples) {
  mutex mu;
  Status result GUARDED_BY(mu);
  auto parse_partition = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto example_indices =
          sparse_example_indices_inputs[i].template flat<int64>();
      auto feature_indices =
          sparse_feature_indices_inputs[i].template flat<int64>();

      // Parse features for each example. Features for a particular example
      // are at the offsets (start_id, end_id]
      int start_id = -1;
      int end_id = 0;
      for (int example_id = 0; example_id < num_examples; ++example_id) {
        start_id = end_id;
        while (end_id < example_indices.size() &&
               example_indices(end_id) == example_id) {
          ++end_id;
        }
        Example::SparseFeatures* const sparse_features =
            &(*examples)[example_id].sparse_features_[i];
        if (start_id < example_indices.size() &&
            example_indices(start_id) == example_id) {
          sparse_features->indices.reset(new UnalignedInt64Vector(
              &(feature_indices(start_id)), end_id - start_id));
          if (sparse_feature_values_inputs.size() > i) {
            auto feature_weights =
                sparse_feature_values_inputs[i].flat<float>();
            sparse_features->values.reset(new UnalignedFloatVector(
                &(feature_weights(start_id)), end_id - start_id));
          }
          // If features are non empty.
          if (end_id - start_id > 0) {
            // TODO(sibyl-Aix6ihai): Write this efficiently using vectorized
            // operations from eigen.
            for (int64 k = 0; k < sparse_features->indices->size(); ++k) {
              const int64 feature_index = (*sparse_features->indices)(k);
              if (feature_index < 0 ||
                  feature_index >=
                      weights.sparse_weights()[i].nominals.size()) {
                mutex_lock l(mu);
                result = errors::InvalidArgument(
                    "Found sparse feature indices out of valid range: ",
                    (*sparse_features->indices)(k));
                return;
              }
            }
          }
        } else {
          // Add a Tensor that has size 0.
          sparse_features->indices.reset(
              new UnalignedInt64Vector(&(feature_indices(0)), 0));
          // If values exist for this feature group.
          if (sparse_feature_values_inputs.size() > i) {
            auto feature_weights =
                sparse_feature_values_inputs[i].flat<float>();
            sparse_features->values.reset(
                new UnalignedFloatVector(&(feature_weights(0)), 0));
          }
        }
      }
    }
  };
  // For each column, the cost of parsing it is O(num_examples). We use
  // num_examples here, as empirically Shard() creates the right amount of
  // threads based on the problem size.
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64 kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_sparse_features,
        kCostPerUnit, parse_partition);
  return result;
}

Status Examples::CreateDenseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_dense_features, const ModelWeights& weights,
    const OpInputList& dense_features_inputs,
    std::vector<Example>* const examples) {
  mutex mu;
  Status result GUARDED_BY(mu);
  auto parse_partition = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto dense_features = dense_features_inputs[i].template matrix<float>();
      for (int example_id = 0; example_id < num_examples; ++example_id) {
        (*examples)[example_id].dense_vectors_[i].reset(
            new Example::DenseVector{dense_features, example_id});
      }
      if (dense_features.dimension(1) !=
          weights.dense_weights()[i].nominals.size()) {
        mutex_lock l(mu);
        result = errors::InvalidArgument(
            "More dense features than we have parameters for: ",
            dense_features.dimension(1));
        return;
      }
    }

  };
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64 kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_dense_features,
        kCostPerUnit, parse_partition);
  return result;
}

void Examples::ComputeSquaredNormPerExample(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_sparse_features, const int num_dense_features,
    std::vector<Example>* const examples) {
  // Compute norm of examples.
  auto compute_example_norm = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int example_id = static_cast<int>(begin); example_id < end;
         ++example_id) {
      double squared_norm = 0;
      Example* const example = &(*examples)[example_id];
      for (int j = 0; j < num_sparse_features; ++j) {
        const Example::SparseFeatures& sparse_features =
            example->sparse_features_[j];
        if (sparse_features.values) {
          const Eigen::Tensor<float, 0, Eigen::RowMajor> sn =
              sparse_features.values->square().sum();
          squared_norm += sn();
        } else {
          squared_norm += sparse_features.indices->size();
        }
      }
      for (int j = 0; j < num_dense_features; ++j) {
        const Eigen::Tensor<float, 0, Eigen::RowMajor> sn =
            example->dense_vectors_[j]->row().square().sum();
        squared_norm += sn();
      }
      example->squared_norm_ = squared_norm;
    }
  };
  // TODO(sibyl-Aix6ihai): Compute the cost optimally.
  const int64 kCostPerUnit = num_dense_features + num_sparse_features;
  Shard(worker_threads.num_threads, worker_threads.workers, num_examples,
        kCostPerUnit, compute_example_norm);
}

}  // namespace

class DistributedSdcaLargeBatchSolver : public OpKernel {
 public:
  explicit DistributedSdcaLargeBatchSolver(OpKernelConstruction* const context)
      : OpKernel(context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      loss_updater_.reset(new LogisticLossUpdater);
    } else if (loss_type == "squared_loss") {
      loss_updater_.reset(new SquaredLossUpdater);
    } else if (loss_type == "hinge_loss") {
      loss_updater_.reset(new HingeLossUpdater);
    } else if (loss_type == "smooth_hinge_loss") {
      loss_updater_.reset(new SmoothHingeLossUpdater);
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }
    OP_REQUIRES_OK(context, context->GetAttr("adaptative", &adaptative_));
    OP_REQUIRES_OK(context, context->GetAttr("num_sparse_features",
                                             &num_sparse_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_sparse_features_with_values",
                                    &num_sparse_features_with_values_));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_dense_features", &num_dense_features_));
    OP_REQUIRES(
        context, num_sparse_features_ + num_dense_features_ > 0,
        errors::InvalidArgument("Requires at least one feature to train."));

    OP_REQUIRES(context, static_cast<int64>(num_sparse_features_) +
                                 static_cast<int64>(num_dense_features_) <=
                             std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    strings::Printf("Too many feature groups: %lld > %d",
                                    static_cast<int64>(num_sparse_features_) +
                                        static_cast<int64>(num_dense_features_),
                                    std::numeric_limits<int>::max())));
    OP_REQUIRES_OK(context, context->GetAttr("num_loss_partitions",
                                             &num_loss_partitions_));
    OP_REQUIRES_OK(context, context->GetAttr("num_inner_iterations",
                                             &num_inner_iterations_));
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
  }

  // TODO(sibyl-Aix6ihai): Refactor/shorten this function.
  void Compute(OpKernelContext* const context) override {
    ModelWeights model_weights;
    OP_REQUIRES_OK(context, model_weights.Initialize(context));

    Examples examples;
    OP_REQUIRES_OK(context,
                   examples.Initialize(
                       context, model_weights, num_sparse_features_,
                       num_sparse_features_with_values_, num_dense_features_));

    const Tensor* example_state_data_t;
    OP_REQUIRES_OK(context,
                   context->input("example_state_data", &example_state_data_t));
    TensorShape expected_example_state_shape({examples.num_examples(), 4});
    OP_REQUIRES(
        context, example_state_data_t->shape() == expected_example_state_shape,
        errors::InvalidArgument("Expected shape ",
                                expected_example_state_shape.DebugString(),
                                " for example_state_data, got ",
                                example_state_data_t->shape().DebugString()));

    Tensor mutable_example_state_data_t(*example_state_data_t);
    auto example_state_data = mutable_example_state_data_t.matrix<float>();
    context->set_output("out_example_state_data", mutable_example_state_data_t);

    if (adaptative_) {
      examples.SampleAdaptativeProbabilities(num_loss_partitions_,
                                             regularizations_, model_weights,
                                             example_state_data, loss_updater_);
    }

    mutex mu;
    Status train_step_status GUARDED_BY(mu);
    std::atomic<std::int64_t> atomic_index(-1);
    auto train_step = [&, this](const int64 begin, const int64 end) {
      // The static_cast here is safe since begin and end can be at most
      // num_examples which is an int.
      for (int id = static_cast<int>(begin); id < end; ++id) {
        const int64 example_index =
            examples.sampled_index(++atomic_index, adaptative_);
        const Example& example = examples.example(example_index);
        const float dual = example_state_data(example_index, 0);
        const float example_weight = example.example_weight();
        float example_label = example.example_label();
        const Status conversion_status =
            loss_updater_->ConvertLabel(&example_label);
        if (!conversion_status.ok()) {
          mutex_lock l(mu);
          train_step_status = conversion_status;
          // Return from this worker thread - the calling thread is
          // responsible for checking context status and returning on error.
          return;
        }

        // Compute wx, example norm weighted by regularization, dual loss,
        // primal loss.
        const ExampleStatistics example_statistics =
            example.ComputeWxAndWeightedExampleNorm(
                num_loss_partitions_, model_weights, regularizations_);

        const double new_dual = loss_updater_->ComputeUpdatedDual(
            num_loss_partitions_, example_label, example_weight, dual,
            example_statistics.wx, example_statistics.normalized_squared_norm);

        // Compute new weights.
        const double normalized_bounded_dual_delta =
            (new_dual - dual) * example_weight /
            regularizations_.symmetric_l2();
        model_weights.UpdateDeltaWeights(context->eigen_cpu_device(), example,
                                         normalized_bounded_dual_delta);

        // Update example data.
        example_state_data(example_index, 0) = new_dual;
        example_state_data(example_index, 1) = loss_updater_->ComputePrimalLoss(
            example_statistics.prev_wx, example_label, example_weight);
        example_state_data(example_index, 2) =
            loss_updater_->ComputeDualLoss(dual, example_label, example_weight);
        example_state_data(example_index, 3) = example_weight;
      }
    };
    // TODO(sibyl-Aix6ihai): Tune this properly based on sparsity of the data,
    // number of cpus, and cost per example.
    const int64 kCostPerUnit = examples.num_features();
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();

    Shard(worker_threads.num_threads, worker_threads.workers,
          examples.num_examples(), kCostPerUnit, train_step);
    OP_REQUIRES_OK(context, train_step_status);
  }

 private:
  // TODO(sibyl-Aix6ihai): We could use the type-constraint on loss_type, and
  // template the entire class to avoid the virtual table lookup penalty in
  // the inner loop.
  std::unique_ptr<DualLossUpdater> loss_updater_;
  int num_sparse_features_ = 0;
  int num_sparse_features_with_values_ = 0;
  int num_dense_features_ = 0;
  int num_inner_iterations_ = 0;
  int num_loss_partitions_ = 0;
  bool adaptative_;
  Regularizations regularizations_;
};
REGISTER_KERNEL_BUILDER(
    Name("DistributedSdcaLargeBatchSolver").Device(DEVICE_CPU),
    DistributedSdcaLargeBatchSolver);

class SdcaShrinkL1 : public OpKernel {
 public:
  explicit SdcaShrinkL1(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
  }

  void Compute(OpKernelContext* const context) override {
    OpMutableInputList weights_inputs;
    OP_REQUIRES_OK(context,
                   context->mutable_input_list("weights", &weights_inputs));

    auto do_work = [&](const int64 begin, const int64 end) {
      for (int i = begin; i < end; ++i) {
        auto prox_w = weights_inputs.at(i, /*lock_held=*/true).flat<float>();
        prox_w.device(context->eigen_cpu_device()) =
            regularizations_.EigenShrink(prox_w);
      }
    };

    if (weights_inputs.size() > 0) {
      int64 num_weights = 0;
      for (int i = 0; i < weights_inputs.size(); ++i) {
        num_weights += weights_inputs.at(i, /*lock_held=*/true).NumElements();
      }
      // TODO(sibyl-Aix6ihai): Tune this value.
      const int64 kCostPerUnit = (num_weights * 50) / weights_inputs.size();
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *context->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers,
            weights_inputs.size(), kCostPerUnit, do_work);
    }
  }

 private:
  Regularizations regularizations_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaShrinkL1").Device(DEVICE_CPU), SdcaShrinkL1);

// Computes platform independent, compact and unique (with very high
// probability) representation of an example id. It shouldn't be put in
// persistent storage, as its implementation may change in the future.
//
// The current probability of at least one collision for 1B example_ids is
// approximately 10^-11 (ie 2^60 / 2^97).
class SdcaFprint : public OpKernel {
 public:
  explicit SdcaFprint(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    const Tensor& input = context->input(0);
    Tensor* out;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &out));

    const auto in_values = input.flat<string>();
    auto out_values = out->flat<string>();

    for (int64 i = 0; i < in_values.size(); ++i) {
      out_values(i) = Fp128ToBinaryString(Fingerprint128(in_values(i)));
    }
  }

 private:
  // Returns a 12 character binary string of the fprint.
  // We use 12 of the 16 fingerprint bytes to save memory, in particular in
  // string implementations that use a short string optimization.
  static string Fp128ToBinaryString(const Fprint128& fprint) {
    string result;
    core::PutFixed64(&result, fprint.low64);
    core::PutFixed32(&result, fprint.high64);
    return result;
  }
};
REGISTER_KERNEL_BUILDER(Name("SdcaFprint").Device(DEVICE_CPU), SdcaFprint);

}  // namespace tensorflow
