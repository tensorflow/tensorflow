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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sdca_internal.h"

#include <limits>
#include <random>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace sdca {

using UnalignedFloatVector = TTypes<const float>::UnalignedConstVec;
using UnalignedInt64Vector = TTypes<const int64>::UnalignedConstVec;

void FeatureWeightsDenseStorage::UpdateDenseDeltaWeights(
    const Eigen::ThreadPoolDevice& device,
    const Example::DenseVector& dense_vector,
    const std::vector<double>& normalized_bounded_dual_delta) {
  const size_t num_weight_vectors = normalized_bounded_dual_delta.size();
  if (num_weight_vectors == 1) {
    deltas_.device(device) =
        deltas_ + dense_vector.RowAsMatrix() *
                      deltas_.constant(normalized_bounded_dual_delta[0]);
  } else {
    // Transform the dual vector into a column matrix.
    const Eigen::TensorMap<Eigen::Tensor<const double, 2, Eigen::RowMajor>>
        dual_matrix(normalized_bounded_dual_delta.data(), num_weight_vectors,
                    1);
    const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    // This computes delta_w += delta_vector / \lamdba * N.
    deltas_.device(device) =
        (deltas_.cast<double>() +
         dual_matrix.contract(dense_vector.RowAsMatrix().cast<double>(),
                              product_dims))
            .cast<float>();
  }
}

void FeatureWeightsSparseStorage::UpdateSparseDeltaWeights(
    const Eigen::ThreadPoolDevice& device,
    const Example::SparseFeatures& sparse_features,
    const std::vector<double>& normalized_bounded_dual_delta) {
  for (int64 k = 0; k < sparse_features.indices->size(); ++k) {
    const double feature_value =
        sparse_features.values == nullptr ? 1.0 : (*sparse_features.values)(k);
    auto it = indices_to_id_.find((*sparse_features.indices)(k));
    for (size_t l = 0; l < normalized_bounded_dual_delta.size(); ++l) {
      deltas_(l, it->second) +=
          feature_value * normalized_bounded_dual_delta[l];
    }
  }
}

void ModelWeights::UpdateDeltaWeights(
    const Eigen::ThreadPoolDevice& device, const Example& example,
    const std::vector<double>& normalized_bounded_dual_delta) {
  // Sparse weights.
  for (size_t j = 0; j < sparse_weights_.size(); ++j) {
    sparse_weights_[j].UpdateSparseDeltaWeights(
        device, example.sparse_features_[j], normalized_bounded_dual_delta);
  }

  // Dense weights.
  for (size_t j = 0; j < dense_weights_.size(); ++j) {
    dense_weights_[j].UpdateDenseDeltaWeights(
        device, *example.dense_vectors_[j], normalized_bounded_dual_delta);
  }
}

Status ModelWeights::Initialize(OpKernelContext* const context) {
  OpInputList sparse_indices_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("sparse_indices", &sparse_indices_inputs));
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
  TF_RETURN_IF_ERROR(
      context->output_list("out_delta_dense_weights", &dense_weights_outputs));

  for (int i = 0; i < sparse_weights_inputs.size(); ++i) {
    Tensor* delta_t;
    TF_RETURN_IF_ERROR(sparse_weights_outputs.allocate(
        i, sparse_weights_inputs[i].shape(), &delta_t));
    // Convert the input vector to a row matrix in internal representation.
    auto deltas = delta_t->shaped<float, 2>({1, delta_t->NumElements()});
    deltas.setZero();
    sparse_weights_.emplace_back(FeatureWeightsSparseStorage{
        sparse_indices_inputs[i].flat<int64>(),
        sparse_weights_inputs[i].shaped<float, 2>(
            {1, sparse_weights_inputs[i].NumElements()}),
        deltas});
  }

  // Reads in the weights, and allocates and initializes the delta weights.
  const auto initialize_weights =
      [&](const OpInputList& weight_inputs, OpOutputList* const weight_outputs,
          std::vector<FeatureWeightsDenseStorage>* const feature_weights) {
        for (int i = 0; i < weight_inputs.size(); ++i) {
          Tensor* delta_t;
          TF_RETURN_IF_ERROR(
              weight_outputs->allocate(i, weight_inputs[i].shape(), &delta_t));
          // Convert the input vector to a row matrix in internal
          // representation.
          auto deltas = delta_t->shaped<float, 2>({1, delta_t->NumElements()});
          deltas.setZero();
          feature_weights->emplace_back(FeatureWeightsDenseStorage{
              weight_inputs[i].shaped<float, 2>(
                  {1, weight_inputs[i].NumElements()}),
              deltas});
        }
        return Status::OK();
      };

  return initialize_weights(dense_weights_inputs, &dense_weights_outputs,
                            &dense_weights_);
}

// Computes the example statistics for given example, and model. Defined here
// as we need definition of ModelWeights and Regularizations.
const ExampleStatistics Example::ComputeWxAndWeightedExampleNorm(
    const int num_loss_partitions, const ModelWeights& model_weights,
    const Regularizations& regularization, const int num_weight_vectors) const {
  ExampleStatistics result(num_weight_vectors);

  result.normalized_squared_norm =
      squared_norm_ / regularization.symmetric_l2();

  // Compute w \dot x and prev_w \dot x.
  // This is for sparse features contribution to the logit.
  for (size_t j = 0; j < sparse_features_.size(); ++j) {
    const Example::SparseFeatures& sparse_features = sparse_features_[j];
    const FeatureWeightsSparseStorage& sparse_weights =
        model_weights.sparse_weights()[j];

    for (int64 k = 0; k < sparse_features.indices->size(); ++k) {
      const int64 feature_index = (*sparse_features.indices)(k);
      const double feature_value = sparse_features.values == nullptr
                                       ? 1.0
                                       : (*sparse_features.values)(k);
      for (int l = 0; l < num_weight_vectors; ++l) {
        const float sparse_weight = sparse_weights.nominals(l, feature_index);
        const double feature_weight =
            sparse_weight +
            sparse_weights.deltas(l, feature_index) * num_loss_partitions;
        result.prev_wx[l] +=
            feature_value * regularization.Shrink(sparse_weight);
        result.wx[l] += feature_value * regularization.Shrink(feature_weight);
      }
    }
  }

  // Compute w \dot x and prev_w \dot x.
  // This is for dense features contribution to the logit.
  for (size_t j = 0; j < dense_vectors_.size(); ++j) {
    const Example::DenseVector& dense_vector = *dense_vectors_[j];
    const FeatureWeightsDenseStorage& dense_weights =
        model_weights.dense_weights()[j];

    const Eigen::Tensor<float, 2, Eigen::RowMajor> feature_weights =
        dense_weights.nominals() +
        dense_weights.deltas() *
            dense_weights.deltas().constant(num_loss_partitions);
    if (num_weight_vectors == 1) {
      const Eigen::Tensor<float, 0, Eigen::RowMajor> prev_prediction =
          (dense_vector.Row() *
           regularization.EigenShrinkVector(
               Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
                   dense_weights.nominals().data(),
                   dense_weights.nominals().dimension(1))))
              .sum();
      const Eigen::Tensor<float, 0, Eigen::RowMajor> prediction =
          (dense_vector.Row() *
           regularization.EigenShrinkVector(
               Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
                   feature_weights.data(), feature_weights.dimension(1))))
              .sum();
      result.prev_wx[0] += prev_prediction();
      result.wx[0] += prediction();
    } else {
      const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 1)};
      const Eigen::Tensor<float, 2, Eigen::RowMajor> prev_prediction =
          regularization.EigenShrinkMatrix(dense_weights.nominals())
              .contract(dense_vector.RowAsMatrix(), product_dims);
      const Eigen::Tensor<float, 2, Eigen::RowMajor> prediction =
          regularization.EigenShrinkMatrix(feature_weights)
              .contract(dense_vector.RowAsMatrix(), product_dims);
      // The result of "tensor contraction" (multiplication)  in the code
      // above is of dimension num_weight_vectors * 1.
      for (int l = 0; l < num_weight_vectors; ++l) {
        result.prev_wx[l] += prev_prediction(l, 0);
        result.wx[l] += prediction(l, 0);
      }
    }
  }

  return result;
}

// Examples contains all the training examples that SDCA uses for a mini-batch.
Status Examples::SampleAdaptiveProbabilities(
    const int num_loss_partitions, const Regularizations& regularization,
    const ModelWeights& model_weights,
    const TTypes<float>::Matrix example_state_data,
    const std::unique_ptr<DualLossUpdater>& loss_updater,
    const int num_weight_vectors) {
  if (num_weight_vectors != 1) {
    return errors::InvalidArgument(
        "Adaptive SDCA only works with binary SDCA, "
        "where num_weight_vectors should be 1.");
  }
  // Compute the probabilities
  for (int example_id = 0; example_id < num_examples(); ++example_id) {
    const Example& example = examples_[example_id];
    const double example_weight = example.example_weight();
    float label = example.example_label();
    const Status conversion_status = loss_updater->ConvertLabel(&label);
    const ExampleStatistics example_statistics =
        example.ComputeWxAndWeightedExampleNorm(num_loss_partitions,
                                                model_weights, regularization,
                                                num_weight_vectors);
    const double kappa = example_state_data(example_id, 0) +
                         loss_updater->PrimalLossDerivative(
                             example_statistics.wx[0], label, example_weight);
    probabilities_[example_id] = example_weight *
                                 sqrt(examples_[example_id].squared_norm_ +
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
    if (dis(gen) > MathUtil::IPow(0.1, sampled_count_[picked_id])) {
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
  return Status::OK();
}

void Examples::RandomShuffle() {
  std::iota(sampled_index_.begin(), sampled_index_.end(), 0);
  std::random_shuffle(sampled_index_.begin(), sampled_index_.end());
}

// TODO(sibyl-Aix6ihai): Refactor/shorten this function.
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
        "Too many examples in a mini-batch: %zu > %d", example_weights.size(),
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
              if (!weights.SparseIndexValid(i, feature_index)) {
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
      if (!weights.DenseIndexValid(i, dense_features.dimension(1) - 1)) {
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
            example->dense_vectors_[j]->Row().square().sum();
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

}  // namespace sdca
}  // namespace tensorflow
