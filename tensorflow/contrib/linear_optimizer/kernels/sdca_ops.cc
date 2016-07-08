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

// See docs in ../ops/sdca_ops.cc.

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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

struct PerExampleData {
  // feature_weights dot feature_values for the example
  double wx = 0;
  // sum of squared feature values occurring in the example divided by
  // L2 * sum(example_weights).
  double normalized_squared_norm = 0;
};

PerExampleData AddPerExampleData(const PerExampleData& data1,
                                 const PerExampleData& data2) {
  PerExampleData result;
  result.wx = data1.wx + data2.wx;
  result.normalized_squared_norm =
      data1.normalized_squared_norm + data2.normalized_squared_norm;
  return result;
}

class Regularizations {
 public:
  Regularizations(){};

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelConstruction* const context) {
    TF_RETURN_IF_ERROR(context->GetAttr("l1", &symmetric_l1_));
    TF_RETURN_IF_ERROR(context->GetAttr("l2", &symmetric_l2_));
    shrinkage_factor_ = symmetric_l1_ / symmetric_l2_;
    return Status::OK();
  }

  // Proximal SDCA shrinking for L1 regularization.
  double Shrink(const double weight) const {
    const double shrink_weight =
        std::max(std::abs(weight) - shrinkage_factor_, 0.0);
    if (shrink_weight > 0.0) {
      return std::copysign(shrink_weight, weight);
    }
    return 0.0;
  }

  float symmetric_l2() const { return symmetric_l2_; }

 private:
  float symmetric_l1_ = 0;
  float symmetric_l2_ = 0;

  // L1 divided by L2, precomputed for use during weight shrinking.
  double shrinkage_factor_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Regularizations);
};

// Tracks feature weights for groups of features which are input as lists
// of weight tensors.  Each list element becomes a "group" allowing us to
// refer to an individual feature by [group_num][feature_num].
class WeightsByGroup {
 public:
  WeightsByGroup(){};

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context,
                    const string& input_list_name) {
    OpMutableInputList weights_inputs;
    TF_RETURN_IF_ERROR(
        context->mutable_input_list(input_list_name, &weights_inputs));
    for (int i = 0; i < weights_inputs.size(); ++i) {
      weights_by_group_.emplace_back(
          weights_inputs.at(i, /*lock_held=*/true).flat<float>());
    }

    return Status::OK();
  }

  // Adds the given 'delta' to the feature indexed by 'group' and 'feature'.
  void AddDelta(const size_t group, const size_t feature, const float delta) {
    weights_by_group_[group](feature) += delta;
  }

  // Modifies all weights according to the shrinkage factor determined by
  // 'regularizations'.
  void Shrink(const Regularizations& regularizations) {
    for (TTypes<float>::Vec weights : weights_by_group_) {
      for (int64 i = 0; i < weights.size(); ++i) {
        weights(i) = regularizations.Shrink(weights(i));
      }
    }
  }

  // Returns an error if these weights do not appear to come from dense
  // features.  Currently this means that each group contains a single feature.
  // TODO(sibyl-Mooth6ku): Support arbitrary dimensional dense weights and remove
  // this.
  Status ValidateAsDense() const {
    for (const TTypes<float>::Vec weights : weights_by_group_) {
      if (weights.size() != 1) {
        return errors::InvalidArgument(strings::Printf(
            "Dense weight vectors should have exactly one entry. Found (%ld). "
            "This is probably due to a misconfiguration in the optimizer "
            "setup.",
            weights.size()));
      }
    }
    return Status::OK();
  }

  size_t NumGroups() const { return weights_by_group_.size(); }

  const TTypes<float>::Vec& WeightsOfGroup(const size_t group) const {
    return weights_by_group_[group];
  }

 private:
  // Weights associated with a (sparse or dense) feature group, such that the
  // size of weights_by_group_ is the number of feature groups.
  std::vector<TTypes<float>::Vec> weights_by_group_;

  TF_DISALLOW_COPY_AND_ASSIGN(WeightsByGroup);
};

// Tracks weights and delta-weights for either sparse or dense features.
// As we process a mini-batch, weights are read from tensors and delta-weights
// are initialized to 0.  During processing, delta-weights are modified and
// at the completion of processing the mini-batch, the delta-weights are added
// into the original weights and then discarded.
class WeightsAndDeltas {
 public:
  WeightsAndDeltas() {}

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context,
                    const string& input_list_name) {
    TF_RETURN_IF_ERROR(weights_by_group_.Initialize(context, input_list_name));
    InitializeDeltaWeightsToZero(
        *context->device()->tensorflow_cpu_worker_threads());
    return Status::OK();
  }

  // Adds all of the delta weights which were computed during processing
  // of this mini-batch into the feature-weights.  Must be called once
  // at the end of mini-batch processing.
  void AddDeltaWeights(const DeviceBase::CpuWorkerThreads& worker_threads) {
    Shard(worker_threads.num_threads, worker_threads.workers,
          static_cast<int64>(delta_weights_by_group_.size()),
          kCostPerUnitGroupWeightAccess,
          [this](const int64 begin, const int64 end) {
            for (int64 group = begin; group < end; ++group) {
              const std::vector<std::atomic<double>>& delta_weights =
                  delta_weights_by_group_[group];
              for (size_t i = 0; i < delta_weights.size(); ++i) {
                weights_by_group_.AddDelta(
                    group, i, static_cast<float>(delta_weights[i].load()));
              }
            }
          });
  }

  std::vector<std::atomic<double>>* DeltaWeightsOfGroup(const size_t group) {
    return &delta_weights_by_group_[group];
  }

  const std::vector<std::atomic<double>>& DeltaWeightsOfGroup(
      const size_t group) const {
    return delta_weights_by_group_[group];
  }

  const TTypes<float>::Vec& WeightsOfGroup(const size_t group) const {
    return weights_by_group_.WeightsOfGroup(group);
  }

  size_t NumGroups() const { return delta_weights_by_group_.size(); }

  size_t NumFeaturesOfGroup(const size_t group) const {
    return delta_weights_by_group_[group].size();
  }

  // Returns an error if these weights do not appear to come from dense
  // features.  Currently this means that each group contains a single feature.
  Status ValidateAsDense() const { return weights_by_group_.ValidateAsDense(); }

 private:
  void InitializeDeltaWeightsToZero(
      const DeviceBase::CpuWorkerThreads& worker_threads) {
    delta_weights_by_group_.resize(weights_by_group_.NumGroups());
    Shard(worker_threads.num_threads, worker_threads.workers,
          static_cast<int64>(weights_by_group_.NumGroups()),
          kCostPerUnitGroupWeightAccess,
          [this](const int64 begin, const int64 end) {
            for (int64 group = begin; group < end; ++group) {
              const TTypes<float>::Vec weights =
                  weights_by_group_.WeightsOfGroup(group);
              std::vector<std::atomic<double>>* delta_weights =
                  &delta_weights_by_group_[group];
              *delta_weights = std::vector<std::atomic<double>>(weights.size());
              std::fill(delta_weights->begin(), delta_weights->end(), 0);
            }
          });
  }

  // Assuming approximately 100K features per group with each feature having a
  // cost of 1 gives decent parallelization for weight related access (eg
  // traversals, initialization etc).
  static constexpr int64 kCostPerUnitGroupWeightAccess = 100000;

  WeightsByGroup weights_by_group_;

  // Delta weights associated with each of the weights in weights_by_group_,
  // indexed by [group_num][feature_num].  Atomicity is required when changing
  // the delta weights in order to have transactional updates.
  std::vector<std::vector<std::atomic<double>>> delta_weights_by_group_;

  TF_DISALLOW_COPY_AND_ASSIGN(WeightsAndDeltas);
};

// Atomically add a double to a std::atomic<double>.
inline void AtomicAdd(const double src, std::atomic<double>* const dst) {
  // We use a strong version of compare-exchange, as weak version can spuriously
  // fail.
  for (double c = dst->load(); !dst->compare_exchange_strong(c, c + src);) {
  }
}

// Tracks all of the information related to the dense features:  weights
// and delta weights, as well as feature occurrences in the current mini-batch.
class DenseFeaturesAndWeights {
 public:
  DenseFeaturesAndWeights() {}

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context) {
    OpInputList dense_features_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("dense_features", &dense_features_inputs));
    for (const auto& dense_feature : dense_features_inputs) {
      features_by_group_.emplace_back(dense_feature.vec<float>());
    }

    TF_RETURN_IF_ERROR(
        weights_and_deltas_.Initialize(context, "dense_weights"));
    TF_RETURN_IF_ERROR(weights_and_deltas_.ValidateAsDense());
    return Status::OK();
  }

  // Computes PerExampleData for 'example_id'.
  PerExampleData ComputeWxAndWeightedExampleNorm(
      const int64 example_id, const Regularizations& regularizations) const {
    PerExampleData result;
    for (size_t group = 0; group < features_by_group_.size(); ++group) {
      const double weight = weights_and_deltas_.WeightsOfGroup(group)(0);
      const std::atomic<double>& delta_weight =
          weights_and_deltas_.DeltaWeightsOfGroup(group)[0];
      const double value = features_by_group_[group](example_id);
      result.wx += regularizations.Shrink(weight + delta_weight.load()) * value;
      result.normalized_squared_norm += value * value;
    }
    result.normalized_squared_norm /= regularizations.symmetric_l2();
    return result;
  }

  // Updates the delta weight for each feature occuring in 'example_id',
  // given the weighted change in the dual for this example
  // (bounded_dual_delta), and the 'l2_regularization'.
  void UpdateDeltaWeights(const int64 example_id,
                          const double bounded_dual_delta,
                          const double l2_regularization) {
    for (size_t group = 0; group < features_by_group_.size(); ++group) {
      std::atomic<double>* const delta_weight =
          &(*weights_and_deltas_.DeltaWeightsOfGroup(group))[0];
      const double value = features_by_group_[group](example_id);
      AtomicAdd(bounded_dual_delta * value / l2_regularization, delta_weight);
    }
  }

  // Adds all of the delta weights which were computed during processing
  // of this mini-batch into the feature-weights.  Must be called once
  // at the end of mini-batch processing.
  void AddDeltaWeights(const DeviceBase::CpuWorkerThreads& worker_threads) {
    weights_and_deltas_.AddDeltaWeights(worker_threads);
  }

  size_t NumGroups() const { return features_by_group_.size(); }

 private:
  // Dense features associated with each dense feature group.
  std::vector<TTypes<const float>::Vec> features_by_group_;

  WeightsAndDeltas weights_and_deltas_;

  TF_DISALLOW_COPY_AND_ASSIGN(DenseFeaturesAndWeights);
};

// Tracks all of the information related to the sparse features:  weights
// and delta weights, as well as feature occurrences in the current mini-batch.
class SparseFeaturesAndWeights {
 public:
  SparseFeaturesAndWeights() {}

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context,
                    const int64 num_sparse_features, const int num_examples) {
    TF_RETURN_IF_ERROR(
        weights_and_deltas_.Initialize(context, "sparse_weights"));
    TF_RETURN_IF_ERROR(
        FillExamples(context, num_sparse_features, num_examples,
                     *context->device()->tensorflow_cpu_worker_threads()));
    return Status::OK();
  }

  // Computes PerExampleData for 'example_id'.
  PerExampleData ComputeWxAndWeightedExampleNorm(
      const int64 example_id, const Regularizations& regularizations) const {
    PerExampleData result;
    for (size_t group = 0; group < examples_by_group_.size(); ++group) {
      const TTypes<float>::Vec weights =
          weights_and_deltas_.WeightsOfGroup(group);
      const std::vector<std::atomic<double>>& delta_weights =
          weights_and_deltas_.DeltaWeightsOfGroup(group);

      const SparseExamples& sparse_indices_values = examples_by_group_[group];
      if (sparse_indices_values[example_id]) {
        const auto indices = sparse_indices_values[example_id]->feature_indices;
        const auto values = sparse_indices_values[example_id]->feature_values;
        for (int64 dim = 0; dim < indices.dimension(0); ++dim) {
          const int64 index = internal::SubtleMustCopy(indices(dim));
          const double weight = weights(index);
          const std::atomic<double>& delta_weight = delta_weights[index];
          const double value = values(dim);
          result.wx +=
              regularizations.Shrink(weight + delta_weight.load()) * value;
        }
        result.normalized_squared_norm +=
            sparse_indices_values[example_id]->squared_norm;
      }
    }
    result.normalized_squared_norm /= regularizations.symmetric_l2();
    return result;
  }

  // Updates the delta weight for each feature occuring in 'example_id',
  // given the weighted change in the dual for this example
  // (bounded_dual_delta), and the 'l2_regularization'.
  void UpdateDeltaWeights(const int64 example_id,
                          const double bounded_dual_delta,
                          const double l2_regularization) {
    for (size_t group = 0; group < examples_by_group_.size(); ++group) {
      std::vector<std::atomic<double>>& delta_weights =
          *weights_and_deltas_.DeltaWeightsOfGroup(group);

      const SparseExamples& sparse_indices_values = examples_by_group_[group];
      if (sparse_indices_values[example_id]) {
        const auto indices = sparse_indices_values[example_id]->feature_indices;
        const auto values = sparse_indices_values[example_id]->feature_values;
        for (int64 dim = 0; dim < indices.dimension(0); ++dim) {
          const int64 index = internal::SubtleMustCopy(indices(dim));
          std::atomic<double>* const delta_weight = &delta_weights[index];
          const double value = values(dim);
          AtomicAdd(bounded_dual_delta * value / l2_regularization,
                    delta_weight);
        }
      }
    }
  }

  // Adds all of the delta weights which were computed during processing
  // of this mini-batch into the feature-weights.  Must be called once
  // at the end of mini-batch processing.
  void AddDeltaWeights(const DeviceBase::CpuWorkerThreads& worker_threads) {
    weights_and_deltas_.AddDeltaWeights(worker_threads);
  }

  size_t NumGroups() const { return examples_by_group_.size(); }

 private:
  // A feature group of a single example by this struct.
  struct PerExampleSparseIndicesValues {
    // N X 1 vector with feature indices.
    Eigen::Tensor</*const*/ int64, 1, Eigen::RowMajor> feature_indices;

    // N X 1 vector with feature values.
    TTypes</*const*/ float>::UnalignedVec feature_values;

    // sum squared norm of the features.
    double squared_norm;
  };

  Status FillExamples(OpKernelContext* const context,
                      const size_t num_sparse_features, const int num_examples,
                      const DeviceBase::CpuWorkerThreads& worker_threads);

  // SparseExamples represent sparse feature groups of each example.
  using SparseExamples =
      std::vector<std::unique_ptr<const PerExampleSparseIndicesValues>>;

  // SparseExamples associated with each sparse feature group.
  std::vector<SparseExamples> examples_by_group_;

  WeightsAndDeltas weights_and_deltas_;

  TF_DISALLOW_COPY_AND_ASSIGN(SparseFeaturesAndWeights);
};

// Goes through the entire training set once, in a parallel and partitioned
// fashion, so that we create per-example structures. A non-OK return status
// indicates that the contents of SparseFeaturesAndWeights cannot be trusted or
// used.
Status SparseFeaturesAndWeights::FillExamples(
    OpKernelContext* const context, const size_t num_sparse_features,
    const int num_examples,
    const DeviceBase::CpuWorkerThreads& worker_threads) {
  OpInputList sparse_features_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_features_indices",
                                         &sparse_features_indices_inputs));
  OpInputList sparse_features_values_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_features_values",
                                         &sparse_features_values_inputs));

  if (sparse_features_indices_inputs.size() != num_sparse_features ||
      sparse_features_values_inputs.size() != num_sparse_features ||
      weights_and_deltas_.NumGroups() != num_sparse_features) {
    return errors::Internal("Unaligned sparse features.");
  }

  examples_by_group_.clear();
  examples_by_group_.resize(num_sparse_features);

  mutex mu;
  Status result GUARDED_BY(mu);
  {
    auto parse_partition = [&](const int64 begin, const int64 end) {
      // We set the order as [0, 1], which specifies that its row-major
      // increasing. This means first column has ids which are lexicographically
      // increasing.
      static const int64 kIndicesDims = 2;
      gtl::InlinedVector<int64, 8> order(kIndicesDims);
      std::iota(order.begin(), order.end(), 0);

      // The static_cast here is safe since begin and end can be at most
      // num_examples which is an int.
      for (int i = static_cast<int>(begin); i < end; ++i) {
        if (sparse_features_indices_inputs[i].shape().dims() != kIndicesDims) {
          mutex_lock l(mu);
          result = errors::InvalidArgument(strings::Printf(
              "Indices should have exactly %lld dimensions. Encountered: %d",
              kIndicesDims, sparse_features_indices_inputs[i].shape().dims()));
          return;
        }

        sparse::SparseTensor st(
            sparse_features_indices_inputs[i], sparse_features_values_inputs[i],
            sparse_features_indices_inputs[i].shape(), order);
        examples_by_group_[i] = SparseExamples(num_examples);
        for (const auto& example_group : st.group({0})) {
          const TTypes<int64>::UnalignedConstMatrix indices =
              example_group.indices();
          const int64 example_index = internal::SubtleMustCopy(indices(0, 0));
          if (example_index < 0 || example_index >= num_examples) {
            mutex_lock l(mu);
            result = errors::Internal(strings::Printf(
                "Example indices should be in [0, %d). Encountered: %lld",
                num_examples, example_index));
            return;
          }

          const auto feature_indices = indices.chip</*dim=*/1>(/*offset=*/1);
          const Eigen::Tensor<int64, 0, Eigen::RowMajor> min_feature_index =
              feature_indices.minimum();
          const Eigen::Tensor<int64, 0, Eigen::RowMajor> max_feature_index =
              feature_indices.maximum();
          if (min_feature_index() < 0 ||
              static_cast<size_t>(max_feature_index()) >=
                  weights_and_deltas_.NumFeaturesOfGroup(i)) {
            mutex_lock l(mu);
            result = errors::InvalidArgument(strings::Printf(
                "Feature indices should be in [0, %ld). Encountered "
                "min:%lld max:%lld for example:%lld",
                weights_and_deltas_.NumFeaturesOfGroup(i), min_feature_index(),
                max_feature_index(), example_index));
            return;
          }

          const Eigen::Tensor<float, 0, Eigen::RowMajor> squared_norm =
              example_group.values<float>().square().sum();
          examples_by_group_[i][example_index].reset(
              new PerExampleSparseIndicesValues{feature_indices,
                                                example_group.values<float>(),
                                                squared_norm()});
        }
      }
    };

    // For each column, the cost of parsing it is O(num_examples). We use
    // num_examples here, as empirically Shard() creates the right amount of
    // threads based on the problem size.
    // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
    const int64 kCostPerUnit = num_examples;
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_sparse_features, kCostPerUnit, parse_partition);
  }

  return result;
}

// FeaturesAndWeights provides a unified view of training features and their
// weights, abstracting away the differences between sparse and dense
// feature representations.
class FeaturesAndWeights {
 public:
  FeaturesAndWeights() {}

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context,
                    const int64 num_sparse_features, const int num_examples) {
    TF_RETURN_IF_ERROR(sparse_features_and_weights_.Initialize(
        context, num_sparse_features, num_examples));
    TF_RETURN_IF_ERROR(dense_features_and_weights_.Initialize(context));
    return Status::OK();
  }

  // Computes PerExampleData for 'example_id'.
  PerExampleData ComputeWxAndWeightedExampleNorm(
      const int64 example_id, const Regularizations& regularizations) const {
    const PerExampleData sparse_data =
        sparse_features_and_weights_.ComputeWxAndWeightedExampleNorm(
            example_id, regularizations);
    const PerExampleData dense_data =
        dense_features_and_weights_.ComputeWxAndWeightedExampleNorm(
            example_id, regularizations);

    return AddPerExampleData(sparse_data, dense_data);
  }

  // Updates the delta weight for each feature occuring in 'example_id',
  // given the weighted change in the dual for this example
  // (bounded_dual_delta), and the 'l2_regularization'.
  void UpdateDeltaWeights(const int64 example_id,
                          const double bounded_dual_delta,
                          const double l2_regularization) {
    sparse_features_and_weights_.UpdateDeltaWeights(
        example_id, bounded_dual_delta, l2_regularization);
    dense_features_and_weights_.UpdateDeltaWeights(
        example_id, bounded_dual_delta, l2_regularization);
  }

  // Adds all of the delta weights which were computed during processing
  // of this mini-batch into the feature-weights.  Must be called once
  // at the end of mini-batch processing.
  void AddDeltaWeights(const DeviceBase::CpuWorkerThreads& worker_threads) {
    sparse_features_and_weights_.AddDeltaWeights(worker_threads);
    dense_features_and_weights_.AddDeltaWeights(worker_threads);
  }

  size_t NumGroups() const {
    return sparse_features_and_weights_.NumGroups() +
           dense_features_and_weights_.NumGroups();
  }

 private:
  SparseFeaturesAndWeights sparse_features_and_weights_;
  DenseFeaturesAndWeights dense_features_and_weights_;

  TF_DISALLOW_COPY_AND_ASSIGN(FeaturesAndWeights);
};

Status RunTrainStepsForMiniBatch(
    const int num_examples, const TTypes<const float>::Vec example_labels,
    const TTypes<const float>::Vec example_weights,
    const DeviceBase::CpuWorkerThreads& worker_threads,
    const Regularizations& regularizations, const DualLossUpdater& loss_updater,
    FeaturesAndWeights* const features_and_weights,
    TTypes<float>::Matrix* const example_state_data) {
  // Process examples in parallel, in a partitioned fashion.
  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  auto train_step = [&](const int64 begin, const int64 end) {
    for (int64 example_index = begin; example_index < end; ++example_index) {
      const float dual = (*example_state_data)(example_index, 0);
      const float example_weight = example_weights(example_index);
      float example_label = example_labels(example_index);
      const Status conversion_status =
          loss_updater.ConvertLabel(&example_label);
      if (!conversion_status.ok()) {
        mutex_lock l(mu);
        train_step_status = conversion_status;
        // Return from this worker thread - the calling thread is
        // responsible for checking context status and returning on error.
        return;
      }

      // Compute wx, example norm weighted by regularization, dual loss,
      // primal loss.
      const PerExampleData per_example_data =
          features_and_weights->ComputeWxAndWeightedExampleNorm(
              example_index, regularizations);

      const double primal_loss = loss_updater.ComputePrimalLoss(
          per_example_data.wx, example_label, example_weight);

      const double dual_loss =
          loss_updater.ComputeDualLoss(dual, example_label, example_weight);

      const double new_dual = loss_updater.ComputeUpdatedDual(
          example_label, example_weight, dual, per_example_data.wx,
          per_example_data.normalized_squared_norm, primal_loss, dual_loss);

      // Compute new weights.
      const double bounded_dual_delta = (new_dual - dual) * example_weight;
      features_and_weights->UpdateDeltaWeights(
          example_index, bounded_dual_delta, regularizations.symmetric_l2());

      // Update example data.
      (*example_state_data)(example_index, 0) = new_dual;
      (*example_state_data)(example_index, 1) = primal_loss;
      (*example_state_data)(example_index, 2) = dual_loss;
      (*example_state_data)(example_index, 3) = example_weight;
    }
  };
  // TODO(sibyl-Aix6ihai): Current multiplier 100K works well empirically
  // but perhaps we can tune it better.
  const int64 kCostPerUnit = 100000 * features_and_weights->NumGroups();
  Shard(worker_threads.num_threads, worker_threads.workers, num_examples,
        kCostPerUnit, train_step);
  return train_step_status;
}

}  // namespace

class SdcaSolver : public OpKernel {
 public:
  explicit SdcaSolver(OpKernelConstruction* const context) : OpKernel(context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      loss_updater_.reset(new LogisticLossUpdater);
    } else if (loss_type == "squared_loss") {
      loss_updater_.reset(new SquaredLossUpdater);
    } else if (loss_type == "hinge_loss") {
      loss_updater_.reset(new HingeLossUpdater);
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }

    OP_REQUIRES_OK(context, context->GetAttr("num_sparse_features",
                                             &num_sparse_features_));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_dense_features", &num_dense_features_));
    OP_REQUIRES(
        context, num_sparse_features_ + num_dense_features_ > 0,
        errors::InvalidArgument("Requires at least one feature to train."));
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
    OP_REQUIRES_OK(context, context->GetAttr("num_inner_iterations",
                                             &num_inner_iterations_));
  }

  void Compute(OpKernelContext* const context) override {
    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input("example_weights", &example_weights_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_weights_t->shape()),
                errors::InvalidArgument("example_weights should be a vector."));
    const auto example_weights = example_weights_t->vec<float>();
    OP_REQUIRES(context,
                example_weights.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument(strings::Printf(
                    "Too many examples in a mini-batch: %ld > %d",
                    example_weights.size(), std::numeric_limits<int>::max())));
    const int num_examples = static_cast<int>(example_weights.size());

    const Tensor* example_labels_t;
    OP_REQUIRES_OK(context,
                   context->input("example_labels", &example_labels_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(example_labels_t->shape()),
                errors::InvalidArgument("example_labels should be a vector."));
    const auto example_labels = example_labels_t->vec<float>();
    OP_REQUIRES(context, example_labels.size() == num_examples,
                errors::InvalidArgument(strings::Printf(
                    "The number of example labels (%ld) should match the "
                    "number of example weights (%d).",
                    example_labels.size(), num_examples)));

    const Tensor* example_state_data_t;
    OP_REQUIRES_OK(context,
                   context->input("example_state_data", &example_state_data_t));
    TensorShape expected_example_state_shape({num_examples, 4});
    OP_REQUIRES(
        context, example_state_data_t->shape() == expected_example_state_shape,
        errors::InvalidArgument("Expected shape ",
                                expected_example_state_shape.DebugString(),
                                " for example_state_data, got ",
                                example_state_data_t->shape().DebugString()));

    Tensor mutable_example_state_data_t(*example_state_data_t);
    auto example_state_data = mutable_example_state_data_t.matrix<float>();

    FeaturesAndWeights features_and_weights;
    OP_REQUIRES_OK(context, features_and_weights.Initialize(
                                context, num_sparse_features_, num_examples));

    for (int i = 0; i < num_inner_iterations_; ++i) {
      OP_REQUIRES_OK(
          context, RunTrainStepsForMiniBatch(
                       num_examples, example_labels, example_weights,
                       *context->device()->tensorflow_cpu_worker_threads(),
                       regularizations_, *loss_updater_, &features_and_weights,
                       &example_state_data));
    }
    features_and_weights.AddDeltaWeights(
        *context->device()->tensorflow_cpu_worker_threads());

    context->set_output("example_data_data_out", mutable_example_state_data_t);
  }

 private:
  // TODO(sibyl-Aix6ihai): We could use the type-constraint on loss_type, and
  // template the entire class to avoid the virtual table lookup penalty in
  // the inner loop.
  std::unique_ptr<DualLossUpdater> loss_updater_;
  int64 num_sparse_features_;
  int64 num_dense_features_;
  Regularizations regularizations_;
  int num_inner_iterations_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaSolver").Device(DEVICE_CPU), SdcaSolver);

class SdcaShrinkL1 : public OpKernel {
 public:
  explicit SdcaShrinkL1(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
  }

  void Compute(OpKernelContext* const context) override {
    for (const string& list_name : {"sparse_weights", "dense_weights"}) {
      WeightsByGroup weights_by_group;
      OP_REQUIRES_OK(context, weights_by_group.Initialize(context, list_name));
      weights_by_group.Shrink(regularizations_);
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
// approximately 10^-21 (ie 2^60 / 2^129).
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
      const Fprint128 fprint = Fingerprint128(in_values(i));
      // Hex encode the fprint as a string (33 characters).
      out_values(i) = strings::StrCat(strings::FpToString(fprint.high64), "-",
                                      strings::FpToString(fprint.low64));
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SdcaFprint").Device(DEVICE_CPU), SdcaFprint);

}  // namespace tensorflow
