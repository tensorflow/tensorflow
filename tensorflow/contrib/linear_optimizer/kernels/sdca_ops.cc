/* Copyright 2016 Google Inc. All Rights Reserved.

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
#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/linear_optimizer/kernels/logistic-loss.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

// A feature group of a single example by this struct.
struct PerExampleSparseIndicesWeights {
  // N X 2 matrix with (example_id, feature_indices).
  tensorflow::TTypes<const int64>::UnalignedMatrix indices;
  // N X 1 vector with feature weights.
  tensorflow::TTypes<float>::UnalignedVec values;
  // sum squared norm of the features.
  double norm;
};

struct Regularizations {
  float symmetric_l1 = 0;
  float symmetric_l2 = 0;
};

struct RegularizationLoss {
  double l1_loss = 0;
  double l2_loss = 0;
};

struct PerExampleData {
  double wx = 0;
  double norm = 0;
};

// Tensor vector of floats which holds the weights.
using Weights = TTypes<float>::Vec;

// Weights associated with feature group, such that size of WeightsByIndex is
// the number of feature groups.
using WeightsByIndex = std::vector<Weights>;

// SparseExamples represent sparse feature groups of each example.
using SparseExamples =
    std::vector<std::unique_ptr<const PerExampleSparseIndicesWeights>>;

// SparseExamples associated with each sparse feature group.
using SparseExamplesByIndex = std::vector<SparseExamples>;

using DenseFeaturesByIndex = std::vector<tensorflow::TTypes<const float>::Vec>;

// Compute the shrinkage factor for proximal sdca.
inline double ShrinkageFactor(const Regularizations& regularizations) {
  return regularizations.symmetric_l1 / regularizations.symmetric_l2;
}

// Proximal SDCA shrinking for L1 regularization.
inline double Shrink(const double weight, const double shrink_by) {
  const double shrink_weight = std::max(std::abs(weight) - shrink_by, 0.0);
  if (shrink_weight > 0.0) {
    return std::copysign(shrink_weight, weight);
  }
  return 0.0;
}

// Compute L1 and L2 regularization loss.
inline RegularizationLoss ComputeRegularizationLoss(
    const WeightsByIndex& sparse_weights_by_index,
    const WeightsByIndex& dense_weights_by_index,
    const Regularizations& regularizations) {
  RegularizationLoss result;

  const double shrink_by = ShrinkageFactor(regularizations);
  auto accumulate_regularization_loss = [&](const double w) {
    const double sw = std::abs(Shrink(w, shrink_by));
    result.l1_loss += sw;
    result.l2_loss += sw * sw;
  };

  for (auto& sparse_weights : sparse_weights_by_index) {
    for (size_t i = 0; i < sparse_weights.size(); ++i) {
      accumulate_regularization_loss(sparse_weights(i));
    }
  }

  for (auto& dense_weights : dense_weights_by_index) {
    accumulate_regularization_loss(dense_weights(0));
  }

  result.l1_loss *= regularizations.symmetric_l1;
  result.l2_loss *= regularizations.symmetric_l2;
  return result;
}

// Compute PerExampleData which contains the logits, and weighted example norm
// for a given example_id. Norm is weighted by 1/(lambda*N).
inline PerExampleData ComputeWxAndWeightedExampleNorm(
    const int64 example_id, const WeightsByIndex& sparse_weights_by_index,
    const SparseExamplesByIndex& sparse_examples_by_index,
    const WeightsByIndex& dense_weights_by_index,
    const DenseFeaturesByIndex& dense_features,
    const Regularizations& regularizations) {
  PerExampleData result;
  const double shrink_by = ShrinkageFactor(regularizations);
  for (size_t i = 0; i < sparse_examples_by_index.size(); ++i) {
    const SparseExamples& sparse_indices_values = sparse_examples_by_index[i];
    const Weights sparse_weights = sparse_weights_by_index[i];
    if (sparse_indices_values[example_id]) {
      const auto indices = sparse_indices_values[example_id]->indices;
      const auto values = sparse_indices_values[example_id]->values;
      for (size_t dim = 0; dim < indices.dimension(0); ++dim) {
        result.wx +=
            Shrink(sparse_weights(indices(dim, 1)), shrink_by) * values(dim);
      }
      result.norm += sparse_indices_values[example_id]->norm;
    }
  }
  for (size_t i = 0; i < dense_features.size(); ++i) {
    const auto dense_values = dense_features[i];
    const Weights dense_weights = dense_weights_by_index[i];
    result.wx += Shrink(dense_weights(0), shrink_by) * dense_values(example_id);
    result.norm += dense_values(example_id) * dense_values(example_id);
  }
  result.norm /= regularizations.symmetric_l2;
  return result;
}

// Apply L1 regularization on the weights,
void ShrinkWeights(const Regularizations& regularizations,
                   WeightsByIndex* const sparse_weights_by_index,
                   WeightsByIndex* const dense_weights_by_index) {
  const double shrink_by = ShrinkageFactor(regularizations);
  for (auto& sparse_weights : *sparse_weights_by_index) {
    for (size_t i = 0; i < sparse_weights.size(); ++i) {
      sparse_weights(i) = Shrink(sparse_weights(i), shrink_by);
    }
  }
  for (auto& dense_weights : *dense_weights_by_index) {
    dense_weights(0) = Shrink(dense_weights(0), shrink_by);
  }
}

void UpdateWeights(const int64 example_id,
                   const SparseExamplesByIndex& sparse_examples_by_index,
                   const DenseFeaturesByIndex& dense_features,
                   const double bounded_dual_delta,
                   const double l2_regularization,
                   WeightsByIndex* const sparse_weights_by_index,
                   WeightsByIndex* const dense_weights_by_index) {
  for (size_t i = 0; i < sparse_examples_by_index.size(); ++i) {
    const SparseExamples& sparse_indices_values = sparse_examples_by_index[i];
    Weights sparse_weights = (*sparse_weights_by_index)[i];
    if (sparse_indices_values[example_id]) {
      const auto indices = sparse_indices_values[example_id]->indices;
      const auto values = sparse_indices_values[example_id]->values;
      for (size_t dim = 0; dim < indices.dimension(0); ++dim) {
        // TODO(rohananil): Atomic updates provide better convergence guarantees
        // However, casting float to atomic<float> is UB. We may consider
        // sharded set of locks, or bring primal-dual relationship to consistent
        // state after several epochs.
        sparse_weights(indices(dim, 1)) +=
            bounded_dual_delta * values(dim) / l2_regularization;
      }
    }
  }
  for (size_t i = 0; i < dense_features.size(); ++i) {
    const auto dense_values = dense_features[i];
    Weights dense_weights = (*dense_weights_by_index)[i];
    // TODO(rohananil): Atomic updates provide better convergence gaurantees
    // However, casting float to atomic<float> is UB. We may consider
    // sharded set of locks, or bring primal-dual relationship to consistent
    // state after several epochs.
    dense_weights(0) +=
        bounded_dual_delta * dense_values(example_id) / l2_regularization;
  }
}

// Atomically add a double to a std::atomic<double>.
inline void AtomicAdd(const double value, std::atomic<double>* const dst) {
  // We use a strong version of compare-exchange, as weak version can spuriously
  // fail.
  for (double c = dst->load(); !dst->compare_exchange_strong(c, c + value);) {
  }
}

}  // namespace

class SdcaSolver : public OpKernel {
 public:
  explicit SdcaSolver(OpKernelConstruction* context) : OpKernel(context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("LossType", &loss_type));
    if (loss_type == "logistic_loss") {
      compute_dual_loss_ = logistic_loss::ComputeDualLoss;
      compute_primal_loss_ = logistic_loss::ComputePrimalLoss;
      compute_dual_update_ = logistic_loss::ComputeUpdatedDual;
    }
    OP_REQUIRES_OK(
        context, context->GetAttr("NumSparseFeatures", &num_sparse_features_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("NumDenseFeatures", &num_dense_features_));
    OP_REQUIRES(
        context, num_sparse_features_ + num_dense_features_ > 0,
        errors::InvalidArgument("Requires at least one feature to train."));

    OP_REQUIRES_OK(context,
                   context->GetAttr("L1", &regularizations_.symmetric_l1));
    OP_REQUIRES_OK(context,
                   context->GetAttr("L2", &regularizations_.symmetric_l2));
    OP_REQUIRES_OK(context, context->GetAttr("DualityGapThreshold",
                                             &duality_gap_threshold_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input("example_weights", &example_weights_t));
    const auto example_weights = example_weights_t->vec<float>();

    Tensor primal_loss_t;
    OP_REQUIRES_OK(context,
                   context->mutable_input("primal_loss", &primal_loss_t,
                                          /*lock_held=*/true));
    auto primal_loss = primal_loss_t.scalar<double>();

    const int64 num_examples = example_weights.size();

    Eigen::Tensor<float, 0, Eigen::RowMajor> example_weights_sum;
    example_weights_sum.device(context->eigen_cpu_device()) =
        example_weights.sum();
    const float weighted_examples = example_weights_sum();

    OP_REQUIRES(context, weighted_examples > 0.0,
                errors::InvalidArgument("No weighted examples in ",
                                        num_examples, " training examples"));
    // We scale it up by weighted examples.
    regularizations_.symmetric_l1 =
        regularizations_.symmetric_l1 * weighted_examples;
    regularizations_.symmetric_l2 =
        std::max(regularizations_.symmetric_l2 * weighted_examples, 1.0f);

    OpInputList sparse_features_indices_inputs;
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_features_indices",
                                       &sparse_features_indices_inputs));
    OpInputList sparse_features_values;
    OP_REQUIRES_OK(context, context->input_list("sparse_features_values",
                                                &sparse_features_values));
    SparseExamplesByIndex sparse_examples_by_index(num_sparse_features_);
    // Goes through the entire training set once, so that we create per example
    // structures. We use it downstream for randomizating order of the examples.
    auto do_parse = [&](const int64 begin, const int64 end) {
      // We set the order as [0, 1], which specifies that its row-major
      // increasing. This means first column has ids which is
      // lexicographically increasing.
      static const int64 kIndicesDims = 2;
      gtl::InlinedVector<int64, 8> order(kIndicesDims);
      std::iota(order.begin(), order.end(), 0);
      for (size_t i = begin; i < end; ++i) {
        OP_REQUIRES(context, sparse_features_indices_inputs[i].shape().dims() ==
                                 kIndicesDims,
                    errors::InvalidArgument(
                        "Indices should have exactly 2 dimensions"));
        tensorflow::sparse::SparseTensor st(
            sparse_features_indices_inputs[i], sparse_features_values[i],
            sparse_features_indices_inputs[i].shape(), order);
        sparse_examples_by_index[i] = SparseExamples(num_examples);
        for (const auto& example_group : st.group({0})) {
          const int64 example_id = example_group.indices()(0, 0);
          const Eigen::Tensor<float, 0, Eigen::RowMajor> norm =
              example_group.values<float>().square().sum();
          sparse_examples_by_index[i][example_id].reset(
              new PerExampleSparseIndicesWeights{example_group.indices(),
                                                 example_group.values<float>(),
                                                 norm()});
        }
      }
    };
    {
      const DeviceBase::CpuWorkerThreads* const worker_threads =
          context->device()->tensorflow_cpu_worker_threads();
      // For each column, the cost of parsing it O(num_examples). We use
      // num_examples here, as empircally Shard() creates the right amount of
      // threads based on the problem size.
      // TODO(rohananil): Tune this as a function of dataset size.
      const int64 kCostPerUnit = num_examples;
      Shard(worker_threads->num_threads, worker_threads->workers,
            num_sparse_features_, kCostPerUnit, do_parse);
    }

    OpInputList dense_features_inputs;
    OP_REQUIRES_OK(
        context, context->input_list("dense_features", &dense_features_inputs));

    DenseFeaturesByIndex dense_features;
    for (auto& dense_features_input : dense_features_inputs) {
      dense_features.emplace_back(dense_features_input.vec<float>());
    }

    const Tensor* example_labels_t;
    OP_REQUIRES_OK(context,
                   context->input("example_labels", &example_labels_t));
    const auto example_labels = example_labels_t->vec<float>();

    Tensor dual_variables_t;
    OP_REQUIRES_OK(context,
                   context->mutable_input("dual_variables", &dual_variables_t,
                                          /*lock_held=*/true));
    auto dual_variables = dual_variables_t.vec<float>();

    OpMutableInputList sparse_weights_by_index_inputs;
    OP_REQUIRES_OK(
        context, context->mutable_input_list("sparse_weights_by_index",
                                             &sparse_weights_by_index_inputs));
    WeightsByIndex sparse_weights_by_index;
    for (size_t i = 0; i < sparse_weights_by_index_inputs.size(); ++i) {
      sparse_weights_by_index.emplace_back(
          sparse_weights_by_index_inputs.at(i, /*lock_held=*/true)
              .flat<float>());
    }

    OpMutableInputList dense_weights_by_index_inputs;
    OP_REQUIRES_OK(context,
                   context->mutable_input_list("dense_weights_by_index",
                                               &dense_weights_by_index_inputs));
    WeightsByIndex dense_weights_by_index;
    for (size_t i = 0; i < dense_weights_by_index_inputs.size(); ++i) {
      dense_weights_by_index.emplace_back(
          dense_weights_by_index_inputs.at(i, /*lock_held=*/true)
              .flat<float>());
    }

    std::vector<int64> example_ids(num_examples);
    std::iota(example_ids.begin(), example_ids.end(), 0);
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::atomic<double> total_primal_loss(0);
    std::atomic<double> total_dual_loss(0);
    // Break when duality gap |P(w) - D(alpha)| is less than
    // duality_gap_threshold_
    double total_approx_duality_gap = std::numeric_limits<double>::max();
    while ((total_approx_duality_gap / weighted_examples) >
           duality_gap_threshold_) {
      // Reset accumulated losses.
      total_primal_loss = 0;
      total_dual_loss = 0;
      std::shuffle(example_ids.begin(), example_ids.end(), random_generator);
      auto do_update = [&](const int64 begin, const int64 end) {
        double dual_loss_on_example_subset = 0;
        double primal_loss_on_example_subset = 0;
        for (int64 offset = begin; offset < end; ++offset) {
          // Get example id, label, and weight.
          const int64 example_id = example_ids[offset];
          OP_REQUIRES(context, !(example_labels(example_id) > 0 &&
                                 example_labels(example_id) < 1),
                      errors::InvalidArgument(
                          "Fractional labels not supported right now. "
                          "Found example with label: ",
                          example_labels(example_id)));
          const float example_label = example_labels(example_id) == 0 ? -1 : 1;
          const double current_dual = dual_variables(example_id);
          const double example_weight = example_weights(example_id);

          // Compute wx, example norm weighted by regularization, dual loss,
          // primal loss.
          const PerExampleData per_example_data =
              ComputeWxAndWeightedExampleNorm(
                  example_id, sparse_weights_by_index, sparse_examples_by_index,
                  dense_weights_by_index, dense_features, regularizations_);

          const double dual_loss = compute_dual_loss_(
              current_dual, example_label, example_weight);
          dual_loss_on_example_subset += dual_loss;
          const double primal_loss = compute_primal_loss_(
              per_example_data.wx, example_label, example_weight);
          primal_loss_on_example_subset += primal_loss;

          // Update dual variable.
          dual_variables(example_id) = compute_dual_update_(
              example_label, example_weight, current_dual, per_example_data.wx,
              per_example_data.norm, primal_loss, dual_loss);

          // Compute new weights.
          const double bounded_dual_delta =
              (dual_variables(example_id) - current_dual) * example_weight;
          UpdateWeights(example_id, sparse_examples_by_index, dense_features,
                        bounded_dual_delta, regularizations_.symmetric_l2,
                        &sparse_weights_by_index, &dense_weights_by_index);
        }
        AtomicAdd(primal_loss_on_example_subset, &total_primal_loss);
        AtomicAdd(dual_loss_on_example_subset, &total_dual_loss);
        // TODO(rohananil): We may in the future want to make the primal-dual
        // relationship consistent as our current updates are not transactional.
      };
      const DeviceBase::CpuWorkerThreads* const worker_threads =
          context->device()->tensorflow_cpu_worker_threads();
      const int64 kCostPerUnit =
          100000 * (num_sparse_features_ + num_dense_features_);
      Shard(worker_threads->num_threads, worker_threads->workers, num_examples,
            kCostPerUnit, do_update);
      const RegularizationLoss regularization_loss = ComputeRegularizationLoss(
          sparse_weights_by_index, dense_weights_by_index, regularizations_);
      total_approx_duality_gap =
          total_primal_loss.load() + total_dual_loss.load() +
          regularization_loss.l1_loss + regularization_loss.l2_loss;
      primal_loss() = (total_primal_loss.load() + regularization_loss.l1_loss +
                       regularization_loss.l2_loss) /
                      weighted_examples;
    }
    ShrinkWeights(regularizations_, &sparse_weights_by_index,
                  &dense_weights_by_index);
  }

 private:
  std::function<decltype(logistic_loss::ComputeDualLoss)> compute_dual_loss_;
  std::function<decltype(logistic_loss::ComputePrimalLoss)>
      compute_primal_loss_;
  std::function<decltype(logistic_loss::ComputeUpdatedDual)>
      compute_dual_update_;
  int64 num_sparse_features_;
  int64 num_dense_features_;
  Regularizations regularizations_;
  float duality_gap_threshold_;
};

REGISTER_KERNEL_BUILDER(Name("SdcaSolver").Device(DEVICE_CPU), SdcaSolver);

}  // namespace tensorflow
