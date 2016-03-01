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
#include "tensorflow/core/framework/resource_mgr.h"
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
  TTypes<const int64>::UnalignedMatrix indices;
  // N X 1 vector with feature weights.
  TTypes<float>::UnalignedVec values;
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

// Resource for storing dual variable per example across many sessions.
// This class is thread-safe.
struct DualsByExample : public ResourceBase {
  DualsByExample(const string& container, const string& solver_uuid)
      : container_(container), solver_uuid_(solver_uuid) {}

  string DebugString() override {
    return strings::StrCat("DualsByExample(", container_, ", ", solver_uuid_,
                           ")");
  }

  float& operator[](const uint64 example) {
    mutex_lock l(mu_);
    return duals_by_example_[example];
  }

 private:
  const string container_;
  const string solver_uuid_;

  // TODO(katsiapis): Come up with a more efficient locking scheme.
  mutex mu_;
  std::unordered_map<uint64, float> duals_by_example_;  // Guarded by mu.
};

// Weights associated with a (sparse or dense) feature group, such that size of
// WeightsByGroup is the number of feature groups.
using WeightsByGroup = std::vector<TTypes<float>::Vec>;

// SparseExamples represent sparse feature groups of each example.
using SparseExamples =
    std::vector<std::unique_ptr<const PerExampleSparseIndicesWeights>>;

// SparseExamples associated with each sparse feature group.
using SparseExamplesByGroup = std::vector<SparseExamples>;

// Dense features associated with each dense feature group.
using DenseFeaturesByGroup = std::vector<TTypes<const float>::Vec>;

// Go through the entire training set once, in a parallel and partitioned
// fashion, so that we create per-example structures. A non-OK return status
// indicates that the contents of sparse_examples_by_group cannot be trusted or
// used.
Status FillSparseExamplesByGroup(
    const int64 num_sparse_features, const size_t num_examples,
    const OpInputList& sparse_features_indices_inputs,
    const OpInputList& sparse_features_values_inputs,
    const DeviceBase::CpuWorkerThreads& worker_threads,
    SparseExamplesByGroup* const sparse_examples_by_group) {
  mutex mu;
  Status result;  // Guarded by mu.

  sparse_examples_by_group->clear();
  sparse_examples_by_group->resize(num_sparse_features);
  {
    auto parse_partition = [&](const int64 begin, const int64 end) {
      // We set the order as [0, 1], which specifies that its row-major
      // increasing. This means first column has ids which is
      // lexicographically increasing.
      static const int64 kIndicesDims = 2;
      gtl::InlinedVector<int64, 8> order(kIndicesDims);
      std::iota(order.begin(), order.end(), 0);
      for (size_t i = begin; i < end; ++i) {
        if (sparse_features_indices_inputs[i].shape().dims() != kIndicesDims) {
          mutex_lock l(mu);
          result = errors::InvalidArgument(strings::StrCat(
              "Indices should have exactly ", kIndicesDims, " dimensions"));
          return;
        }
        sparse::SparseTensor st(
            sparse_features_indices_inputs[i], sparse_features_values_inputs[i],
            sparse_features_indices_inputs[i].shape(), order);
        (*sparse_examples_by_group)[i] = SparseExamples(num_examples);
        for (const auto& example_group : st.group({0})) {
          const int64 example_id = example_group.indices()(0, 0);
          const Eigen::Tensor<float, 0, Eigen::RowMajor> norm =
              example_group.values<float>().square().sum();
          (*sparse_examples_by_group)[i][example_id].reset(
              new PerExampleSparseIndicesWeights{example_group.indices(),
                                                 example_group.values<float>(),
                                                 norm()});
        }
      }
    };

    // For each column, the cost of parsing it is O(num_examples). We use
    // num_examples here, as empirically Shard() creates the right amount of
    // threads based on the problem size.
    // TODO(rohananil): Tune this as a function of dataset size.
    const int64 kCostPerUnit = static_cast<int64>(num_examples);
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_sparse_features, kCostPerUnit, parse_partition);
  }

  return result;
}

// A globally unique id for the given example.
//
// TODO(katsiapis): Change this to work with hashes of an instance's unique
// identifier and possibly move to a larger key space to prevent colisions.
inline uint64 Example(const int64 example_id) {
  return static_cast<uint64>(example_id);
}

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
    const WeightsByGroup& sparse_weights_by_group,
    const WeightsByGroup& dense_weights_by_group,
    const Regularizations& regularizations) {
  RegularizationLoss result;

  const double shrink_by = ShrinkageFactor(regularizations);
  auto accumulate_regularization_loss = [&](const double w) {
    const double sw = std::abs(Shrink(w, shrink_by));
    result.l1_loss += sw;
    result.l2_loss += sw * sw;
  };

  for (auto& sparse_weights : sparse_weights_by_group) {
    for (size_t i = 0; i < sparse_weights.size(); ++i) {
      accumulate_regularization_loss(sparse_weights(i));
    }
  }

  for (auto& dense_weights : dense_weights_by_group) {
    accumulate_regularization_loss(dense_weights(0));
  }

  result.l1_loss *= regularizations.symmetric_l1;
  result.l2_loss *= regularizations.symmetric_l2;
  return result;
}

// Compute PerExampleData which contains the logits, and weighted example norm
// for a given example_id. Norm is weighted by 1/(lambda*N).
inline PerExampleData ComputeWxAndWeightedExampleNorm(
    const int64 example_id, const WeightsByGroup& sparse_weights_by_group,
    const SparseExamplesByGroup& sparse_examples_by_group,
    const WeightsByGroup& dense_weights_by_group,
    const DenseFeaturesByGroup& dense_features_by_group,
    const Regularizations& regularizations) {
  PerExampleData result;
  const double shrink_by = ShrinkageFactor(regularizations);
  for (size_t i = 0; i < sparse_examples_by_group.size(); ++i) {
    const SparseExamples& sparse_indices_values = sparse_examples_by_group[i];
    const auto sparse_weights = sparse_weights_by_group[i];
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
  for (size_t i = 0; i < dense_features_by_group.size(); ++i) {
    const auto dense_values = dense_features_by_group[i];
    const auto dense_weights = dense_weights_by_group[i];
    result.wx += Shrink(dense_weights(0), shrink_by) * dense_values(example_id);
    result.norm += dense_values(example_id) * dense_values(example_id);
  }
  result.norm /= regularizations.symmetric_l2;
  return result;
}

// Apply L1 regularization on the weights,
void ShrinkWeights(const Regularizations& regularizations,
                   WeightsByGroup* const sparse_weights_by_group,
                   WeightsByGroup* const dense_weights_by_group) {
  const double shrink_by = ShrinkageFactor(regularizations);
  for (auto& sparse_weights : *sparse_weights_by_group) {
    for (size_t i = 0; i < sparse_weights.size(); ++i) {
      sparse_weights(i) = Shrink(sparse_weights(i), shrink_by);
    }
  }
  for (auto& dense_weights : *dense_weights_by_group) {
    dense_weights(0) = Shrink(dense_weights(0), shrink_by);
  }
}

void UpdateWeights(const int64 example_id,
                   const SparseExamplesByGroup& sparse_examples_by_group,
                   const DenseFeaturesByGroup& dense_features_by_group,
                   const double bounded_dual_delta,
                   const double l2_regularization,
                   WeightsByGroup* const sparse_weights_by_group,
                   WeightsByGroup* const dense_weights_by_group) {
  for (size_t i = 0; i < sparse_examples_by_group.size(); ++i) {
    const SparseExamples& sparse_examples = sparse_examples_by_group[i];
    auto sparse_weights = (*sparse_weights_by_group)[i];
    if (sparse_examples[example_id]) {
      const auto indices = sparse_examples[example_id]->indices;
      const auto values = sparse_examples[example_id]->values;
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
  for (size_t i = 0; i < dense_features_by_group.size(); ++i) {
    const auto dense_values = dense_features_by_group[i];
    auto dense_weights = (*dense_weights_by_group)[i];
    // TODO(rohananil): Atomic updates provide better convergence gaurantees
    // However, casting float to atomic<float> is UB. We may consider
    // sharded set of locks, or bring primal-dual relationship to consistent
    // state after several epochs.
    dense_weights(0) +=
        bounded_dual_delta * dense_values(example_id) / l2_regularization;
  }
}

// Atomically add a double to a std::atomic<double>.
inline void AtomicAdd(const double src, std::atomic<double>* const dst) {
  // We use a strong version of compare-exchange, as weak version can spuriously
  // fail.
  for (double c = dst->load(); !dst->compare_exchange_strong(c, c + src);) {
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
    OP_REQUIRES_OK(context, context->GetAttr("Container", &container_));
    OP_REQUIRES_OK(context, context->GetAttr("SolverUUID", &solver_uuid_));
  }

  void Compute(OpKernelContext* context) override {
    // Get a handle on a shared container across invocations of this Kernel.
    // The shared container is intended to maintain state (values of dual
    // variables) across invocations of the kernel on different input data.
    //
    // TODO(katsiapis): Replace this in-Kernel data structure with a first class
    // citizen mutable Dictionary in tensorflow proper, that we will initialize
    // and update externally.
    DualsByExample* duals_by_example = nullptr;
    OP_REQUIRES_OK(context,
                   context->resource_manager()->LookupOrCreate<DualsByExample>(
                       container_, solver_uuid_, &duals_by_example,
                       [this](DualsByExample** ret) {
                         *ret = new DualsByExample(container_, solver_uuid_);
                         return Status::OK();
                       }));
    OP_REQUIRES(
        context, !duals_by_example->RefCountIsOne(),
        errors::Internal("Expected shared-ownership of duals_by_example."));

    const Tensor* example_weights_t;
    OP_REQUIRES_OK(context,
                   context->input("example_weights", &example_weights_t));
    const auto example_weights = example_weights_t->vec<float>();

    Tensor primal_loss_t;
    OP_REQUIRES_OK(context,
                   context->mutable_input("primal_loss", &primal_loss_t,
                                          /*lock_held=*/true));
    auto primal_loss = primal_loss_t.scalar<double>();

    Eigen::Tensor<float, 0, Eigen::RowMajor> example_weights_sum;
    example_weights_sum.device(context->eigen_cpu_device()) =
        example_weights.sum();
    const float weighted_examples = example_weights_sum();

    const size_t num_examples = example_weights.size();
    OP_REQUIRES(context, weighted_examples > 0,
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
    OpInputList sparse_features_values_inputs;
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_features_values",
                                       &sparse_features_values_inputs));

    SparseExamplesByGroup sparse_examples_by_group;
    OP_REQUIRES_OK(
        context,
        FillSparseExamplesByGroup(
            num_sparse_features_, num_examples, sparse_features_indices_inputs,
            sparse_features_values_inputs,
            *context->device()->tensorflow_cpu_worker_threads(),
            &sparse_examples_by_group));

    OpInputList dense_features_inputs;
    OP_REQUIRES_OK(
        context, context->input_list("dense_features", &dense_features_inputs));

    DenseFeaturesByGroup dense_features_by_group;
    for (const auto& dense_feature : dense_features_inputs) {
      dense_features_by_group.emplace_back(dense_feature.vec<float>());
    }

    const Tensor* example_labels_t;
    OP_REQUIRES_OK(context,
                   context->input("example_labels", &example_labels_t));
    const auto example_labels = example_labels_t->vec<float>();

    OpMutableInputList sparse_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list(
                                "sparse_weights", &sparse_weights_inputs));
    WeightsByGroup sparse_weights_by_group;
    for (size_t i = 0; i < sparse_weights_inputs.size(); ++i) {
      sparse_weights_by_group.emplace_back(
          sparse_weights_inputs.at(i, /*lock_held=*/true).flat<float>());
    }

    OpMutableInputList dense_weights_inputs;
    OP_REQUIRES_OK(context, context->mutable_input_list("dense_weights",
                                                        &dense_weights_inputs));
    WeightsByGroup dense_weights_by_group;
    for (size_t i = 0; i < dense_weights_inputs.size(); ++i) {
      dense_weights_by_group.emplace_back(
          dense_weights_inputs.at(i, /*lock_held=*/true).flat<float>());
    }

    // Those will be shuffled below at each iteration and processed in a
    // partitioned fashion across multiple threads.
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
      // Randomize the examples across iterations for faster convergence.
      std::shuffle(example_ids.begin(), example_ids.end(), random_generator);
      // Process examples in parallel, in a partitioned fashion.
      {
        mutex mu;  // Guards this->context.
        auto update_partition = [&](const int64 begin, const int64 end) {
          double dual_loss_on_example_subset = 0;
          double primal_loss_on_example_subset = 0;
          for (int64 offset = begin; offset < end; ++offset) {
            // Get example id, label, and weight.
            const int64 example_id = example_ids[offset];
            if (!(example_labels(example_id) == 0 ||
                  example_labels(example_id) == 1)) {
              mutex_lock l(mu);
              OP_REQUIRES(context, false,
                          errors::InvalidArgument(
                              "Fractional labels not supported right now. "
                              "Found example with label: ",
                              example_labels(example_id)));
            }
            const float example_label =
                example_labels(example_id) == 0 ? -1 : 1;
            const uint64 example = Example(example_id);
            const double current_dual = (*duals_by_example)[example];
            const double example_weight = example_weights(example_id);

            // Compute wx, example norm weighted by regularization, dual loss,
            // primal loss.
            const PerExampleData per_example_data =
                ComputeWxAndWeightedExampleNorm(
                    example_id, sparse_weights_by_group,
                    sparse_examples_by_group, dense_weights_by_group,
                    dense_features_by_group, regularizations_);

            const double dual_loss =
                compute_dual_loss_(current_dual, example_label, example_weight);
            dual_loss_on_example_subset += dual_loss;
            const double primal_loss = compute_primal_loss_(
                per_example_data.wx, example_label, example_weight);
            primal_loss_on_example_subset += primal_loss;

            const double new_dual = compute_dual_update_(
                example_label, example_weight, current_dual,
                per_example_data.wx, per_example_data.norm, primal_loss,
                dual_loss);

            // Compute new weights.
            const double bounded_dual_delta =
                (new_dual - current_dual) * example_weight;
            UpdateWeights(example_id, sparse_examples_by_group,
                          dense_features_by_group, bounded_dual_delta,
                          regularizations_.symmetric_l2,
                          &sparse_weights_by_group, &dense_weights_by_group);

            // Update dual variable.
            (*duals_by_example)[example] = new_dual;
          }
          AtomicAdd(primal_loss_on_example_subset, &total_primal_loss);
          AtomicAdd(dual_loss_on_example_subset, &total_dual_loss);
          // TODO(rohananil): We may in the future want to make the primal-dual
          // relationship consistent as our current updates are not
          // transactional.
        };
        const DeviceBase::CpuWorkerThreads* const worker_threads =
            context->device()->tensorflow_cpu_worker_threads();
        const int64 kCostPerUnit =
            100000 * (num_sparse_features_ + num_dense_features_);
        Shard(worker_threads->num_threads, worker_threads->workers,
              static_cast<int64>(num_examples), kCostPerUnit, update_partition);
      }

      const RegularizationLoss regularization_loss = ComputeRegularizationLoss(
          sparse_weights_by_group, dense_weights_by_group, regularizations_);
      total_approx_duality_gap =
          total_primal_loss.load() + total_dual_loss.load() +
          regularization_loss.l1_loss + regularization_loss.l2_loss;
      primal_loss() = (total_primal_loss.load() + regularization_loss.l1_loss +
                       regularization_loss.l2_loss) /
                      weighted_examples;
    }
    ShrinkWeights(regularizations_, &sparse_weights_by_group,
                  &dense_weights_by_group);

    // TODO(katsiapis): Use core::ScopedUnref once it's moved out of internal.
    duals_by_example->Unref();
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
  string container_;
  string solver_uuid_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaSolver").Device(DEVICE_CPU), SdcaSolver);

}  // namespace tensorflow
