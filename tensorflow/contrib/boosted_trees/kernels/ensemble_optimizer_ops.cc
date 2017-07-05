// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include <string>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/utils/dropout_utils.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"
#include "tensorflow/contrib/boosted_trees/resources/decision_tree_ensemble_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using boosted_trees::models::DecisionTreeEnsembleResource;
using boosted_trees::trees::DecisionTreeEnsembleConfig;
using boosted_trees::utils::DropoutUtils;
using errors::InvalidArgument;

namespace {

// Learning rate epsilon.
const float kLearningRateEps = 1e-8;

}  // namespace

class AddTreesToEnsembleOp : public OpKernel {
 public:
  explicit AddTreesToEnsembleOp(OpKernelConstruction* const context)
      : OpKernel(context) {
    // Ensure feature importance lhs inputs are references.
    OP_REQUIRES(
        context,
        IsRefType(context->input_type(kFeatureColumnUsageCountsHandleIdx)),
        errors::InvalidArgument(
            "Feature usage counts lhs input needs to be a ref type"));
    OP_REQUIRES(context,
                IsRefType(context->input_type(kFeatureColumnGainsHandleIdx)),
                errors::InvalidArgument(
                    "Feature gains lhs input needs to be a ref type"));
  }

  void Compute(OpKernelContext* const context) override {
    DecisionTreeEnsembleResource* decision_tree_ensemble_resource;
    // Create a reference to the underlying resource using the handle.
    OP_REQUIRES_OK(
        context, LookupResource(
                     context, HandleFromInput(context, kTreeEnsembleHandleIdx),
                     &decision_tree_ensemble_resource));
    // Lock the resource since we're mutating it.
    mutex_lock l(*decision_tree_ensemble_resource->get_mutex());
    // Remove the reference at the end of this scope.
    core::ScopedUnref unref_me(decision_tree_ensemble_resource);

    // Read feature importance info.
    mutex_lock fc_usage_counts_mutex_lock(
        *context->input_ref_mutex(kFeatureColumnUsageCountsHandleIdx));
    mutex_lock fc_gains_mutex_lock(
        *context->input_ref_mutex(kFeatureColumnGainsHandleIdx));
    Tensor fc_usage_counts_lhs_t =
        context->mutable_input(kFeatureColumnUsageCountsHandleIdx, true);
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(fc_usage_counts_lhs_t.shape()),
                InvalidArgument("Feature usage counts should be a vector."));
    OP_REQUIRES(context, fc_usage_counts_lhs_t.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: ",
                    requested_input(kFeatureColumnUsageCountsHandleIdx)));

    Tensor fc_gains_lhs_t =
        context->mutable_input(kFeatureColumnGainsHandleIdx, true);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(fc_gains_lhs_t.shape()),
                InvalidArgument("Feature gains should be a vector."));
    OP_REQUIRES(context, fc_gains_lhs_t.IsInitialized(),
                errors::FailedPrecondition(
                    "Attempting to use uninitialized variables: ",
                    requested_input(kFeatureColumnGainsHandleIdx)));

    const Tensor fc_usage_counts_rhs_t =
        context->input(kFeatureColumnUsageCountsToAddIdx);
    OP_REQUIRES(
        context,
        fc_usage_counts_lhs_t.shape().IsSameSize(fc_usage_counts_rhs_t.shape()),
        errors::InvalidArgument(
            "Shapes of both feature usage counts tensors should match.",
            " lhs shape= ", fc_usage_counts_lhs_t.shape().DebugString(),
            " rhs shape= ", fc_usage_counts_rhs_t.shape().DebugString()));

    const Tensor fc_gains_rhs_t = context->input(kFeatureColumnGainsToAddIdx);
    OP_REQUIRES(context,
                fc_gains_lhs_t.shape().IsSameSize(fc_gains_rhs_t.shape()),
                errors::InvalidArgument(
                    "Shapes of both feature gains tensors should match.",
                    " lhs shape= ", fc_gains_lhs_t.shape().DebugString(),
                    " rhs shape= ", fc_gains_rhs_t.shape().DebugString()));

    // Read in info about trees that were dropped.
    Tensor dropped_trees_info_t = context->input(kDropedTreesInfoTensorIdx);
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(dropped_trees_info_t.shape()),
                InvalidArgument("Dropped trees info should be matrix."));

    const auto& dropout_info = dropped_trees_info_t.matrix<float>();

    // Parse the passed in tree ensemble.
    Tensor tree_ensemble_config_t = context->input(kEnsembleToAddTensorIdx);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(tree_ensemble_config_t.shape()),
        errors::InvalidArgument("Tree ensemble config must be a scalar."));
    // Arena increase spatial locality which reduces the average latency to
    // access memory, as working set of pages will be fewer.
    // arena has type proto2::Arena*.
    auto* arena =
        decision_tree_ensemble_resource->mutable_decision_tree_ensemble()
            ->GetArena();
    DecisionTreeEnsembleConfig* ensemble_to_add =
        protobuf::Arena::CreateMessage<DecisionTreeEnsembleConfig>(arena);
    OP_REQUIRES(
        context, ParseProtoUnlimited(ensemble_to_add,
                                     tree_ensemble_config_t.scalar<string>()()),
        errors::InvalidArgument("Unable to parse tree ensemble config."));

    auto* mutable_ensemble =
        decision_tree_ensemble_resource->mutable_decision_tree_ensemble();

    // Read the learning_rate
    Tensor learning_rate_t = context->input(kLearningRateTensorIdx);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(learning_rate_t.shape()),
                InvalidArgument("Learning rate should be a scalar."));

    const float learning_rate = learning_rate_t.scalar<float>()();
    if (learning_rate < kLearningRateEps) {
      return;
    }
    // Prepare current weights vec.
    std::vector<float> current_weights;
    current_weights.reserve(mutable_ensemble->tree_weights_size());
    for (const float weight : mutable_ensemble->tree_weights()) {
      current_weights.push_back(weight);
    }
    const int32 num_dropped = dropped_trees_info_t.dim_size(1);
    std::vector<int> dropped_trees;
    dropped_trees.reserve(num_dropped);
    std::vector<float> dropped_trees_original_weights;
    dropped_trees_original_weights.reserve(num_dropped);
    for (int i = 0; i < num_dropped; ++i) {
      dropped_trees.push_back(dropout_info(0, i));
      dropped_trees_original_weights.push_back(dropout_info(1, i));
    }

    std::vector<int32> num_updates;
    num_updates.reserve(mutable_ensemble->tree_metadata_size());

    for (const auto& meta : mutable_ensemble->tree_metadata()) {
      num_updates.push_back(meta.num_tree_weight_updates());
    }

    // If there was a dropout, come up with tree weights
    const bool was_dropout = !dropped_trees.empty();
    if (was_dropout) {
      // New tree/s will be added to the end of the ensemble's tree list.
      const int32 new_tree_index = current_weights.size();
      DropoutUtils::GetTreesWeightsForAddingTrees(
          dropped_trees, dropped_trees_original_weights, new_tree_index,
          ensemble_to_add->trees_size(), &current_weights, &num_updates);

      // Update the weights of trees according to current weights;
      for (int i = 0; i < mutable_ensemble->trees_size(); ++i) {
        mutable_ensemble->set_tree_weights(i, current_weights[i]);
      }
    }

    // Add the trees from ensemble_to_add to the tree ensemble variable.
    int i = mutable_ensemble->trees_size();
    for (auto& tree : *ensemble_to_add->mutable_trees()) {
      (*mutable_ensemble->add_trees()).Swap(&tree);

      // New trees were updated only once.
      auto* meta = mutable_ensemble->add_tree_metadata();
      meta->set_num_tree_weight_updates(1);

      // When we add complete trees to the ensemble in one step, each tree
      // that's added is final.
      meta->set_is_finalized(true);

      if (was_dropout) {
        mutable_ensemble->add_tree_weights(current_weights[i++]);
      } else {
        mutable_ensemble->add_tree_weights(learning_rate);
      }
    }

    // Update the number of updates.
    if (was_dropout) {
      for (int i = 0; i < num_updates.size(); ++i) {
        mutable_ensemble->mutable_tree_metadata(i)->set_num_tree_weight_updates(
            num_updates[i]);
      }
    }

    // Update feature importance.
    fc_usage_counts_lhs_t.vec<int64>() += fc_usage_counts_rhs_t.vec<int64>();
    fc_gains_lhs_t.vec<float>() += learning_rate * fc_gains_rhs_t.vec<float>();
  }

 private:
  // Input tensor indices.
  // Note that Op definition changes might cause input indices to need
  // changing as well.
  static const int kTreeEnsembleHandleIdx = 0;
  static const int kEnsembleToAddTensorIdx = 1;
  static const int kFeatureColumnUsageCountsHandleIdx = 2;
  static const int kFeatureColumnUsageCountsToAddIdx = 3;
  static const int kFeatureColumnGainsHandleIdx = 4;
  static const int kFeatureColumnGainsToAddIdx = 5;
  static const int kDropedTreesInfoTensorIdx = 6;
  static const int kLearningRateTensorIdx = 7;
};

REGISTER_KERNEL_BUILDER(Name("AddTreesToEnsemble").Device(DEVICE_CPU),
                        AddTreesToEnsembleOp);

}  // namespace tensorflow
