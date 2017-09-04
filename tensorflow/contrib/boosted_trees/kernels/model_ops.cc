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

#include "tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"
#include "tensorflow/contrib/boosted_trees/resources/decision_tree_ensemble_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace boosted_trees {

using boosted_trees::models::DecisionTreeEnsembleResource;

// Creates a tree ensemble variable.
class CreateTreeEnsembleVariableOp : public OpKernel {
 public:
  explicit CreateTreeEnsembleVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Get the tree ensemble config.
    const Tensor* tree_ensemble_config_t;
    OP_REQUIRES_OK(context, context->input("tree_ensemble_config",
                                           &tree_ensemble_config_t));
    auto* result = new boosted_trees::models::DecisionTreeEnsembleResource();
    result->set_stamp(stamp_token);
    if (!ParseProtoUnlimited(result->mutable_decision_tree_ensemble(),
                             tree_ensemble_config_t->scalar<string>()())) {
      result->Unref();
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unable to parse tree ensemble config."));
    }

    // Only create one, if one does not exist already. Report status for all
    // other exceptions.
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }
};

// Op for retrieving a model stamp token without having to serialize.
class TreeEnsembleStampTokenOp : public OpKernel {
 public:
  explicit TreeEnsembleStampTokenOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    boosted_trees::models::DecisionTreeEnsembleResource*
        decision_tree_ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_ensemble_resource));
    tf_shared_lock l(*decision_tree_ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_ensemble_resource);
    Tensor* output_stamp_token_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &output_stamp_token_t));
    output_stamp_token_t->scalar<int64>()() =
        decision_tree_ensemble_resource->stamp();
  }
};

// Op for serializing a model.
class TreeEnsembleSerializeOp : public OpKernel {
 public:
  explicit TreeEnsembleSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    boosted_trees::models::DecisionTreeEnsembleResource*
        decision_tree_ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_ensemble_resource));
    tf_shared_lock l(*decision_tree_ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_ensemble_resource);
    Tensor* output_stamp_token_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &output_stamp_token_t));
    output_stamp_token_t->scalar<int64>()() =
        decision_tree_ensemble_resource->stamp();
    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape(), &output_config_t));
    output_config_t->scalar<string>()() =
        decision_tree_ensemble_resource->decision_tree_ensemble()
            .SerializeAsString();
  }
};

// Op for deserializing a tree ensemble variable from a checkpoint.
class TreeEnsembleDeserializeOp : public OpKernel {
 public:
  explicit TreeEnsembleDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    boosted_trees::models::DecisionTreeEnsembleResource*
        decision_tree_ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_ensemble_resource));
    mutex_lock l(*decision_tree_ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_ensemble_resource);

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Get the tree ensemble config.
    const Tensor* tree_ensemble_config_t;
    OP_REQUIRES_OK(context, context->input("tree_ensemble_config",
                                           &tree_ensemble_config_t));
    // Deallocate all the previous objects on the resource.
    decision_tree_ensemble_resource->Reset();
    decision_tree_ensemble_resource->set_stamp(stamp_token);
    boosted_trees::trees::DecisionTreeEnsembleConfig* config =
        decision_tree_ensemble_resource->mutable_decision_tree_ensemble();
    OP_REQUIRES(
        context,
        ParseProtoUnlimited(config, tree_ensemble_config_t->scalar<string>()()),
        errors::InvalidArgument("Unable to parse tree ensemble config."));
  }
};

REGISTER_RESOURCE_HANDLE_KERNEL(DecisionTreeEnsembleResource);

REGISTER_KERNEL_BUILDER(
    Name("TreeEnsembleIsInitializedOp").Device(DEVICE_CPU),
    IsResourceInitialized<boosted_trees::models::DecisionTreeEnsembleResource>);

REGISTER_KERNEL_BUILDER(Name("CreateTreeEnsembleVariable").Device(DEVICE_CPU),
                        CreateTreeEnsembleVariableOp);

REGISTER_KERNEL_BUILDER(Name("TreeEnsembleStampToken").Device(DEVICE_CPU),
                        TreeEnsembleStampTokenOp);

REGISTER_KERNEL_BUILDER(Name("TreeEnsembleSerialize").Device(DEVICE_CPU),
                        TreeEnsembleSerializeOp);

REGISTER_KERNEL_BUILDER(Name("TreeEnsembleDeserialize").Device(DEVICE_CPU),
                        TreeEnsembleDeserializeOp);

}  // namespace boosted_trees
}  // namespace tensorflow
