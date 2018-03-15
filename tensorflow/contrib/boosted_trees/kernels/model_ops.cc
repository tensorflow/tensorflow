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
    if (!result->InitFromSerialized(tree_ensemble_config_t->scalar<string>()(),
                                    stamp_token)) {
      result->Unref();
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument("Unable to parse tree ensemble config."));
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
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    tf_shared_lock l(*ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(ensemble_resource);
    Tensor* output_stamp_token_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &output_stamp_token_t));
    output_stamp_token_t->scalar<int64>()() = ensemble_resource->stamp();
  }
};

// Op for serializing a model.
class TreeEnsembleSerializeOp : public OpKernel {
 public:
  explicit TreeEnsembleSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    tf_shared_lock l(*ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(ensemble_resource);
    Tensor* output_stamp_token_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &output_stamp_token_t));
    output_stamp_token_t->scalar<int64>()() = ensemble_resource->stamp();
    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape(), &output_config_t));
    output_config_t->scalar<string>()() =
        ensemble_resource->SerializeAsString();
  }
};

// Op for deserializing a tree ensemble variable from a checkpoint.
class TreeEnsembleDeserializeOp : public OpKernel {
 public:
  explicit TreeEnsembleDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    mutex_lock l(*ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(ensemble_resource);

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Get the tree ensemble config.
    const Tensor* tree_ensemble_config_t;
    OP_REQUIRES_OK(context, context->input("tree_ensemble_config",
                                           &tree_ensemble_config_t));
    // Deallocate all the previous objects on the resource.
    ensemble_resource->Reset();
    OP_REQUIRES(
        context,
        ensemble_resource->InitFromSerialized(
            tree_ensemble_config_t->scalar<string>()(), stamp_token),
        errors::InvalidArgument("Unable to parse tree ensemble config."));
  }
};

class TreeEnsembleUsedHandlersOp : public OpKernel {
 public:
  explicit TreeEnsembleUsedHandlersOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_all_handlers", &num_handlers_));
  }

  void Compute(OpKernelContext* context) override {
    boosted_trees::models::DecisionTreeEnsembleResource* ensemble_resource;

    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &ensemble_resource));
    tf_shared_lock l(*ensemble_resource->get_mutex());
    core::ScopedUnref unref_me(ensemble_resource);

    // Get the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // Only the Chief should run this Op and it is guaranteed to be in
    // a consistent state so the stamps must always match.
    CHECK(ensemble_resource->is_stamp_valid(stamp_token));

    Tensor* output_used_handlers_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("used_handlers_mask", {num_handlers_},
                                          &output_used_handlers_t));
    auto output_used_handlers = output_used_handlers_t->vec<bool>();

    Tensor* output_num_used_handlers_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_used_handlers", {},
                                            &output_num_used_handlers_t));
    int handler_idx = 0;
    std::vector<int64> used_handlers = ensemble_resource->GetUsedHandlers();
    output_num_used_handlers_t->scalar<int64>()() = used_handlers.size();
    for (int64 i = 0; i < num_handlers_; ++i) {
      if (handler_idx >= used_handlers.size() ||
          used_handlers[handler_idx] > i) {
        output_used_handlers(i) = false;
      } else {
        OP_REQUIRES(context, used_handlers[handler_idx] == i,
                    errors::InvalidArgument("Handler IDs should be sorted."));
        ++handler_idx;
        output_used_handlers(i) = true;
      }
    }
  }

 private:
  int64 num_handlers_;
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

REGISTER_KERNEL_BUILDER(Name("TreeEnsembleUsedHandlers").Device(DEVICE_CPU),
                        TreeEnsembleUsedHandlersOp);
}  // namespace boosted_trees
}  // namespace tensorflow
