/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/kernels/tensor_forest/resources.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

class TensorForestCreateTreeVariableOp : public OpKernel {
 public:
  explicit TensorForestCreateTreeVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));

    auto* const result = new TensorForestTreeResource();

    if (!result->InitFromSerialized(tree_config_t->scalar<tstring>()())) {
      result->Unref();
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse tree config."));
    }

    // Only create one, if one does not exist already. Report status for all
    // other exceptions.
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }
};

// Op for serializing a model.
class TensorForestTreeSerializeOp : public OpKernel {
 public:
  explicit TensorForestTreeSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<TensorForestTreeResource> decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(), &output_config_t));
    output_config_t->scalar<tstring>()() =
        decision_tree_resource->decision_tree().SerializeAsString();
  }
};

// Op for deserializing a tree variable from a checkpoint.
class TensorForestTreeDeserializeOp : public OpKernel {
 public:
  explicit TensorForestTreeDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<TensorForestTreeResource> decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));

    mutex_lock l(*decision_tree_resource->get_mutex());
    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));

    // Deallocate all the previous objects on the resource.
    decision_tree_resource->Reset();

    if (!decision_tree_resource->InitFromSerialized(
            tree_config_t->scalar<tstring>()())) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse tree config."));
    }
  }
};

// Op for getting tree size.
class TensorForestTreeSizeOp : public OpKernel {
 public:
  explicit TensorForestTreeSizeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<TensorForestTreeResource> decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_t));
    output_t->scalar<int32>()() = decision_tree_resource->get_size();
  }
};

REGISTER_RESOURCE_HANDLE_KERNEL(TensorForestTreeResource);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestTreeIsInitializedOp").Device(DEVICE_CPU),
    IsResourceInitialized<TensorForestTreeResource>);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestCreateTreeVariable").Device(DEVICE_CPU),
    TensorForestCreateTreeVariableOp);

REGISTER_KERNEL_BUILDER(Name("TensorForestTreeSerialize").Device(DEVICE_CPU),
                        TensorForestTreeSerializeOp);

REGISTER_KERNEL_BUILDER(Name("TensorForestTreeDeserialize").Device(DEVICE_CPU),
                        TensorForestTreeDeserializeOp);

REGISTER_KERNEL_BUILDER(Name("TensorForestTreeSize").Device(DEVICE_CPU),
                        TensorForestTreeSizeOp);

}  // namespace tensorflow
