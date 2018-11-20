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

namespace tensorflow {

class TensorForestCreateTreeVariableOp : public OpKernel {
 public:
  explicit TensorForestCreateTreeVariableOp(OpKernelConstruction* context)
      : OpKernel(context){};

  void Compute(OpKernelContext* context) override {
    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));

    auto* const result = new TensorForestTreeResource();

    if (!result->InitFromSerialized(tree_config_t->scalar<string>()())) {
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
    TensorForestTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);
    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(), &output_config_t));
    output_config_t->scalar<string>()() =
        decision_tree_resource->decision_tree().SerializeAsString();
  }
};

// Op for deserializing a tree variable from a checkpoint.
class TensorForestTreeDeserializeOp : public OpKernel {
 public:
  explicit TensorForestTreeDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    TensorForestTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));

    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));

    // Deallocate all the previous objects on the resource.
    decision_tree_resource->Reset();

    if (!decision_tree_resource->InitFromSerialized(
            tree_config_t->scalar<string>()())) {
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
    TensorForestTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_t));
    output_t->scalar<int32>()() = decision_tree_resource->get_size();
  }
};

class TensorForestCreateFertileStatsVariableOp : public OpKernel {
 public:
  explicit TensorForestCreateFertileStatsVariableOp(
      OpKernelConstruction* context)
      : OpKernel(context){};

  void Compute(OpKernelContext* context) override {
    const Tensor* stats_config_t;
    OP_REQUIRES_OK(context, context->input("stats_config", &stats_config_t));
    auto* result = new TensorForestFertileStatsResource();

    if (!result->InitFromSerialized(stats_config_t->scalar<string>()())) {
      result->Unref();
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse stats config."));
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
class TensorForestFertileStatsSerializeOp : public OpKernel {
 public:
  explicit TensorForestFertileStatsSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorForestFertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &fertile_stats_resource));
    mutex_lock l(*fertile_stats_resource->get_mutex());
    core::ScopedUnref unref_me(fertile_stats_resource);

    Tensor* output_config_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape(), &output_config_t));
    output_config_t->scalar<string>()() =
        fertile_stats_resource->fertile_stats().SerializeAsString();
  }
};

// Op for deserializing a stats variable from a checkpoint.
class TensorForestFertileStatsDeserializeOp : public OpKernel {
 public:
  explicit TensorForestFertileStatsDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorForestFertileStatsResource* fertile_stats_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &fertile_stats_resource));
    mutex_lock l(*fertile_stats_resource->get_mutex());
    core::ScopedUnref unref_me(fertile_stats_resource);

    const Tensor* stats_config_t;
    OP_REQUIRES_OK(context, context->input("stats_config", &stats_config_t));

    // Deallocate all the previous objects on the resource.
    fertile_stats_resource->Reset();

    if (!fertile_stats_resource->InitFromSerialized(
            stats_config_t->scalar<string>()())) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse stats config."));
    }
  }
};

REGISTER_RESOURCE_HANDLE_KERNEL(TensorForestTreeResource);

REGISTER_RESOURCE_HANDLE_KERNEL(TensorForestFertileStatsResource);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestTreeIsInitializedOp").Device(DEVICE_CPU),
    IsResourceInitialized<TensorForestTreeResource>);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestFertileStatsIsInitializedOp").Device(DEVICE_CPU),
    IsResourceInitialized<TensorForestFertileStatsResource>);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestCreateTreeVariable").Device(DEVICE_CPU),
    TensorForestCreateTreeVariableOp);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestCreateFertileStatsVariable").Device(DEVICE_CPU),
    TensorForestCreateFertileStatsVariableOp);

REGISTER_KERNEL_BUILDER(Name("TensorForestTreeSerialize").Device(DEVICE_CPU),
                        TensorForestTreeSerializeOp);

REGISTER_KERNEL_BUILDER(Name("TensorForestTreeDeserialize").Device(DEVICE_CPU),
                        TensorForestTreeDeserializeOp);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestFertileStatsSerialize").Device(DEVICE_CPU),
    TensorForestFertileStatsSerializeOp);

REGISTER_KERNEL_BUILDER(
    Name("TensorForestFertileStatsDeserialize").Device(DEVICE_CPU),
    TensorForestFertileStatsDeserializeOp);

REGISTER_KERNEL_BUILDER(Name("TensorForestTreeSize").Device(DEVICE_CPU),
                        TensorForestTreeSizeOp);

}  // namespace tensorflow
