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
#include "tensorflow/core/kernels/tensor_forest/leaf_model.h"
#include "tensorflow/core/kernels/tensor_forest/resources.h"

namespace tensorflow {

class CreateTreeVariableOp : public OpKernel {
 public:
  explicit CreateTreeVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("leaf_model_type", &leaf_model_type_));
    OP_REQUIRES_OK(context, context->GetAttr("leaf_model_type", &num_output_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tree_config_t->shape()),
                errors::InvalidArgument("Tree config must be a scalar."));

    auto* result = new DecisionTreeResource(leaf_model_type_, num_output_);
    if (!ParseProtoUnlimited(result->mutable_decision_tree(),
                             tree_config_t->scalar<string>()())) {
      result->Unref();
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unable to parse tree  config."));
    }

    result->MaybeInitialize();

    // Only create one, if one does not exist already. Report status for all
    // other exceptions.
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }

 private:
  int32 leaf_model_type_;
  int32 num_output_;
};

// Op for serializing a model.
class TreeSerializeOp : public OpKernel {
 public:
  explicit TreeSerializeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
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
class TreeDeserializeOp : public OpKernel {
 public:
  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    auto handle = HandleFromInput(context, 0);
    OP_REQUIRES_OK(context,
                   LookupResource(context, handle, &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);

    const Tensor* tree_config_t;
    OP_REQUIRES_OK(context, context->input("tree_config", &tree_config_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tree_config_t->shape()),
                errors::InvalidArgument("Tree config must be a scalar."));
    // Deallocate all the previous objects on the resource.
    decision_tree_resource->Reset();
    tensorforest::Model* config =
        decision_tree_resource->mutable_decision_tree();
    OP_REQUIRES(context,
                ParseProtoUnlimited(config, tree_config_t->scalar<string>()()),
                errors::InvalidArgument("Unable to parse tree  config."));
    decision_tree_resource->MaybeInitialize();
  }
};

// Op for getting tree size.
class TreeSizeOp : public OpKernel {
 public:
  explicit TreeSizeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DecisionTreeResource* decision_tree_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &decision_tree_resource));
    mutex_lock l(*decision_tree_resource->get_mutex());
    core::ScopedUnref unref_me(decision_tree_resource);
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_t));
    output_t->scalar<int32>()() =
        decision_tree_resource->decision_tree().decision_tree().nodes_size();
  }
};
}  // namespace tensorflow
