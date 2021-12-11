/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

REGISTER_OP("TestSetConfigKeyValue")
    .Input("key: string")
    .Input("value: string")
    .SetIsStateful()  // side-effective op
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"doc(
Test op setting distributed configs using coordination service.
)doc");

// Kernel that sets distributed configures using coordination service.
class TestSetConfigKeyValueOp : public OpKernel {
 public:
  explicit TestSetConfigKeyValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor* key_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(key_tensor->shape()),
                errors::InvalidArgument("Key must be scalar."));
    const string& config_key = key_tensor->scalar<tstring>()();
    const Tensor* val_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("value", &val_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(key_tensor->shape()),
                errors::InvalidArgument("Value must be scalar."));
    const string& config_value = val_tensor->scalar<tstring>()();
    LOG(INFO) << "TestSetConfigKeyValueOp key=" << config_key
              << "value=" << config_value;
    auto* coord_agent = ctx->coordination_service_agent();
    if (coord_agent == nullptr || !coord_agent->IsInitialized()) {
      ctx->SetStatus(
          errors::Internal("Coordination service agent is not instantiated or "
                           "initialized properly."));
      return;
    }
    OP_REQUIRES_OK(ctx, coord_agent->InsertKeyValue(config_key, config_value));
  }
};
REGISTER_KERNEL_BUILDER(Name("TestSetConfigKeyValue").Device(DEVICE_DEFAULT),
                        TestSetConfigKeyValueOp);

REGISTER_OP("TestGetConfigKeyValue")
    .Input("key: string")
    .Output("value: string")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"doc(
Test op getting distributed configs using coordination service.
)doc");

// Kernel that gets distributed configures using coordination service.
class TestGetConfigKeyValueOp : public OpKernel {
 public:
  explicit TestGetConfigKeyValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor* key_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(key_tensor->shape()),
                errors::InvalidArgument("Key must be scalar."));
    const string& config_key = key_tensor->scalar<tstring>()();
    LOG(INFO) << "TestGetConfigKeyValueOp key=" << config_key;

    auto* coord_agent = ctx->coordination_service_agent();
    if (coord_agent == nullptr || !coord_agent->IsInitialized()) {
      ctx->SetStatus(
          errors::Internal("Coordination service agent is not instantiated or "
                           "initialized properly."));
      return;
    }
    auto status_or_val = coord_agent->GetKeyValue(config_key);
    OP_REQUIRES_OK(ctx, status_or_val.status());

    Tensor* val_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("value", key_tensor->shape(), &val_tensor));
    auto value = val_tensor->scalar<tstring>()();
    val_tensor->scalar<tstring>()() = status_or_val.ValueOrDie();
  }
};
REGISTER_KERNEL_BUILDER(Name("TestGetConfigKeyValue").Device(DEVICE_DEFAULT),
                        TestGetConfigKeyValueOp);

}  // namespace
}  // namespace tensorflow
