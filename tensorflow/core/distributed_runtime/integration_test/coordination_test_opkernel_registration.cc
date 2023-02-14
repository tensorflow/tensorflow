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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/tsl/distributed_runtime/coordination/coordination_service_error_util.h"

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
    .Attr("blocking: bool = true")
    .Output("value: string")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"doc(
Test op getting distributed configs using coordination service.

blocking: If true, wait for the config key to become available and return its
          value; otherwise, error out of the key does not exist.
)doc");

// Kernel that gets distributed configures using coordination service.
class TestGetConfigKeyValueOp : public OpKernel {
 public:
  explicit TestGetConfigKeyValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocking", &blocking_));
  }

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
    auto status_or_val = blocking_ ? coord_agent->GetKeyValue(config_key)
                                   : coord_agent->TryGetKeyValue(config_key);
    OP_REQUIRES_OK(ctx, status_or_val.status());

    Tensor* val_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("value", key_tensor->shape(), &val_tensor));
    auto value = val_tensor->scalar<tstring>()();
    val_tensor->scalar<tstring>()() = status_or_val.value();
  }

 private:
  bool blocking_;
};
REGISTER_KERNEL_BUILDER(Name("TestGetConfigKeyValue").Device(DEVICE_DEFAULT),
                        TestGetConfigKeyValueOp);

REGISTER_OP("TestReportErrorToCluster")
    .Input("error_code: int32")
    .Input("error_message: string")
    .SetIsStateful()  // side-effective op
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"doc(
Test op report errors to coordination service.
)doc");

// Kernel that reports errors to coordination service.
class TestReportErrorToClusterOp : public OpKernel {
 public:
  explicit TestReportErrorToClusterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* error_code_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("error_code", &error_code_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(error_code_tensor->shape()),
                errors::InvalidArgument("Error code must be scalar."));
    const int& error_code = error_code_tensor->scalar<int32_t>()();
    const Tensor* error_message_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("error_message", &error_message_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(error_message_tensor->shape()),
                errors::InvalidArgument("Error message must be scalar."));
    const string& error_message = error_message_tensor->scalar<tstring>()();
    LOG(INFO) << "TestReportErrorToClusterOp error_code=" << error_code
              << " error_message=" << error_message;
    auto* coord_agent = ctx->coordination_service_agent();
    if (coord_agent == nullptr || !coord_agent->IsInitialized()) {
      ctx->SetStatus(
          errors::Internal("Coordination service agent is not instantiated or "
                           "initialized properly."));
      return;
    }
    tensorflow::Status s(static_cast<tensorflow::error::Code>(error_code),
                         error_message);
    s.SetPayload(tsl::CoordinationErrorPayloadKey(),
                 absl::Cord("testing error payload"));
    OP_REQUIRES_OK(ctx, coord_agent->ReportError(s));
  }
};
REGISTER_KERNEL_BUILDER(Name("TestReportErrorToCluster").Device(DEVICE_DEFAULT),
                        TestReportErrorToClusterOp);

}  // namespace
}  // namespace tensorflow
