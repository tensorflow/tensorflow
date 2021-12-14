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
#include "tensorflow/core/tfrt/eager/function_cache.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"

namespace tfrt {
namespace tf {
namespace {

using tensorflow::Status;
using tensorflow::StatusFromTF_Status;
using tensorflow::TF_StatusPtr;

constexpr char kCpuName[] = "/job:localhost/replica:0/task:0/device:CPU:0";
constexpr char kFunctionName[] = "test_fn";

class CppTests : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(GetParam(), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
  }
};

// Computes `inputs[0] + inputs[1]` and records it on the tape.
tensorflow::Status Add(
    tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
  tensorflow::AbstractOperationPtr add_op(ctx->CreateOperation());

  TF_RETURN_IF_ERROR(add_op.get()->Reset("Add", /*raw_device_name=*/nullptr));

  if (isa<tensorflow::tracing::TracingOperation>(add_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tensorflow::tracing::TracingOperation>(add_op.get())
            ->SetOpName("my_add"));
  }

  TF_RETURN_IF_ERROR(add_op.get()->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(add_op.get()->AddInput(inputs[1]));
  int num_retvals = 1;
  return add_op.get()->Execute(outputs, &num_retvals);
}

// Computes
// return inputs[0] + inputs[1]
tensorflow::Status AddModel(
    tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
  std::vector<tensorflow::AbstractTensorHandle*> add_outputs(1);
  // Compute x+y.
  TF_RETURN_IF_ERROR(Add(ctx, inputs, absl::MakeSpan(add_outputs)));

  outputs[0] = add_outputs[0];
  return tensorflow::Status::OK();
}

tensorflow::AbstractContext* BuildFunction(const char* fn_name) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name, status.get());
  return tensorflow::unwrap(graph_ctx);
}

tensorflow::Status CreateParamsForInputs(
    tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    std::vector<tensorflow::AbstractTensorHandle*>* params) {
  tensorflow::tracing::TracingTensorHandle* handle = nullptr;
  for (auto input : inputs) {
    tensorflow::PartialTensorShape shape;
    TF_RETURN_IF_ERROR(input->Shape(&shape));
    TF_RETURN_IF_ERROR(
        dyn_cast<tensorflow::tracing::TracingContext>(ctx)->AddParameter(
            input->DataType(), shape, &handle));
    params->emplace_back(handle);
  }
  return tensorflow::Status::OK();
}

using Model = std::function<tensorflow::Status(
    tensorflow::AbstractContext*,
    absl::Span<tensorflow::AbstractTensorHandle* const>,
    absl::Span<tensorflow::AbstractTensorHandle*>)>;

tensorflow::Status PrepareFunction(
    Model model, tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
  tensorflow::core::RefCountPtr<tensorflow::AbstractFunction> scoped_func;

  tensorflow::AbstractContextPtr func_ctx(BuildFunction(kFunctionName));
  std::vector<tensorflow::AbstractTensorHandle*> func_inputs;
  func_inputs.reserve(inputs.size());
  TF_RETURN_IF_ERROR(
      CreateParamsForInputs(func_ctx.get(), inputs, &func_inputs));
  tensorflow::OutputList output_list;
  output_list.expected_num_outputs = outputs.size();
  output_list.outputs.resize(outputs.size());
  TF_RETURN_IF_ERROR(model(func_ctx.get(), absl::MakeSpan(func_inputs),
                           absl::MakeSpan(output_list.outputs)));
  for (auto func_input : func_inputs) {
    func_input->Unref();
  }
  tensorflow::AbstractFunction* func = nullptr;
  TF_RETURN_IF_ERROR(
      dyn_cast<tensorflow::tracing::TracingContext>(func_ctx.get())
          ->Finalize(&output_list, &func));
  scoped_func.reset(func);
  for (auto output : output_list.outputs) {
    output->Unref();
  }
  TF_RETURN_IF_ERROR(ctx->RegisterFunction(func));

  return tensorflow::Status::OK();
}

tensorflow::Status BuildImmediateExecutionContext(
    bool use_tfrt, tensorflow::AbstractContext** ctx) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(opts, use_tfrt);
  *ctx = tensorflow::unwrap(TF_NewEagerExecutionContext(opts, status.get()));
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(status.get()));
  TFE_DeleteContextOptions(opts);
  return tensorflow::Status::OK();
}

tensorflow::Status TestScalarTensorHandle(
    tensorflow::AbstractContext* ctx, float value,
    tensorflow::AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, value);
  *tensor = tensorflow::unwrap(
      TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return tensorflow::Status::OK();
}

TEST_P(CppTests, TestFunctionCacheWithAdd) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  tensorflow::AbstractContextPtr ctx;
  {
    tensorflow::AbstractContext* ctx_raw = nullptr;
    tensorflow::Status s = BuildImmediateExecutionContext(true, &ctx_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  tensorflow::AbstractTensorHandlePtr x;
  {
    tensorflow::AbstractTensorHandle* x_raw = nullptr;
    tensorflow::Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  tensorflow::AbstractTensorHandlePtr y;
  {
    tensorflow::AbstractTensorHandle* y_raw = nullptr;
    tensorflow::Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &y_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Pseudo-code:
  // outputs = x + y
  tensorflow::Status s;
  std::vector<tensorflow::AbstractTensorHandle*> outputs(1);
  s = PrepareFunction(AddModel, ctx.get(), {x.get(), y.get()},
                      absl::MakeSpan(outputs));

  ::tfrt::tf::FunctionCache cache;
  ::tfrt::tf::ContextInterface* tfrt_ctx =
      static_cast<::tfrt::tf::ContextInterface*>(ctx.get());
  ::tfrt::CoreRuntime* corert = tfrt_ctx->GetCoreRuntime();
  tensorflow::EagerContext* eager_ctx = tfrt_ctx->GetEagerContext();

  // Cache is empty initially.
  ASSERT_EQ(cache.Size(), 0);
  ASSERT_EQ(cache.Contains(kFunctionName, kCpuName), false);

  tensorflow::DeviceSet dev_set;
  const tensorflow::DeviceMgr* device_mgr =
      tfrt_ctx->GetEagerContext()->local_device_mgr();
  for (auto d : device_mgr->ListDevices()) dev_set.AddDevice(d);
  auto& device = corert->GetHostContext()->GetHostDevice();
  const Device* input_devices[2] = {&device, &device};
  auto req_ctx = RequestContextBuilder(corert->GetHostContext(),
                                       /*resource_context=*/nullptr)
                     .build();
  ExecutionContext exec_ctx(std::move(*req_ctx));

  auto request_ctx_fn =
      [host = corert->GetHostContext()](
          tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
          RCReference<RequestContext>* request_ctx) {
        *request_ctx =
            std::move(*RequestContextBuilder(host,
                                             /*resource_context=*/nullptr)
                           .build());
        return Status::OK();
      };

  // Inserts a new cache entry.
  FunctionCache::FunctionCacheResult result;
  TF_ASSERT_OK(cache.GetOrAddFunction(
      kFunctionName, kCpuName, dev_set, eager_ctx, corert, request_ctx_fn,
      /*loc=*/{}, tensorflow::TfrtFunctionCompileOptions(), input_devices,
      &result));
  ASSERT_NE(result.function_state.get(), nullptr);
  // Cache contains the inserted entry now.
  ASSERT_EQ(cache.Contains(kFunctionName, kCpuName), true);

  // There's one entry in the cache.
  ASSERT_EQ(cache.Size(), 1);

  // This lookup is a cache hit.
  TF_ASSERT_OK(cache.GetOrAddFunction(
      kFunctionName, kCpuName, dev_set, eager_ctx, corert, request_ctx_fn,
      /*loc=*/{}, tensorflow::TfrtFunctionCompileOptions(), input_devices,
      &result));
  ASSERT_NE(result.function_state.get(), nullptr);
  // Cache hit doesn't create new entry in the cache.
  ASSERT_EQ(cache.Size(), 1);

  // Add another entry with the same function name but different device name.
  // This lookup is a cache miss.
  TF_ASSERT_OK(cache.GetOrAddFunction(
      kFunctionName, "", dev_set, eager_ctx, corert, request_ctx_fn,
      /*loc=*/{}, tensorflow::TfrtFunctionCompileOptions(), input_devices,
      &result));
  ASSERT_NE(result.function_state.get(), nullptr);
  // Cache miss adds a new entry in the cache.
  ASSERT_EQ(cache.Size(), 2);

  cache.RemoveFunction(kFunctionName);

  // RemoveFunction removes all entries in the cache since they have the same
  // function name.
  ASSERT_EQ(cache.Size(), 0);
}

INSTANTIATE_TEST_SUITE_P(UnifiedCAPI, CppTests, ::testing::Values("graphdef"));

}  // namespace
}  // namespace tf
}  // namespace tfrt
