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
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/tfrt/fallback/device_with_custom_allocator.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/interpreter_testutil.h"
#include "tensorflow/core/tfrt/mlrt/kernel/batch_kernel.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {
namespace {

struct TestOp : mlrt::KernelFrame {
  static constexpr char kName[] = "test";
  using KernelFrame::KernelFrame;
  void Invoke() {}
};

TEST(KernelTest, OptionalRegistry) {
  GetTfMlrtOptionalKernelRegistry().Register<TestOp>();

  mlrt::KernelRegistry registry;

  RegisterTfMlrtKernels(registry);

  EXPECT_TRUE(registry.Get(TestOp::kName));
}

mlrt::bc::Buffer CreateExecutableForCreateExecuteOp(absl::string_view op_name) {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  auto kernel_names_ctor = executable_ctor.construct_kernel_names(3);
  kernel_names_ctor.ConstructAt(0, "tf_mlrt.createop");
  kernel_names_ctor.ConstructAt(1, "tf_mlrt.executeop");
  kernel_names_ctor.ConstructAt(2, "return");

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(2));

  attributes.Add("node_def_str",
                 absl::Substitute(
                     R"pb(name: "$0"
                          op: "$0"
                          input: "dummy_arg"
                          input: "dummy_arg"
                          device: "/job:localhost/replica:0/task:0/device:CPU:0"
                          attr {
                            key: "T"
                            value { type: DT_INT32 }
                          })pb",
                     op_name));

  attributes.Add("op_key", 0);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");
  function_ctor.set_num_regs(2);
  function_ctor.construct_input_regs(1).Assign({0});
  function_ctor.construct_output_regs(1).Assign({1});

  auto kernels_ctor = function_ctor.construct_kernels(3);

  auto createop_ctor = kernels_ctor.ConstructAt(0);
  createop_ctor.set_code(0);
  createop_ctor.construct_arguments(0);
  createop_ctor.construct_results(0);
  createop_ctor.construct_attributes(2).Assign(
      {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});

  auto executeop_ctor = kernels_ctor.ConstructAt(1);
  executeop_ctor.set_code(1);
  executeop_ctor.construct_arguments(2).Assign({0, 0});
  executeop_ctor.construct_results(1).Assign({1});
  executeop_ctor.construct_attributes(2).Assign(
      {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
  executeop_ctor.construct_last_uses(2).Assign({0, 0});

  auto return_ctor = kernels_ctor.ConstructAt(2);
  return_ctor.set_code(2);
  return_ctor.construct_arguments(1).Assign({1});
  return_ctor.construct_results(0);

  return buffer;
}

TEST(KernelTest, CreateExecuteOp) {
  auto buffer = CreateExecutableForCreateExecuteOp("AddV2");

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext execution_context(&loaded_executable);

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  int32_t input = 100;
  tensorflow::Tensor input_tensor(input);
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value result;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  TF_ASSERT_OK(execution_context.status());

  int32_t output = 200;
  tensorflow::Tensor expected(output);

  tensorflow::test::ExpectEqual(
      result.Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

mlrt::bc::Buffer CreateExecutableForCreateExecuteOpCustomDevice(
    absl::string_view op_name) {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;

  std::vector<std::string> kernel_names = {"tf_mlrt.createop",
                                           "tf_mlrt.executeop.device",
                                           "tf_mlrt.executeop", "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(2));

  attributes.Add("node_def_str",
                 absl::Substitute(
                     R"pb(name: "$0"
                          op: "$0"
                          device: "/job:localhost/replica:0/task:0/device:CPU:0"
                          attr {
                            key: "T"
                            value { type: DT_STRING }
                          })pb",
                     op_name));

  attributes.Add("op_key", 0);

  mlrt::testing::SymbolTable regs;

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");
  function_ctor.construct_input_regs(1).Assign({regs.Def("device")});
  auto kernels_ctor = function_ctor.construct_kernels(4);

  auto createop_ctor = kernels_ctor.ConstructAt(0);
  createop_ctor.set_code(kernels.Use("tf_mlrt.createop"));
  createop_ctor.construct_arguments(0);
  createop_ctor.construct_results(0);
  createop_ctor.construct_attributes(2).Assign(
      {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});

  {
    auto executeop_ctor = kernels_ctor.ConstructAt(1);
    executeop_ctor.set_code(kernels.Use("tf_mlrt.executeop.device"));
    executeop_ctor.construct_arguments(1).Assign({regs.Use("device")});
    executeop_ctor.construct_results(1).Assign(
        {regs.Def("custom_device_name")});
    executeop_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
    executeop_ctor.construct_last_uses(1).Assign({true});
  }

  {
    auto executeop_ctor = kernels_ctor.ConstructAt(2);
    executeop_ctor.set_code(kernels.Use("tf_mlrt.executeop"));
    executeop_ctor.construct_results(1).Assign({regs.Def("device_name")});
    executeop_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
  }

  auto return_ctor = kernels_ctor.ConstructAt(3);
  return_ctor.set_code(kernels.Use("return"));
  return_ctor.construct_arguments(2).Assign(
      {regs.Use("custom_device_name"), regs.Use("device_name")});

  function_ctor.construct_output_regs(2).Assign(
      {regs.Use("custom_device_name"), regs.Use("device_name")});
  function_ctor.set_num_regs(regs.size());

  return buffer;
}

REGISTER_OP("TestDevice")
    .Output("z: T")
    .Attr("T: {string}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

class TestDeviceKernel : public OpKernel {
 public:
  explicit TestDeviceKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    ctx->set_output(0, tensorflow::Tensor(ctx->device()->name()));
  }
};

REGISTER_KERNEL_BUILDER(Name("TestDevice").Device(DEVICE_CPU),
                        TestDeviceKernel);

TEST(KernelTest, CreateExecuteDeviceOp) {
  auto buffer = CreateExecutableForCreateExecuteOpCustomDevice("TestDevice");

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext execution_context(&loaded_executable);

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  class TestDevice : public tfrt_stub::DeviceWithCustomAllocator {
   public:
    using DeviceWithCustomAllocator::DeviceWithCustomAllocator;

    const std::string& name() const override { return name_; }

   private:
    std::string name_ = "test_device";
  };

  mlrt::Value arg;
  arg.Set<std::shared_ptr<Device>>(std::make_shared<TestDevice>(
      fallback_request_state.cpu_device(),
      fallback_request_state.cpu_device()->GetAllocator({})));
  mlrt::Value results[2];

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(results));
  mlrt::Execute(execution_context);

  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(
      results[0].Get<tfrt_stub::FallbackTensor>().tensor().scalar<tstring>()(),
      "test_device");
  EXPECT_EQ(
      results[1].Get<tfrt_stub::FallbackTensor>().tensor().scalar<tstring>()(),
      fallback_request_state.cpu_device()->name());
}

REGISTER_OP("TestError")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

class TestErrorKernel : public OpKernel {
 public:
  explicit TestErrorKernel(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    ctx->SetStatus(absl::InternalError("test error"));
  }
};

REGISTER_KERNEL_BUILDER(Name("TestError").Device(DEVICE_CPU), TestErrorKernel);

TEST(KernelTest, CreateExecuteOpError) {
  auto buffer = CreateExecutableForCreateExecuteOp("TestError");

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext execution_context(&loaded_executable);

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  int32_t input = 100;
  tensorflow::Tensor input_tensor(input);
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value result;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  EXPECT_THAT(
      execution_context.status(),
      ::tsl::testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

REGISTER_OP("TestAsyncIdentity")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {int32}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

class TestAsyncIdentityKernel : public AsyncOpKernel {
 public:
  explicit TestAsyncIdentityKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    DCHECK(ctx->device()->tensorflow_cpu_worker_threads()->workers);
    ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
        [done = std::move(done), ctx]() {
          const Tensor& x = ctx->input(0);
          ctx->set_output(0, x);
          done();
        });
  }

 private:
  TestAsyncIdentityKernel(const TestAsyncIdentityKernel&) = delete;
  void operator=(const TestAsyncIdentityKernel&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("TestAsyncIdentity").Device(DEVICE_CPU),
                        TestAsyncIdentityKernel);

mlrt::bc::Buffer CreateExecutableForCreateAsyncExecuteOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;

  std::vector<std::string> kernel_names = {"tf_mlrt.createop",
                                           "tf_mlrt.async_executeop",
                                           "tf_mlrt.await_all", "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(2));

  attributes.Add("node_def_str",
                 R"pb(name: "TestAsyncIdentity"
                      op: "TestAsyncIdentity"
                      input: "dummy_arg"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "T"
                        value { type: DT_INT32 }
                      })pb");

  attributes.Add("op_key", 0);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  mlrt::testing::SymbolTable regs;

  function_ctor.construct_input_regs(1).Assign({regs.Def("input")});

  auto kernels_ctor = function_ctor.construct_kernels(5);

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.createop"));
    kernel_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(1);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.async_executeop"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"input", "input"}));
    kernel_ctor.construct_results(1).Assign({regs.Def("result_future_0")});
    kernel_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
    kernel_ctor.construct_last_uses(2).Assign({0, 0});
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(2);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.async_executeop"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"input", "input"}));
    kernel_ctor.construct_results(1).Assign({regs.Def("result_future_1")});
    kernel_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
    kernel_ctor.construct_last_uses(2).Assign({0, 0});
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(3);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.await_all"));
    kernel_ctor.construct_arguments(2).Assign(
        regs.Use({"result_future_0", "result_future_1"}));
    kernel_ctor.construct_last_uses(2).Assign({true, true});
    kernel_ctor.construct_results(2).Assign(
        regs.Def(absl::Span<const std::string>{"result_0", "result_1"}));
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(4);
    kernel_ctor.set_code(kernels.Use("return"));
    kernel_ctor.construct_arguments(2).Assign(
        regs.Use({"result_0", "result_1"}));
  }

  function_ctor.set_num_regs(regs.size());
  function_ctor.construct_output_regs(2).Assign(
      regs.Use({"result_0", "result_1"}));

  return buffer;
}

TEST(KernelTest, CreateAsyncExecuteOp) {
  auto buffer = CreateExecutableForCreateAsyncExecuteOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  int32_t input = 100;
  tensorflow::Tensor input_tensor(input);
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  std::vector<mlrt::Value> results(2);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });
  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(results));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  int32_t output = 100;
  tensorflow::Tensor expected(output);

  tensorflow::test::ExpectEqual(
      results[0].Get<tfrt_stub::FallbackTensor>().tensor(), expected);
  tensorflow::test::ExpectEqual(
      results[1].Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

TEST(KernelTest, AsyncExecuteOpCanCancell) {
  auto buffer = CreateExecutableForCreateAsyncExecuteOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::RCReference<tfrt::CancellationContext> cancellation_context =
      tfrt::TakeRef(new tfrt::CancellationContext());

  tfrt::ResourceContext resource_context;

  auto tf_context = std::make_unique<Context>(
      &fallback_request_state, &resource_context, cancellation_context.get());
  execution_context.AddUserContext(std::move(tf_context));

  int32_t input = 100;
  tensorflow::Tensor input_tensor(input);
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  std::vector<mlrt::Value> results(2);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });
  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(results));

  cancellation_context->Cancel();

  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_THAT(execution_context.status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kCancelled));
}

mlrt::bc::Buffer CreateExecutableForSetGetResourceOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  auto kernel_names_ctor = executable_ctor.construct_kernel_names(3);
  kernel_names_ctor.ConstructAt(0, "tf_mlrt.set_resource");
  kernel_names_ctor.ConstructAt(1, "tf_mlrt.get_resource");
  kernel_names_ctor.ConstructAt(2, "return");

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(2));

  attributes.Add("index", static_cast<int64_t>(0));

  {
    mlrt::bc::Buffer attr_buffer;
    mlrt::bc::Allocator attr_allocator(&attr_buffer);
    mlrt::bc::New<mlrt::bc::Vector<int64_t>>(&attr_allocator,
                                             std::vector<int64_t>{0});
    attributes.Add("i64_array",
                   absl::string_view(attr_buffer.data(), attr_buffer.size()));
  }

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");
  function_ctor.set_num_regs(2);
  function_ctor.construct_input_regs(1).Assign({0});
  function_ctor.construct_output_regs(1).Assign({1});

  auto kernels_ctor = function_ctor.construct_kernels(3);

  auto set_resource_ctor = kernels_ctor.ConstructAt(0);
  set_resource_ctor.set_code(0);
  set_resource_ctor.construct_arguments(1).Assign({0});
  set_resource_ctor.construct_results(0);
  set_resource_ctor.construct_attributes(1).Assign(
      {attributes.GetHandle("index")});
  set_resource_ctor.construct_last_uses(1).Assign({0});

  auto get_resource_ctor = kernels_ctor.ConstructAt(1);
  get_resource_ctor.set_code(1);
  get_resource_ctor.construct_arguments(0);
  get_resource_ctor.construct_results(1).Assign({1});
  get_resource_ctor.construct_attributes(1).Assign(
      {attributes.GetHandle("i64_array")});

  auto return_ctor = kernels_ctor.ConstructAt(2);
  return_ctor.set_code(2);
  return_ctor.construct_arguments(1).Assign({1});
  return_ctor.construct_results(0);

  return buffer;
}

TEST(KernelTest, SetGetResource) {
  auto buffer = CreateExecutableForSetGetResourceOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext execution_context(&loaded_executable);

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  int32_t input = 100;
  tensorflow::Tensor input_tensor(input);
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value result;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);
  TF_ASSERT_OK(execution_context.status());

  int32_t output = 100;
  tensorflow::Tensor expected(output);

  tensorflow::test::ExpectEqual(
      result.Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

mlrt::bc::Buffer CreateExecutableForPredicateOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  auto kernel_names_ctor = executable_ctor.construct_kernel_names(2);
  kernel_names_ctor.ConstructAt(0, "tf_mlrt.predicate");
  kernel_names_ctor.ConstructAt(1, "return");

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");
  function_ctor.set_num_regs(2);
  function_ctor.construct_input_regs(1).Assign({0});
  function_ctor.construct_output_regs(1).Assign({1});

  auto kernels_ctor = function_ctor.construct_kernels(2);

  auto predicate_ctor = kernels_ctor.ConstructAt(0);
  predicate_ctor.set_code(0);
  predicate_ctor.construct_arguments(1).Assign({0});
  predicate_ctor.construct_results(1).Assign({1});

  auto return_ctor = kernels_ctor.ConstructAt(1);
  return_ctor.set_code(1);
  return_ctor.construct_arguments(1).Assign({1});

  return buffer;
}

TEST(KernelTest, Predicate) {
  auto buffer = CreateExecutableForPredicateOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext execution_context(&loaded_executable);

  std::vector<tensorflow::Tensor> inputs = {
      tensorflow::Tensor(true),
      tensorflow::Tensor(static_cast<int8_t>(100)),
      tensorflow::Tensor(static_cast<uint8_t>(100)),
      tensorflow::Tensor(static_cast<int16_t>(100)),
      tensorflow::Tensor(static_cast<int32_t>(100)),
      tensorflow::Tensor(static_cast<int64_t>(100)),
      tensorflow::Tensor(100.0f),
      tensorflow::Tensor(100.0),
      tensorflow::Tensor("100"),
      tensorflow::Tensor(tensorflow::DT_INT32, {4}),
  };

  for (const auto& input : inputs) {
    mlrt::Value arg((tfrt_stub::FallbackTensor(input)));
    mlrt::Value result;

    std::vector<uint8_t> last_uses = {true};
    execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                           absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
    mlrt::Execute(execution_context);
    TF_ASSERT_OK(execution_context.status());

    ASSERT_TRUE(result.HasValue());
    EXPECT_TRUE(result.Get<bool>());
  }

  tensorflow::Tensor error_input(tensorflow::DT_VARIANT, {});
  mlrt::Value arg((tfrt_stub::FallbackTensor(error_input)));
  mlrt::Value result;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);
  EXPECT_THAT(
      execution_context.status(),
      ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               "variant cannot be converted to a boolean"));
}

mlrt::bc::Buffer CreateExecutableForPromiseAwaitOps() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(5));

  attributes.Add("func_idx", 1);

  attributes.Add("copy_arg_size", 1);

  attributes.Add("num_futures", 1);

  attributes.Add("node_def_str",
                 R"pb(name: "AddV2"
                      op: "AddV2"
                      input: "dummy_arg"
                      input: "dummy_arg"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "T"
                        value { type: DT_INT32 }
                      })pb");

  attributes.Add("op_key", 0);

  auto kernel_names_ctor = executable_ctor.construct_kernel_names(8);
  kernel_names_ctor.ConstructAt(0, "mlrt.async");
  kernel_names_ctor.ConstructAt(1, "mlrt.await_handle");
  kernel_names_ctor.ConstructAt(2, "tf_mlrt.allocate_futures");
  kernel_names_ctor.ConstructAt(3, "tf_mlrt.await");
  kernel_names_ctor.ConstructAt(4, "tf_mlrt.promise");
  kernel_names_ctor.ConstructAt(5, "return");
  kernel_names_ctor.ConstructAt(6, "tf_mlrt.createop");
  kernel_names_ctor.ConstructAt(7, "tf_mlrt.executeop");

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");
    function_ctor.set_num_regs(7);
    function_ctor.construct_input_regs(1).Assign({0});
    function_ctor.construct_output_regs(1).Assign({4});

    auto kernels_ctor = function_ctor.construct_kernels(8);
    {
      // tf_mlrt.allocate_futures
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(2);
      kernel_ctor.construct_results(2).Assign({1, 2});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("num_futures")});
    }

    {
      // tf_mlrt.createop
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(6);
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("node_def_str"),
           attributes.GetHandle("op_key")});
    }

    {
      // mlrt.async
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(0);
      kernel_ctor.construct_arguments(2).Assign({0, 1});
      kernel_ctor.construct_last_uses(2).Assign({false, true});
      kernel_ctor.construct_results(1).Assign({3});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("func_idx"),
           attributes.GetHandle("copy_arg_size")});
    }

    {
      // tf_mlrt.executeop "tf.AddV2"
      auto kernel_ctor = kernels_ctor.ConstructAt(3);
      kernel_ctor.set_code(7);
      kernel_ctor.construct_arguments(2).Assign({0, 0});
      kernel_ctor.construct_results(1).Assign({5});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("node_def_str"),
           attributes.GetHandle("op_key")});
      kernel_ctor.construct_last_uses(2).Assign({0, 0});
    }

    {
      // tf_mlrt.await
      auto kernel_ctor = kernels_ctor.ConstructAt(4);
      kernel_ctor.set_code(3);
      kernel_ctor.construct_arguments(1).Assign({2});
      kernel_ctor.construct_last_uses(1).Assign({true});
      kernel_ctor.construct_results(1).Assign({4});
    }

    {
      // tf_mlrt.executeop "tf.AddV2"
      auto kernel_ctor = kernels_ctor.ConstructAt(5);
      kernel_ctor.set_code(7);
      kernel_ctor.construct_arguments(2).Assign({4, 5});
      kernel_ctor.construct_results(1).Assign({6});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("node_def_str"),
           attributes.GetHandle("op_key")});
      kernel_ctor.construct_last_uses(2).Assign({0, 0});
    }

    {
      // mlrt.await_handle
      auto kernel_ctor = kernels_ctor.ConstructAt(6);
      kernel_ctor.set_code(1);
      kernel_ctor.construct_arguments(1).Assign({3});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(7);
      kernel_ctor.set_code(5);
      kernel_ctor.construct_arguments(1).Assign({6});
    }
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("callee");
    function_ctor.set_num_regs(3);
    function_ctor.construct_input_regs(2).Assign({0, 1});

    auto kernels_ctor = function_ctor.construct_kernels(3);

    {
      // tf_mlrt.executeop "tf.AddV2"
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(7);
      kernel_ctor.construct_arguments(2).Assign({0, 0});
      kernel_ctor.construct_results(1).Assign({2});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("node_def_str"),
           attributes.GetHandle("op_key")});
      kernel_ctor.construct_last_uses(2).Assign({0, 0});
    }

    {
      // tf_mlrt.promise
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(4);
      kernel_ctor.construct_arguments(2).Assign({1, 2});
      kernel_ctor.construct_last_uses(2).Assign({true, true});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(5);
    }
  }

  return buffer;
}

TEST(KernelTest, PromiseAwait) {
  auto buffer = CreateExecutableForPromiseAwaitOps();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  tensorflow::Tensor input(100);
  mlrt::Value arg((tfrt_stub::FallbackTensor(input)));
  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  tensorflow::test::ExpectEqual(
      result.Get<tfrt_stub::FallbackTensor>().tensor(),
      tensorflow::Tensor(400));
}

mlrt::bc::Buffer CreateExecutableForPromiseFutureOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;

  std::vector<std::string> kernel_names = {
      "tf_mlrt.createop",        "tf_mlrt.allocate_futures",
      "tf_mlrt.async_executeop", "tf_mlrt.promise_future",
      "tf_mlrt.await_all",       "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(3));

  attributes.Add("node_def_str",
                 R"pb(name: "TestAsyncIdentity"
                      op: "TestAsyncIdentity"
                      input: "dummy_arg"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "T"
                        value { type: DT_INT32 }
                      })pb");

  attributes.Add("op_key", 0);
  attributes.Add("num_futures", 1);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  mlrt::testing::SymbolTable regs;

  function_ctor.construct_input_regs(1).Assign({regs.Def("input")});

  auto kernels_ctor = function_ctor.construct_kernels(6);

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.createop"));
    kernel_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
  }

  {
    // tf_mlrt.allocate_futures
    auto kernel_ctor = kernels_ctor.ConstructAt(1);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.allocate_futures"));
    kernel_ctor.construct_results(2).Assign(
        regs.Def(absl::Span<const std::string>{"promise", "future"}));
    kernel_ctor.construct_attributes(1).Assign(
        {attributes.GetHandle("num_futures")});
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(2);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.async_executeop"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"input", "input"}));
    kernel_ctor.construct_results(1).Assign({regs.Def("result_future_0")});
    kernel_ctor.construct_attributes(2).Assign(
        {attributes.GetHandle("node_def_str"), attributes.GetHandle("op_key")});
    kernel_ctor.construct_last_uses(2).Assign({0, 0});
  }
  {
    // tf_mlrt.promise_future
    auto kernel_ctor = kernels_ctor.ConstructAt(3);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.promise_future"));
    kernel_ctor.construct_arguments(2).Assign(
        regs.Use({"promise", "result_future_0"}));
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(4);
    kernel_ctor.set_code(kernels.Use("tf_mlrt.await_all"));
    kernel_ctor.construct_arguments(2).Assign(
        regs.Use({"future", "result_future_0"}));
    kernel_ctor.construct_last_uses(2).Assign({true, true});
    kernel_ctor.construct_results(2).Assign(
        {regs.Def("result_0"), regs.Def("result_1")});
  }

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(5);
    kernel_ctor.set_code(kernels.Use("return"));
    kernel_ctor.construct_arguments(2).Assign(
        regs.Use({"result_0", "result_1"}));
  }

  function_ctor.set_num_regs(regs.size());
  function_ctor.construct_output_regs(2).Assign(
      regs.Use({"result_0", "result_1"}));

  return buffer;
}

TEST(KernelTest, PromiseFutureOp) {
  auto buffer = CreateExecutableForPromiseFutureOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  int32_t input = 100;
  tensorflow::Tensor input_tensor(input);
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value results[2];

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });
  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(results));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  int32_t output = 100;
  tensorflow::Tensor expected(output);
  tensorflow::test::ExpectEqual(
      results[0].Get<tfrt_stub::FallbackTensor>().tensor(), expected);
  tensorflow::test::ExpectEqual(
      results[1].Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

mlrt::bc::Buffer CreateExecutableForBatchFunctionOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;

  std::vector<std::string> kernel_names = {
      "tf_mlrt.createop", "tf_mlrt.executeop", "tf_mlrt.batch_function",
      "tf_mlrt.await", "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(5));

  attributes.Add("node_def_str",
                 R"pb(name: "AddV2"
                      op: "AddV2"
                      input: "dummy_arg"
                      input: "dummy_arg"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "T"
                        value { type: DT_INT32 }
                      })pb");

  attributes.Add("op_key", 0);

  attributes.Add("func_idx", 1);

  // attributes[3] is NodeDef for batch function.
  attributes.Add("batch_node_def_str",
                 R"pb(name: "BatchFunction"
                      op: "MlrtBatchFunction"
                      input: "dummy_arg"
                      input: "dummy_arg"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "num_batch_threads"
                        value: { i: 16 }
                      }
                      attr {
                        key: "max_batch_size"
                        value { i: 1 }
                      }
                      attr {
                        key: "allowed_batch_sizes"
                        value { list { i: 1 } }
                      }
                      attr {
                        key: "batch_timeout_micros"
                        value { i: 0 }
                      }
                      attr {
                        key: "low_priority_max_batch_size"
                        value { i: 1 }
                      }
                      attr {
                        key: "low_priority_batch_timeout_micros"
                        value { i: 0 }
                      }
                      attr {
                        key: "low_priority_allowed_batch_sizes"
                        value { list { i: 1 } }
                      }
                      attr {
                        key: "low_priority_max_enqueued_batches"
                        value { i: 1 }
                      }
                      attr {
                        key: "mixed_priority_policy"
                        value { s: "low_priority_padding_with_max_batch_size" }
                      }
                      attr {
                        key: "container"
                        value { s: "container" }
                      }
                      attr {
                        key: "shared_name"
                        value { s: "shared_name" }
                      }
                      attr {
                        key: "batching_queue"
                        value { s: "batching_queue" }
                      }
                      attr {
                        key: "enable_large_batch_splitting"
                        value { b: false }
                      }
                      attr {
                        key: "Tin"
                        value { list { type: DT_INT32 type: DT_INT32 } }
                      }
                      attr {
                        key: "Tcaptured"
                        value { list {} }
                      }
                      attr {
                        key: "Tout"
                        value { list { type: DT_INT32 } }
                      })pb");

  attributes.Add("device", "/device:CPU:0");

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(1).Assign({regs.Def("input")});

    auto kernels_ctor = function_ctor.construct_kernels(4);

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.createop"));
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("node_def_str"),
           attributes.GetHandle("op_key")});
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.batch_function"));
      kernel_ctor.construct_arguments(2).Assign(regs.Use({"input", "input"}));
      kernel_ctor.construct_results(1).Assign({regs.Def("result_future")});
      kernel_ctor.construct_attributes(3).Assign(
          {attributes.GetHandle("device"), attributes.GetHandle("func_idx"),
           attributes.GetHandle("batch_node_def_str")});
      kernel_ctor.construct_last_uses(2).Assign({0, 0});
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.await"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result_future")});
      kernel_ctor.construct_last_uses(1).Assign({true});
      kernel_ctor.construct_results(1).Assign({regs.Def("result")});
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(3);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
    }

    function_ctor.set_num_regs(regs.size());
    function_ctor.construct_output_regs(1).Assign({regs.Use("result")});
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("batch_function");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"input_0", "input_1"}));

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.executeop"));
      kernel_ctor.construct_arguments(2).Assign(
          regs.Use({"input_0", "input_1"}));
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("node_def_str"),
           attributes.GetHandle("op_key")});
      kernel_ctor.construct_results(1).Assign({regs.Def("result")});
      kernel_ctor.construct_last_uses(2).Assign({0, 0});
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
    }

    function_ctor.set_num_regs(regs.size());
    function_ctor.construct_output_regs(1).Assign({regs.Use("result")});
  }

  return buffer;
}

TEST(KernelTest, BatchFunctionOp) {
  auto buffer = CreateExecutableForBatchFunctionOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  RegisterTfMlrtBatchKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  tensorflow::Tensor input_tensor(tensorflow::DT_INT32, {1});
  input_tensor.flat<int32_t>()(0) = 100;
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  tensorflow::Tensor expected(tensorflow::DT_INT32, {1});
  expected.flat<int32_t>()(0) = 200;

  tensorflow::test::ExpectEqual(
      result.Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

mlrt::bc::Buffer CreateExecutableForCancelOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  auto kernel_names_ctor = executable_ctor.construct_kernel_names(3);
  kernel_names_ctor.ConstructAt(0, "tf_mlrt.cancel");
  kernel_names_ctor.ConstructAt(1, "tf_mlrt.predicate");
  kernel_names_ctor.ConstructAt(2, "return");

  auto functions_ctor = executable_ctor.construct_functions(1);
  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");
    function_ctor.set_num_regs(2);
    function_ctor.construct_input_regs(1).Assign({0});
    function_ctor.construct_output_regs(1).Assign({1});

    auto kernels_ctor = function_ctor.construct_kernels(3);

    {
      // cancel
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(0);
    }
    {
      // predicate
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(1);
      kernel_ctor.construct_arguments(1).Assign({0});
      kernel_ctor.construct_results(1).Assign({1});
    }
    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(2);
      kernel_ctor.construct_arguments(1).Assign({1});
    }
  }

  return buffer;
}

TEST(KernelTest, CancelCanEarlyReturn) {
  auto buffer = CreateExecutableForCancelOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::RCReference<tfrt::CancellationContext> cancellation_context =
      tfrt::TakeRef(new tfrt::CancellationContext());

  tfrt::ResourceContext resource_context;

  auto tf_context = std::make_unique<Context>(
      &fallback_request_state, &resource_context, cancellation_context.get());
  execution_context.AddUserContext(std::move(tf_context));

  tensorflow::Tensor input_tensor =
      tensorflow::Tensor(static_cast<int8_t>(100));
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));

  // Cancelling
  cancellation_context->Cancel();

  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_THAT(execution_context.status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(result.HasValue(), false);
}

// Have a cancel OP has no side effect when there is no cancel request.
TEST(KernelTest, NoCancel) {
  auto buffer = CreateExecutableForCancelOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  tensorflow::Tensor input_tensor =
      tensorflow::Tensor(static_cast<int8_t>(100));
  mlrt::Value arg(tfrt_stub::FallbackTensor(std::move(input_tensor)));
  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));

  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  ASSERT_TRUE(result.HasValue());
  EXPECT_TRUE(result.Get<bool>());
}

mlrt::bc::Buffer CreateAsyncExecutableForCancelOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(2));
  attributes.Add("async_callee", 1);
  attributes.Add("copy_arg_size", 0);

  mlrt::testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.async", "mlrt.await_handle",
                                    "tf_mlrt.cancel", "tf_mlrt.promise",
                                    "return"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);
  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;
    function_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"in", "promise"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(2).Assign(regs.Use({"in", "promise"}));
      kernel_ctor.construct_last_uses(2).Assign({true, true});
      kernel_ctor.construct_results(1).Assign({regs.Def("handle")});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("async_callee"),
           attributes.GetHandle("copy_arg_size")});
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.await_handle"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("handle")});
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }
    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("async_callee");

    mlrt::testing::SymbolTable regs;
    function_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"in", "promise"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.cancel"));
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.promise"));
      kernel_ctor.construct_arguments(2).Assign(regs.Use({"promise", "in"}));
      kernel_ctor.construct_last_uses(2).Assign({true, true});
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }
    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(KernelTest, CancelInAsyncCanEarlyReturn) {
  auto buffer = CreateAsyncExecutableForCancelOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::RCReference<tfrt::CancellationContext> cancellation_context =
      tfrt::TakeRef(new tfrt::CancellationContext());

  tfrt::ResourceContext resource_context;

  auto tf_context = std::make_unique<Context>(
      &fallback_request_state, &resource_context, cancellation_context.get());
  execution_context.AddUserContext(std::move(tf_context));

  tensorflow::Tensor input_tensor =
      tensorflow::Tensor(static_cast<int8_t>(100));

  auto promise =
      mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>();
  auto future = promise.GetFuture();

  std::vector<mlrt::Value> input_args;
  input_args.push_back(
      mlrt::Value(tfrt_stub::FallbackTensor(std::move(input_tensor))));
  input_args.push_back(mlrt::Value(std::move(promise)));

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(input_args), absl::Span<mlrt::Value>());

  // Cancelling
  cancellation_context->Cancel();

  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_THAT(execution_context.status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kCancelled));

  ASSERT_TRUE(future.IsError());

  EXPECT_THAT(future.GetError(),
              ::tsl::testing::StatusIs(absl::StatusCode::kCancelled));
}

mlrt::bc::Buffer CreateExecutableForTensorToInt32Op() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;
  std::vector<std::string> names = {"tf_mlrt.tensor_to_int32", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  mlrt::testing::SymbolTable regs;

  function_ctor.construct_input_regs(1).Assign({regs.Def("tensor_int")});

  auto kernels_ctor = function_ctor.construct_kernels(2);

  auto predicate_ctor = kernels_ctor.ConstructAt(0);
  predicate_ctor.set_code(kernels.Use("tf_mlrt.tensor_to_int32"));
  predicate_ctor.construct_arguments(1).Assign({regs.Use("tensor_int")});
  predicate_ctor.construct_results(1).Assign({regs.Def("scalar_int")});

  auto return_ctor = kernels_ctor.ConstructAt(1);
  return_ctor.set_code(kernels.Use("return"));
  return_ctor.construct_arguments(1).Assign({regs.Use("scalar_int")});

  function_ctor.construct_output_regs(1).Assign({regs.Use("scalar_int")});
  function_ctor.set_num_regs(regs.size());

  return buffer;
}

TEST(KernelTest, TensorToInt32) {
  auto buffer = CreateExecutableForTensorToInt32Op();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  {
    tensorflow::Tensor input = tensorflow::Tensor(static_cast<int32_t>(100));

    mlrt::Value arg((tfrt_stub::FallbackTensor(input)));
    mlrt::Value result;

    std::vector<uint8_t> last_uses = {true};
    execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                           absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
    mlrt::Execute(execution_context);
    TF_ASSERT_OK(execution_context.status());

    ASSERT_TRUE(result.HasValue());
    EXPECT_EQ(result.Get<int32_t>(), 100);
  }
  {
    tensorflow::Tensor error_input(tensorflow::DT_VARIANT, {});
    mlrt::Value arg((tfrt_stub::FallbackTensor(error_input)));
    mlrt::Value result;

    std::vector<uint8_t> last_uses = {true};
    execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                           absl::MakeSpan(&arg, 1), absl::MakeSpan(&result, 1));
    mlrt::Execute(execution_context);
    EXPECT_THAT(
        execution_context.status(),
        ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                 "variant cannot be converted to a int32"));
  }
}

// A function body for MapFnOp.
constexpr int32_t kMapFnOpMaxIteration = 2;
constexpr int32_t kMapFnBias = 100;
void TestMapFnBody(mlrt::KernelFrame frame) {
  ASSERT_EQ(frame.arguments().size(), 5);

  auto future = frame.arguments()[0].Get<mlrt::Future>();
  auto promise = std::move(frame.arguments()[1].Get<mlrt::Promise>());

  int32_t loop_index = frame.arguments()[2]
                           .Get<tensorflow::tfrt_stub::FallbackTensor>()
                           .tensor()
                           .scalar<int32_t>()();

  int32_t element_index = frame.arguments()[3]
                              .Get<tensorflow::tfrt_stub::FallbackTensor>()
                              .tensor()
                              .scalar<int32_t>()();

  int32_t offset = frame.arguments()[4]
                       .Get<tensorflow::tfrt_stub::FallbackTensor>()
                       .tensor()
                       .scalar<int32_t>()();

  for (; !future.IsReady();) {
    // wait for future to be ready
  }
  auto in_tensor = future.Get<tensorflow::tfrt_stub::FallbackTensor>().tensor();
  tensorflow::Tensor out_tensor(DT_INT32, {kMapFnOpMaxIteration});
  for (int i = 0; i < element_index; i++) {
    out_tensor.flat<int32_t>()(i) = in_tensor.flat<int32_t>()(i);
  }

  out_tensor.flat<int32_t>()(element_index) = loop_index + offset;

  std::move(promise).Set<tensorflow::tfrt_stub::FallbackTensor>(out_tensor);
}
void TestMapFnBodyError(mlrt::KernelFrame frame) {
  ASSERT_EQ(frame.arguments().size(), 5);

  auto future = frame.arguments()[0].Get<mlrt::Future>();
  auto promise = std::move(frame.arguments()[1].Get<mlrt::Promise>());
  std::move(future).Then([promise = std::move(promise)]() mutable {
    std::move(promise).SetError(absl::InternalError("Test Error"));
  });
}

// An MapFn body that errors out but with its return promise is already set.
// Note that currently, while_to_map_fn pass ensure promise is only set right
// before return.
void TestMapFnBodyErrorWithPromiseSet(mlrt::KernelFrame frame) {
  ASSERT_EQ(frame.arguments().size(), 5);

  auto future = frame.arguments()[0].Get<mlrt::Future>();
  auto promise = std::move(frame.arguments()[1].Get<mlrt::Promise>());

  auto loop_index =
      frame.arguments()[2].Get<tensorflow::tfrt_stub::FallbackTensor>();
  std::move(promise).Set<tensorflow::tfrt_stub::FallbackTensor>(
      std::move(loop_index));

  auto control_promise = mlrt::Promise::Allocate<mlrt::Control>();
  auto control_future = control_promise.GetFuture();
  std::move(future).Then(
      [control_promise = std::move(control_promise)]() mutable {
        std::move(control_promise).SetError(absl::InternalError("Test Error"));
      });

  frame.execution_context().Await(std::move(control_future));
}

mlrt::bc::Buffer CreateExecutableForMapFnOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;

  std::vector<std::string> kernel_names = {"tf_mlrt.map_fn", "return",
                                           "test_map_fn_body"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(1));
  attributes.Add("body_idx", 1);
  attributes.Add("num_tensor_list", 1);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(3).Assign(
        regs.Def({"max_iteration", "tensor_list", "bias"}));

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.map_fn"));
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("body_idx"),
           attributes.GetHandle("num_tensor_list")});
      kernel_ctor.construct_arguments(3).Assign(
          regs.Use({"max_iteration", "tensor_list", "bias"}));
      kernel_ctor.construct_results(1).Assign({regs.Def("result0")});
      kernel_ctor.construct_last_uses(3).Assign({0, 1, 0});
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result0")});
    }

    function_ctor.set_num_regs(regs.size());
    function_ctor.construct_output_regs(1).Assign({regs.Use("result0")});
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("body_function");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(5).Assign(regs.Def(
        {"future", "promise", "loop_counter", "element_index", "bias"}));

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("test_map_fn_body"));
      kernel_ctor.construct_arguments(5).Assign(regs.Use(
          {"future", "promise", "loop_counter", "element_index", "bias"}));
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(KernelTest, MapFnOp) {
  auto buffer = CreateExecutableForMapFnOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  registry.Register("test_map_fn_body", TestMapFnBody);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::vector<mlrt::Value> args;
  args.resize(3);

  tensorflow::Tensor loop_max_iteration_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {},
                                              &loop_max_iteration_tensor));
  loop_max_iteration_tensor.scalar<int32_t>()() = kMapFnOpMaxIteration;
  args.at(0).Set(
      tfrt_stub::FallbackTensor(std::move(loop_max_iteration_tensor)));

  tensorflow::Tensor tensor_list;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {kMapFnOpMaxIteration},
                                              &tensor_list));
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(tensor_list)));

  tensorflow::Tensor bias_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &bias_tensor));
  bias_tensor.scalar<int32_t>()() = kMapFnBias;
  args.at(2).Set(tfrt_stub::FallbackTensor(bias_tensor));

  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  tensorflow::Tensor expected(tensorflow::DT_INT32, {kMapFnOpMaxIteration});
  expected.flat<int32_t>()(0) = 100;
  expected.flat<int32_t>()(1) = 101;

  auto& to_be = result.Get<tensorflow::tfrt_stub::FallbackTensor>();
  tensorflow::test::ExpectEqual(to_be.tensor(), expected);
}

TEST(KernelTest, MapFnOpZeroIteration) {
  auto buffer = CreateExecutableForMapFnOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  registry.Register("test_map_fn_body", TestMapFnBody);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::vector<mlrt::Value> args;
  args.resize(3);

  tensorflow::Tensor loop_max_iteration_tensor(DT_INT32, {});
  loop_max_iteration_tensor.scalar<int32_t>()() = 0;
  args.at(0).Set(
      tfrt_stub::FallbackTensor(std::move(loop_max_iteration_tensor)));

  tensorflow::Tensor tensor_list(DT_INT32, {});
  tensor_list.scalar<int32_t>()() = 1000;
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(tensor_list)));

  tensorflow::Tensor bias_tensor(DT_INT32, {});
  bias_tensor.scalar<int32_t>()() = kMapFnBias;
  args.at(2).Set(tfrt_stub::FallbackTensor(bias_tensor));

  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  tensorflow::Tensor expected(tensorflow::DT_INT32, {});
  expected.scalar<int32_t>()() = 1000;

  auto& to_be = result.Get<tensorflow::tfrt_stub::FallbackTensor>();
  tensorflow::test::ExpectEqual(to_be.tensor(), expected);
}

TEST(KernelTest, MapFnOpError) {
  auto buffer = CreateExecutableForMapFnOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  registry.Register("test_map_fn_body", TestMapFnBodyError);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::vector<mlrt::Value> args;
  args.resize(3);

  tensorflow::Tensor loop_max_iteration_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {},
                                              &loop_max_iteration_tensor));
  loop_max_iteration_tensor.scalar<int32_t>()() = kMapFnOpMaxIteration;
  args.at(0).Set(
      tfrt_stub::FallbackTensor(std::move(loop_max_iteration_tensor)));

  tensorflow::Tensor tensor_list;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {kMapFnOpMaxIteration},
                                              &tensor_list));
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(tensor_list)));

  tensorflow::Tensor bias_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &bias_tensor));
  bias_tensor.scalar<int32_t>()() = kMapFnBias;
  args.at(2).Set(tfrt_stub::FallbackTensor(bias_tensor));

  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_THAT(execution_context.status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kInternal,
                                       "Test Error. First Error Index=0 of 1"));
}

TEST(KernelTest, MapFnOpErrorWithPromiseSet) {
  auto buffer = CreateExecutableForMapFnOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  registry.Register("test_map_fn_body", TestMapFnBodyErrorWithPromiseSet);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::vector<mlrt::Value> args;
  args.resize(3);

  tensorflow::Tensor loop_max_iteration_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {},
                                              &loop_max_iteration_tensor));
  loop_max_iteration_tensor.scalar<int32_t>()() = kMapFnOpMaxIteration;
  args.at(0).Set(
      tfrt_stub::FallbackTensor(std::move(loop_max_iteration_tensor)));

  tensorflow::Tensor tensor_list;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {kMapFnOpMaxIteration},
                                              &tensor_list));
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(tensor_list)));

  tensorflow::Tensor bias_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &bias_tensor));
  bias_tensor.scalar<int32_t>()() = kMapFnBias;
  args.at(2).Set(tfrt_stub::FallbackTensor(bias_tensor));

  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_THAT(
      execution_context.status(),
      ::tsl::testing::StatusIs(absl::StatusCode::kInternal, "Test Error"));
}

mlrt::bc::Buffer CreatePromiseReturnExecutable() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;
  std::vector<std::string> names = {"tf_mlrt.await", "tf_mlrt.promise_return",
                                    "return"};
  executable_ctor.construct_kernel_names(3).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);
  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("consumer");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(1).Assign({regs.Def("future")});

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      // Await
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.await"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("future")});
      kernel_ctor.construct_last_uses(1).Assign({true});
      kernel_ctor.construct_results(1).Assign({regs.Def("result")});
    }

    {
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
    }

    function_ctor.set_num_regs(regs.size());
    function_ctor.construct_output_regs(1).Assign({regs.Use("result")});
    function_ctor.construct_output_last_uses(1).Assign({true});
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("producer");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(2).Assign(
        {regs.Def("promise"), regs.Def("value")});

    auto kernels_ctor = function_ctor.construct_kernels(1);

    {
      // promise_return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.promise_return"));
      kernel_ctor.construct_arguments(2).Assign(
          {regs.Use("promise"), regs.Use("value")});
      kernel_ctor.construct_last_uses(2).Assign({true, true});
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(KernelTest, PromiseReturn) {
  auto buffer = CreatePromiseReturnExecutable();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  mlrt::ExecutionContext consumer_context(&loaded_executable);

  absl::Notification notification;
  consumer_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise =
      mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>();
  tensorflow::Tensor expected(static_cast<int32_t>(100));

  mlrt::Value output;
  {
    mlrt::Value input(promise.GetFuture());

    std::vector<uint8_t> last_uses = {true};
    consumer_context.Call(loaded_executable.GetFunction("consumer"), last_uses,
                          absl::Span<mlrt::Value>(&input, 1),
                          absl::Span<mlrt::Value>(&output, 1));
    mlrt::Execute(consumer_context);
  }

  {
    mlrt::Value inputs[2];
    inputs[0].Set(std::move(promise));
    inputs[1].Set(tensorflow::tfrt_stub::FallbackTensor(expected));

    mlrt::ExecutionContext producer_context(&loaded_executable);
    std::vector<uint8_t> last_uses = {true, true};
    producer_context.Call(loaded_executable.GetFunction("producer"), last_uses,
                          absl::Span<mlrt::Value>(inputs),
                          absl::Span<mlrt::Value>());
    mlrt::Execute(producer_context);
  }

  ASSERT_TRUE(notification.HasBeenNotified());
  tensorflow::test::ExpectEqual(
      output.Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

// A function body for AsyncWhile.
void TestAsyncWhileFnBody(mlrt::KernelFrame frame) {
  ASSERT_EQ(frame.arguments().size(), 4);

  auto predicate_promise = std::move(frame.arguments()[0].Get<mlrt::Promise>());
  auto prev_loop_count_future = frame.arguments()[1].Get<mlrt::Future>();
  auto next_loop_count_promise =
      std::move(frame.arguments()[2].Get<mlrt::Promise>());

  int32_t max_iteration = frame.arguments()[3]
                              .Get<tensorflow::tfrt_stub::FallbackTensor>()
                              .tensor()
                              .scalar<int32_t>()();

  for (; !prev_loop_count_future.IsReady();) {
    // wait for future to be ready
  }
  int32_t prev_loop_count =
      prev_loop_count_future.Get<tensorflow::tfrt_stub::FallbackTensor>()
          .tensor()
          .scalar<int32_t>()();
  tensorflow::Tensor next_loop_count(DT_INT32, {});
  next_loop_count.scalar<int32_t>()() = prev_loop_count + 1;

  tensorflow::Tensor predicate(DT_BOOL, {});
  predicate.scalar<bool>()() = prev_loop_count + 1 < max_iteration;
  std::move(predicate_promise)
      .Set<tensorflow::tfrt_stub::FallbackTensor>(std::move(predicate));

  std::move(next_loop_count_promise)
      .Set<tensorflow::tfrt_stub::FallbackTensor>(std::move(next_loop_count));
}

mlrt::bc::Buffer CreateAsyncWhileExecutable() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);
  mlrt::testing::SymbolTable kernels;
  std::vector<std::string> kernel_names = {"tf_mlrt.async_while",
                                           "tf_mlrt.await_all",
                                           "test_async_while_body", "return"};
  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);
  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(1));

  attributes.Add("body_idx", 1);
  attributes.Add("invariant_size", 1);
  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");
    mlrt::testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(
        regs.Def({"initial_predicate", "loop_count", "max_iterations"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.async_while"));
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("body_idx"),
           attributes.GetHandle("invariant_size")});
      kernel_ctor.construct_arguments(3).Assign(
          regs.Use({"initial_predicate", "loop_count", "max_iterations"}));
      kernel_ctor.construct_results(3).Assign(
          regs.Def({"last_predicate_future", "final_loop_count_future",
                    "final_max_iterations_future"}));
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.await_all"));
      kernel_ctor.construct_arguments(3).Assign(
          regs.Use({"last_predicate_future", "final_loop_count_future",
                    "final_max_iterations_future"}));
      kernel_ctor.construct_last_uses(3).Assign({true, true, true});
      kernel_ctor.construct_results(3).Assign(regs.Def(
          {"last_predicate", "final_loop_count", "final_max_iterations"}));
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("final_loop_count")});
    }
    function_ctor.set_num_regs(regs.size());
    function_ctor.construct_output_regs(1).Assign(
        {regs.Use("final_loop_count")});
  }
  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("body_function");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(4).Assign(
        regs.Def({"predicate_promise", "prev_loop_count_future",
                  "loop_count_promise", "max_iterations"}));
    auto kernels_ctor = function_ctor.construct_kernels(2);
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("test_async_while_body"));
      kernel_ctor.construct_arguments(4).Assign(
          regs.Use({"predicate_promise", "prev_loop_count_future",
                    "loop_count_promise", "max_iterations"}));
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }
    function_ctor.set_num_regs(regs.size());
  }
  return buffer;
}

struct AsyncWhileOpTestParams {
  bool initial_predicate;
  int final_result;
};
class AsyncWhileOpTestFixture
    : public ::testing::TestWithParam<AsyncWhileOpTestParams> {};
TEST_P(AsyncWhileOpTestFixture, AsyncWhileOp) {
  auto params = GetParam();
  auto buffer = CreateAsyncWhileExecutable();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  registry.Register("test_async_while_body", TestAsyncWhileFnBody);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::vector<mlrt::Value> args;
  args.resize(3);

  // initial predicate is true
  tensorflow::Tensor initial_predicate_tensor{DT_BOOL, {}};
  initial_predicate_tensor.scalar<bool>()() = params.initial_predicate;
  args.at(0).Set(
      tfrt_stub::FallbackTensor(std::move(initial_predicate_tensor)));

  tensorflow::Tensor loop_count_tensor{DT_INT32, {}};
  loop_count_tensor.scalar<int32_t>()() = 0;
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(loop_count_tensor)));

  tensorflow::Tensor max_iteration_tensor{DT_INT32, {}};
  max_iteration_tensor.scalar<int32_t>()() = 2;
  args.at(2).Set(tfrt_stub::FallbackTensor(std::move(max_iteration_tensor)));

  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  ASSERT_OK(execution_context.status());

  tensorflow::Tensor expected(tensorflow::DT_INT32, {});
  expected.scalar<int32_t>()() = params.final_result;

  auto& to_be = result.Get<tensorflow::tfrt_stub::FallbackTensor>();
  tensorflow::test::ExpectEqual(to_be.tensor(), expected);
}

INSTANTIATE_TEST_SUITE_P(
    AsyncWhileOpTestSuite, AsyncWhileOpTestFixture,
    ::testing::ValuesIn<AsyncWhileOpTestParams>({{true, 2}, {false, 0}}));

// A AsyncWhile body function that triggers failure.
void TestAsyncWhileFnBodyError(mlrt::KernelFrame frame) {
  ASSERT_EQ(frame.arguments().size(), 4);

  frame.execution_context().Fail(absl::InternalError("Test error"));
}
TEST(KernelTest, AsyncWhileOpError) {
  auto buffer = CreateAsyncWhileExecutable();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  RegisterTfMlrtKernels(registry);
  registry.Register("test_async_while_body", TestAsyncWhileFnBodyError);

  mlrt::LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state, tfrt_stub::FallbackState::Create(
                                                   session_options, fdef_lib));

  std::function<void(std::function<void()>)> runner =
      [](const std::function<void()>& f) { f(); };
  tfrt_stub::OpKernelRunnerTable runner_table;
  tfd::FallbackResourceArray resource_array;
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      &runner, &fallback_state->device_manager(), /*step_id=*/0, &runner_table,
      &resource_array, /*user_intra_op_threadpool=*/nullptr,
      /*model_metadata=*/std::nullopt,
      &fallback_state->process_function_library_runtime());

  tfrt::ResourceContext resource_context;

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::vector<mlrt::Value> args;
  args.resize(3);

  // initial predicate is true
  tensorflow::Tensor initial_predicate_tensor{DT_BOOL, {}};
  initial_predicate_tensor.scalar<bool>()() = true;
  args.at(0).Set(
      tfrt_stub::FallbackTensor(std::move(initial_predicate_tensor)));

  tensorflow::Tensor loop_count_tensor{DT_INT32, {}};
  loop_count_tensor.scalar<int32_t>()() = 0;
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(loop_count_tensor)));

  tensorflow::Tensor max_iteration_tensor{DT_INT32, {}};
  max_iteration_tensor.scalar<int32_t>()() = 2;
  args.at(2).Set(tfrt_stub::FallbackTensor(std::move(max_iteration_tensor)));

  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      ::tsl::testing::StatusIs(absl::StatusCode::kInternal, "Test error"));
}

}  // namespace
}  // namespace tf_mlrt
}  // namespace tensorflow
