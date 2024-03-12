/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/builtin_kernels.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/interpreter_testutil.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/platform/tstring.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {
namespace {
using tensorflow::test::AsScalar;
using tensorflow::test::ExpectEqual;

static absl::string_view kVariableName = "test_variable";

tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

mlrt::bc::Buffer CreateExecutableForIfrtLoadVariableOp(
    bool redundant_ifrt_load_variable_op = false) {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;
  std::vector<std::string> kernel_names = {
      "tf_mlrt.createop", "tf_mlrt.executeop", "tf_mlrt.ifrt_load_variable",
      "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(6));

  tensorflow::ifrt_serving::VariableDeviceShardingConfigProto sharding_config;
  sharding_config.add_device_ids(0);
  std::string serialized_sharding_config;
  tsl::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.PrintToString(sharding_config, &serialized_sharding_config);

  attributes.Add("sharding_config", serialized_sharding_config);
  attributes.Add("variable_name", kVariableName);

  attributes.Add("var_handle_op_node_def",
                 R"pb(name: "VarHandleOp"
                      op: "VarHandleOp"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "container"
                        value { s: "test" }
                      }
                      attr {
                        key: "shared_name"
                        value { s: "y" }
                      }
                      attr {
                        key: "dtype"
                        value { type: DT_INT32 }
                      }
                      attr {
                        key: "shape"
                        value { shape { dim { size: 1 } } }
                      }
                 )pb");

  attributes.Add("var_handle_op_key", 0);

  attributes.Add("assign_variable_op_node_def",
                 R"pb(name: "AssignVariableOp"
                      op: "AssignVariableOp"
                      input: "dummy_resource"
                      input: "dummy_tensor"
                      device: "/job:localhost/replica:0/task:0/device:CPU:0"
                      attr {
                        key: "dtype"
                        value { type: DT_INT32 }
                      }
                 )pb");

  attributes.Add("assign_variable_op_key", 1);

  auto functions_ctor = executable_ctor.construct_functions(1);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(1).Assign({regs.Def("input_tensor")});
    function_ctor.construct_output_regs(1).Assign({regs.Def("output_tensor")});

    const int kNumKernels = 6 + (redundant_ifrt_load_variable_op ? 1 : 0);
    auto kernels_ctor = function_ctor.construct_kernels(kNumKernels);
    int kernel_index = 0;

    {
      // Create VarHandleOp
      auto createop_ctor = kernels_ctor.ConstructAt(kernel_index);
      createop_ctor.set_code(kernels.Use("tf_mlrt.createop"));
      createop_ctor.construct_arguments(0);
      createop_ctor.construct_results(0);
      createop_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("var_handle_op_node_def"),
           attributes.GetHandle("var_handle_op_key")});
      kernel_index++;
    }
    {
      // Create AssignVariableOp
      auto createop_ctor = kernels_ctor.ConstructAt(kernel_index);
      createop_ctor.set_code(kernels.Use("tf_mlrt.createop"));
      createop_ctor.construct_arguments(0);
      createop_ctor.construct_results(0);
      createop_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("assign_variable_op_node_def"),
           attributes.GetHandle("assign_variable_op_key")});
      kernel_index++;
    }
    {
      // Execute VarHandleOp
      auto executeop_ctor = kernels_ctor.ConstructAt(kernel_index);
      executeop_ctor.set_code(kernels.Use("tf_mlrt.executeop"));
      executeop_ctor.construct_arguments(0);
      executeop_ctor.construct_results(1).Assign({regs.Def("variable_handle")});
      executeop_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("var_handle_op_node_def"),
           attributes.GetHandle("var_handle_op_key")});
      executeop_ctor.construct_last_uses(1).Assign({0});
      kernel_index++;
    }

    {
      // Execute AssignVariableOp
      auto executeop_ctor = kernels_ctor.ConstructAt(kernel_index);
      executeop_ctor.set_code(kernels.Use("tf_mlrt.executeop"));
      executeop_ctor.construct_arguments(2).Assign(
          regs.Use({"variable_handle", "input_tensor"}));
      executeop_ctor.construct_results(0);
      executeop_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("assign_variable_op_node_def"),
           attributes.GetHandle("assign_variable_op_key")});
      executeop_ctor.construct_last_uses(2).Assign({0, 0});
      kernel_index++;
    }

    {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.ifrt_load_variable"));
      kernel_ctor.construct_results(1).Assign({regs.Use("output_tensor")});
      kernel_ctor.construct_arguments(1).Assign({regs.Use("variable_handle")});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("sharding_config"),
           attributes.GetHandle("variable_name")});
      kernel_ctor.construct_last_uses(1).Assign({1});
      kernel_index++;
    }
    if (redundant_ifrt_load_variable_op) {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.ifrt_load_variable"));
      kernel_ctor.construct_results(1).Assign({regs.Def("dummy")});
      kernel_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("sharding_config"),
           attributes.GetHandle("variable_name")});
      kernel_ctor.construct_arguments(1).Assign({regs.Use("input_tensor")});
      kernel_ctor.construct_last_uses(1).Assign({1});
      kernel_index++;
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("output_tensor")});
      kernel_index++;
    }
    DCHECK_EQ(kernel_index, kNumKernels);

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(KernelTest, IfrtLoadVariableOp) {
  auto buffer = CreateExecutableForIfrtLoadVariableOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  mlrt::RegisterBuiltinKernels(registry);
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

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  resource_context.CreateResource<tensorflow::ifrt_serving::IfrtModelContext>(
      "IfrtModelContext", client, &GetThreadPool());

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::optional<tensorflow::ifrt_serving::IfrtModelContext*>
      ifrt_model_context =
          resource_context
              .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                  "IfrtModelContext");

  ASSERT_TRUE(ifrt_model_context.has_value());
  EXPECT_THAT((*ifrt_model_context)
                  ->GetLoadedVariableRegistry()
                  .GetLoadedVariable(kVariableName)
                  .status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kNotFound));

  std::vector<mlrt::Value> args;
  args.resize(1);
  tensorflow::Tensor input_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &input_tensor));
  input_tensor.scalar<int32_t>()() = 1234;
  args.at(0).Set(tfrt_stub::FallbackTensor(std::move(input_tensor)));

  std::vector<uint8_t> last_uses = {true};
  std::vector<mlrt::Value> results;
  results.resize(1);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(results));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  TF_ASSERT_OK((*ifrt_model_context)
                   ->GetLoadedVariableRegistry()
                   .GetLoadedVariable(kVariableName)
                   .status());

  ExpectEqual(results[0].Get<tfrt_stub::FallbackTensor>().tensor(),
              AsScalar(tsl::tstring(kVariableName)));
}

TEST(KernelTest, DuplicateIfrtLoadVariableOpShallSucceed) {
  auto buffer = CreateExecutableForIfrtLoadVariableOp(
      /*redundant_ifrt_load_variable_op=*/true);

  mlrt::bc::Executable executable(buffer.data());

  mlrt::KernelRegistry registry;
  mlrt::RegisterBuiltinKernels(registry);
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

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  resource_context.CreateResource<tensorflow::ifrt_serving::IfrtModelContext>(
      "IfrtModelContext", client, &GetThreadPool());

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::optional<tensorflow::ifrt_serving::IfrtModelContext*>
      ifrt_model_context =
          resource_context
              .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                  "IfrtModelContext");

  ASSERT_TRUE(ifrt_model_context.has_value());
  EXPECT_THAT((*ifrt_model_context)
                  ->GetLoadedVariableRegistry()
                  .GetLoadedVariable(kVariableName)
                  .status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kNotFound));

  std::vector<mlrt::Value> args;
  args.resize(1);
  tensorflow::Tensor input_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &input_tensor));
  input_tensor.scalar<int32_t>()() = 1234;
  args.at(0).Set(tfrt_stub::FallbackTensor(std::move(input_tensor)));

  std::vector<uint8_t> last_uses = {true};
  std::vector<mlrt::Value> results;
  results.resize(1);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(results));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  TF_ASSERT_OK((*ifrt_model_context)
                   ->GetLoadedVariableRegistry()
                   .GetLoadedVariable(kVariableName)
                   .status());

  ExpectEqual(results[0].Get<tfrt_stub::FallbackTensor>().tensor(),
              AsScalar(tsl::tstring(kVariableName)));
}

}  // namespace
}  // namespace tf_mlrt
}  // namespace tensorflow
