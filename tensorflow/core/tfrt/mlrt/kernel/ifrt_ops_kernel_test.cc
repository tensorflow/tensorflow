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
#include "absl/strings/substitute.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
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
#include "tsl/framework/test_util/mock_serving_device_selector.h"
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
using tensorflow::test::AsTensor;
using tensorflow::test::ExpectEqual;
using tensorflow::test::TensorEq;

constexpr absl::string_view kContainer = "test";
constexpr absl::string_view kSharedName = "y";
constexpr absl::string_view kVariableRuntimeName = "test__y";

tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

std::string EncodeRestoreDtypesInt32(int num_outputs) {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto ctor = mlrt::bc::New<mlrt::bc::Vector<tensorflow::DataType>>(
      &allocator, num_outputs);

  for (int i = 0; i < num_outputs; ++i) {
    ctor.ConstructAt(i, tensorflow::DT_INT32);
  }
  return std::string(buffer.data(), buffer.size());
}

mlrt::bc::Buffer CreateExecutableForIfrtRestoreVariableOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;
  std::vector<std::string> kernel_names = {
      "tf_mlrt.createop", "tf_mlrt.executeop", "tf_mlrt.ifrt_restore_variable",
      "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);

  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(4));

  std::string restore_dtypes = EncodeRestoreDtypesInt32(1);
  attributes.Add("restore_dtypes", restore_dtypes);

  attributes.Add("var_handle_op_node_def",
                 absl::Substitute(
                     R"pb(name: "VarHandleOp"
                          op: "VarHandleOp"
                          device: "/job:localhost/replica:0/task:0/device:CPU:0"
                          attr {
                            key: "container"
                            value { s: "$0" }
                          }
                          attr {
                            key: "shared_name"
                            value { s: "$1" }
                          }
                          attr {
                            key: "dtype"
                            value { type: DT_INT32 }
                          }
                          attr {
                            key: "shape"
                            value { shape { dim { size: 1 } } }
                          }
                     )pb",
                     kContainer, kSharedName));

  attributes.Add("var_handle_op_key", 0);

  auto functions_ctor = executable_ctor.construct_functions(1);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(3).Assign(
        regs.Def({"prefix_tensor", "name_tensor", "slice_tensor"}));

    const int kNumKernels = 4;
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
      auto restore_ctor = kernels_ctor.ConstructAt(kernel_index);
      restore_ctor.set_code(kernels.Use("tf_mlrt.ifrt_restore_variable"));
      restore_ctor.construct_arguments(4).Assign(regs.Use(
          {"prefix_tensor", "name_tensor", "slice_tensor", "variable_handle"}));
      restore_ctor.construct_results(0);
      restore_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("restore_dtypes")});
      kernel_index++;
    }
    {
      auto return_ctor = kernels_ctor.ConstructAt(kernel_index);
      return_ctor.set_code(kernels.Use("return"));
      return_ctor.construct_arguments(0);
      kernel_index++;
    }
    DCHECK_EQ(kernel_index, kNumKernels);
    function_ctor.set_num_regs(regs.size());
  }
  return buffer;
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
      executable_ctor.construct_attributes(5));

  // TODO(b/330360798) Redefine the IfrtLoadVariableOp as it doesn't require the
  // sharding info in the attribute after confirming multihost do not need it.
  attributes.Add("variable_name", kVariableRuntimeName);

  attributes.Add("var_handle_op_node_def",
                 absl::Substitute(
                     R"pb(name: "VarHandleOp"
                          op: "VarHandleOp"
                          device: "/job:localhost/replica:0/task:0/device:CPU:0"
                          attr {
                            key: "container"
                            value { s: "$0" }
                          }
                          attr {
                            key: "shared_name"
                            value { s: "$1" }
                          }
                          attr {
                            key: "dtype"
                            value { type: DT_INT32 }
                          }
                          attr {
                            key: "shape"
                            value { shape { dim { size: 1 } } }
                          }
                     )pb",
                     kContainer, kSharedName));

  attributes.Add("var_handle_op_key", 0);
  attributes.Add("used_by_host", false);
  attributes.Add("sharding_config_proto", "");

  auto functions_ctor = executable_ctor.construct_functions(1);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_output_regs(1).Assign({regs.Def("output_tensor")});

    const int kNumKernels = 4 + (redundant_ifrt_load_variable_op ? 1 : 0);
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
      // Execute VarHandleOp
      auto executeop_ctor = kernels_ctor.ConstructAt(kernel_index);
      executeop_ctor.set_code(kernels.Use("tf_mlrt.executeop"));
      executeop_ctor.construct_arguments(0);
      executeop_ctor.construct_results(1).Assign({regs.Def("variable_handle")});
      executeop_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("var_handle_op_node_def"),
           attributes.GetHandle("var_handle_op_key")});
      kernel_index++;
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.ifrt_load_variable"));
      kernel_ctor.construct_results(2).Assign(
          {regs.Use("output_tensor"), regs.Def("dummy_future")});
      kernel_ctor.construct_arguments(1).Assign({regs.Use("variable_handle")});
      kernel_ctor.construct_attributes(3).Assign(
          {attributes.GetHandle("variable_name"),
           attributes.GetHandle("sharding_config_proto"),
           attributes.GetHandle("used_by_host")});
      kernel_ctor.construct_last_uses(1).Assign(
          {redundant_ifrt_load_variable_op ? 0 : 1});
      kernel_index++;
    }
    if (redundant_ifrt_load_variable_op) {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.ifrt_load_variable"));
      kernel_ctor.construct_results(2).Assign(
          {regs.Def("dummy"), regs.Def("dummy_future2")});
      kernel_ctor.construct_attributes(3).Assign(
          {attributes.GetHandle("variable_name"),
           attributes.GetHandle("sharding_config_proto"),
           attributes.GetHandle("used_by_host")});
      kernel_ctor.construct_arguments(1).Assign({regs.Use("variable_handle")});
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

// TODO(b/330360798) Move other boilerplate code to SetUp.
class KernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    serving_device_selector_ =
        std::make_unique<tsl::test_util::MockServingDeviceSelector>();
    ifrt_core_selector_ =
        std::make_unique<ifrt_serving::IfrtServingCoreSelector>(
            serving_device_selector_.get());
  }

  std::unique_ptr<tsl::test_util::MockServingDeviceSelector>
      serving_device_selector_;
  std::unique_ptr<ifrt_serving::IfrtServingCoreSelector> ifrt_core_selector_;
};

TEST_F(KernelTest, IfrtLoadVariableOp) {
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
      "IfrtModelContext", client, ifrt_core_selector_.get(), &GetThreadPool());

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::optional<tensorflow::ifrt_serving::IfrtModelContext*>
      ifrt_model_context =
          resource_context
              .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                  "IfrtModelContext");

  ASSERT_TRUE(ifrt_model_context.has_value());
  auto restore_work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  (*ifrt_model_context)->set_checkpoint_loader_queue(restore_work_queue.get());

  tensorflow::Tensor input_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &input_tensor));
  input_tensor.scalar<int32_t>()() = 1234;
  auto input_tensor_promise =
      xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto input_tensor_future =
      xla::ifrt::Future<tensorflow::Tensor>(input_tensor_promise);
  ifrt_serving::IfrtRestoreTensorRegistry::RestoredTensorInfo
      restore_tensor_info{.dtype_and_shape = {.dtype = input_tensor.dtype(),
                                              .shape = input_tensor.shape()},
                          .tensor_future = input_tensor_future};
  input_tensor_promise.Set(input_tensor);
  TF_ASSERT_OK((*ifrt_model_context)
                   ->GetRestoreTensorRegistry()
                   .TryRegister(kVariableRuntimeName, restore_tensor_info));

  std::vector<mlrt::Value> args;
  std::vector<uint8_t> last_uses;
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

  ExpectEqual(results[0].Get<tfrt_stub::FallbackTensor>().tensor(),
              AsScalar(tsl::tstring(kVariableRuntimeName)));
}

TEST_F(KernelTest, DuplicateIfrtLoadVariableOpShallSucceed) {
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
      "IfrtModelContext", client, ifrt_core_selector_.get(), &GetThreadPool());

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::optional<tensorflow::ifrt_serving::IfrtModelContext*>
      ifrt_model_context =
          resource_context
              .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                  "IfrtModelContext");

  ASSERT_TRUE(ifrt_model_context.has_value());

  auto restore_work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  (*ifrt_model_context)->set_checkpoint_loader_queue(restore_work_queue.get());

  tensorflow::Tensor input_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &input_tensor));
  input_tensor.scalar<int32_t>()() = 1234;
  auto input_tensor_promise =
      xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto input_tensor_future =
      xla::ifrt::Future<tensorflow::Tensor>(input_tensor_promise);
  ifrt_serving::IfrtRestoreTensorRegistry::RestoredTensorInfo
      restore_tensor_info{.dtype_and_shape = {.dtype = input_tensor.dtype(),
                                              .shape = input_tensor.shape()},
                          .tensor_future = input_tensor_future};
  input_tensor_promise.Set(input_tensor);
  TF_ASSERT_OK((*ifrt_model_context)
                   ->GetRestoreTensorRegistry()
                   .TryRegister(kVariableRuntimeName, restore_tensor_info));

  std::vector<mlrt::Value> args;
  std::vector<uint8_t> last_uses;
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

  ExpectEqual(results[0].Get<tfrt_stub::FallbackTensor>().tensor(),
              AsScalar(tsl::tstring(kVariableRuntimeName)));
}

TEST_F(KernelTest, IfrtRestoreVariableOp) {
  std::string checkpoint_prefix =
      tensorflow::GetDataDependencyFilepath(
          "tensorflow/core/tfrt/mlrt/kernel/testdata/"
          "gen_checkpoint_data/variables") +
      "/variables";

  auto buffer = CreateExecutableForIfrtRestoreVariableOp();

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
      "IfrtModelContext", client, ifrt_core_selector_.get(), &GetThreadPool());

  auto tf_context =
      std::make_unique<Context>(&fallback_request_state, &resource_context);
  execution_context.AddUserContext(std::move(tf_context));

  std::optional<tensorflow::ifrt_serving::IfrtModelContext*>
      ifrt_model_context =
          resource_context
              .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                  "IfrtModelContext");

  ASSERT_TRUE(ifrt_model_context.has_value());
  xla::ifrt::Future<tensorflow::Tensor> uninitialized_entry =
      (*ifrt_model_context)
          ->GetRestoreTensorRegistry()
          .GetRestoredTensor(kVariableRuntimeName);
  ASSERT_TRUE(uninitialized_entry.IsReady());
  EXPECT_THAT(uninitialized_entry.Await().status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kNotFound));

  auto restore_work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  (*ifrt_model_context)->set_checkpoint_loader_queue(restore_work_queue.get());

  std::vector<mlrt::Value> args;
  args.resize(3);

  tensorflow::Tensor prefix_tensor =
      AsTensor<tsl::tstring>({tsl::tstring(checkpoint_prefix)});
  args.at(0).Set(tfrt_stub::FallbackTensor(std::move(prefix_tensor)));

  tensorflow::Tensor name_tensor =
      AsTensor<tsl::tstring>({tsl::tstring("w/.ATTRIBUTES/VARIABLE_VALUE")});
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(name_tensor)));

  tensorflow::Tensor slice_tensor = AsTensor<tsl::tstring>({tsl::tstring("")});
  args.at(2).Set(tfrt_stub::FallbackTensor(std::move(slice_tensor)));

  std::vector<uint8_t> last_uses = {true, true, true};
  std::vector<mlrt::Value> results;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(results));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  xla::ifrt::Future<tensorflow::Tensor> restored_future =
      (*ifrt_model_context)
          ->GetRestoreTensorRegistry()
          .GetRestoredTensor(kVariableRuntimeName);
  absl::StatusOr<tensorflow::Tensor> restored_tensor = restored_future.Await();
  TF_ASSERT_OK(restored_tensor.status());
  EXPECT_THAT(*restored_tensor, TensorEq(AsTensor<int32_t>({1, 2, 3}, {3})));
}

}  // namespace
}  // namespace tf_mlrt
}  // namespace tensorflow
