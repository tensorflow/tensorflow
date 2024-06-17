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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "tensorflow/core/framework/resource_var.h"
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
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/refcount.h"
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

std::string EncodeTruncateInCast(int num_outputs) {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto ctor = mlrt::bc::New<mlrt::bc::Vector<bool>>(&allocator, num_outputs);

  for (int i = 0; i < num_outputs; ++i) {
    ctor.ConstructAt(i, false);
  }
  return std::string(buffer.data(), buffer.size());
}

mlrt::bc::Buffer CreateExecutableForIfrtRestoreVariableOp(
    int num_variables = 1) {
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

  static constexpr int kNumAttributes =
      5;  // Size of attributes when there are 1 variable.
  mlrt::testing::AttributeTable attributes(executable_ctor.construct_attributes(
      kNumAttributes + 2 * (num_variables - 1)));

  std::string restore_dtypes = EncodeRestoreDtypesInt32(num_variables);
  attributes.Add("restore_dtypes", restore_dtypes);
  std::vector<bool> truncate_in_cast(num_variables, false);
  attributes.Add("truncate_in_cast", EncodeTruncateInCast(num_variables));

  for (int i = 0; i < num_variables; ++i) {
    attributes.Add(
        absl::StrCat("var_handle_op_node_def", i),
        absl::Substitute(
            R"pb(name: "$0"
                 op: "VarHandleOp"
                 device: "/job:localhost/replica:0/task:0/device:CPU:0"
                 attr {
                   key: "container"
                   value { s: "$1" }
                 }
                 attr {
                   key: "shared_name"
                   value { s: "$2" }
                 }
                 attr {
                   key: "dtype"
                   value { type: DT_INT16 }
                 }
                 attr {
                   key: "shape"
                   value { shape { dim { size: 3 } } }
                 }
            )pb",
            absl::StrCat("VarHandleOp", i), kContainer,
            absl::StrCat(kSharedName, i)));

    attributes.Add(absl::StrCat("var_handle_op_key", i), i);
  }

  auto functions_ctor = executable_ctor.construct_functions(1);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(3).Assign(
        regs.Def({"prefix_tensor", "name_tensor", "slice_tensor"}));

    const int kNumKernels = 4;
    auto kernels_ctor =
        function_ctor.construct_kernels(kNumKernels + 2 * (num_variables - 1));
    int kernel_index = 0;

    std::vector<std::string> variable_handle_names;
    variable_handle_names.reserve(num_variables);
    for (int i = 0; i < num_variables; ++i) {
      variable_handle_names.push_back(absl::StrCat("variable_handle", i));
      std::string variable_handle_op_node_def =
          absl::StrCat("var_handle_op_node_def", i);
      std::string variable_handle_op_key = absl::StrCat("var_handle_op_key", i);

      {
        // Create VarHandleOp
        auto createop_ctor = kernels_ctor.ConstructAt(kernel_index);
        createop_ctor.set_code(kernels.Use("tf_mlrt.createop"));
        createop_ctor.construct_arguments(0);
        createop_ctor.construct_results(0);
        createop_ctor.construct_attributes(2).Assign(
            {attributes.GetHandle(variable_handle_op_node_def),
             attributes.GetHandle(variable_handle_op_key)});
        kernel_index++;
      }
      {
        // Execute VarHandleOp
        auto executeop_ctor = kernels_ctor.ConstructAt(kernel_index);
        executeop_ctor.set_code(kernels.Use("tf_mlrt.executeop"));
        executeop_ctor.construct_arguments(0);
        executeop_ctor.construct_results(1).Assign(
            {regs.Def(variable_handle_names.back())});
        executeop_ctor.construct_attributes(2).Assign(
            {attributes.GetHandle(variable_handle_op_node_def),
             attributes.GetHandle(variable_handle_op_key)});
        executeop_ctor.construct_last_uses(1).Assign({0});
        kernel_index++;
      }
    }
    {
      std::vector<std::string> args;
      args.reserve(3 + num_variables);
      args.push_back("prefix_tensor");
      args.push_back("name_tensor");
      args.push_back("slice_tensor");
      for (int i = 0; i < num_variables; ++i) {
        args.push_back(variable_handle_names[i]);
      }
      auto restore_ctor = kernels_ctor.ConstructAt(kernel_index);
      restore_ctor.set_code(kernels.Use("tf_mlrt.ifrt_restore_variable"));
      restore_ctor.construct_arguments(args.size()).Assign(regs.Use(args));
      restore_ctor.construct_results(0);
      restore_ctor.construct_attributes(2).Assign(
          {attributes.GetHandle("restore_dtypes"),
           attributes.GetHandle("truncate_in_cast")});
      kernel_index++;
    }
    {
      auto return_ctor = kernels_ctor.ConstructAt(kernel_index);
      return_ctor.set_code(kernels.Use("return"));
      return_ctor.construct_arguments(0);
      kernel_index++;
    }
    function_ctor.set_num_regs(regs.size());
  }
  return buffer;
}

mlrt::bc::Buffer CreateExecutableForIfrtLoadVariableOp(
    bool redundant_ifrt_load_variable_op = false, bool used_by_host = false) {
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
      executable_ctor.construct_attributes(3));

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
  attributes.Add("used_by_host", used_by_host);

  auto functions_ctor = executable_ctor.construct_functions(1);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_output_regs(2).Assign(
        {regs.Def("output_tensor"), regs.Def("output_future")});

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
          {regs.Use("output_tensor"), regs.Use("output_future")});
      kernel_ctor.construct_arguments(1).Assign({regs.Use("variable_handle")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("used_by_host")});
      kernel_ctor.construct_last_uses(1).Assign(
          {redundant_ifrt_load_variable_op ? 0 : 1});
      kernel_index++;
    }
    if (redundant_ifrt_load_variable_op) {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("tf_mlrt.ifrt_load_variable"));
      kernel_ctor.construct_results(2).Assign(
          {regs.Def("dummy"), regs.Def("dummy_future2")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("used_by_host")});
      kernel_ctor.construct_arguments(1).Assign({regs.Use("variable_handle")});
      kernel_ctor.construct_last_uses(1).Assign({1});
      kernel_index++;
    }
    {
      auto kernel_ctor = kernels_ctor.ConstructAt(kernel_index);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(2).Assign(
          {regs.Use("output_tensor"), regs.Use("output_future")});
      kernel_index++;
    }
    DCHECK_EQ(kernel_index, kNumKernels);

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

class KernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlrt::RegisterBuiltinKernels(registry_);
    RegisterTfMlrtKernels(registry_);
    execution_work_queue_ = tfrt::CreateMultiThreadedWorkQueue(
        /*num_threads=*/4, /*num_blocking_threads=*/4);
    restore_work_queue_ = tfrt::CreateMultiThreadedWorkQueue(
        /*num_threads=*/4, /*num_blocking_threads=*/4);
    TF_ASSERT_OK_AND_ASSIGN(fallback_state_, tfrt_stub::FallbackState::Create(
                                                 session_options_, fdef_lib_));
    runner_ = [](const std::function<void()>& f) { f(); };
    fallback_request_state_ =
        std::make_unique<tfd::KernelFallbackCompatRequestState>(
            &runner_, &fallback_state_->device_manager(), /*step_id=*/0,
            &runner_table_, &resource_array_,
            /*user_intra_op_threadpool=*/nullptr,
            /*model_metadata=*/std::nullopt,
            &fallback_state_->process_function_library_runtime());

    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    resource_context_
        .CreateResource<tensorflow::ifrt_serving::IfrtModelContext>(
            "IfrtModelContext", client_, ifrt_core_selector_.get(),
            &GetThreadPool(), /*compilation_environment_proto=*/nullptr);

    tf_context_ = std::make_unique<Context>(fallback_request_state_.get(),
                                            &resource_context_);
    ifrt_model_context_ =
        resource_context_
            .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                "IfrtModelContext")
            .value();
    ifrt_model_context_->set_checkpoint_loader_queue(restore_work_queue_.get());

    serving_device_selector_ =
        std::make_unique<tsl::test_util::MockServingDeviceSelector>();
    ifrt_core_selector_ =
        std::make_unique<ifrt_serving::IfrtServingCoreSelector>(
            serving_device_selector_.get(),
            client_->addressable_device_count());
  }

  std::unique_ptr<tsl::test_util::MockServingDeviceSelector>
      serving_device_selector_;
  std::unique_ptr<ifrt_serving::IfrtServingCoreSelector> ifrt_core_selector_;
  mlrt::KernelRegistry registry_;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> execution_work_queue_;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> restore_work_queue_;
  tensorflow::SessionOptions session_options_;
  tensorflow::FunctionDefLibrary fdef_lib_;
  std::function<void(std::function<void()>)> runner_;
  tfrt_stub::OpKernelRunnerTable runner_table_;
  tfd::FallbackResourceArray resource_array_;
  std::unique_ptr<tfrt_stub::FallbackState> fallback_state_;
  tfrt::ResourceContext resource_context_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tfd::KernelFallbackCompatRequestState>
      fallback_request_state_;
  std::unique_ptr<Context> tf_context_;
  tensorflow::ifrt_serving::IfrtModelContext* ifrt_model_context_;
};

TEST_F(KernelTest, IfrtLoadVariableOpCanGetTensorFromResourceManager) {
  auto buffer = CreateExecutableForIfrtLoadVariableOp(
      /*redundant_ifrt_load_variable_op=*/false, /*used_by_host=*/true);

  mlrt::bc::Executable executable(buffer.data());

  mlrt::LoadedExecutable loaded_executable(executable, registry_);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(execution_work_queue_.get());

  execution_context.AddUserContext(std::move(tf_context_));

  tensorflow::Tensor input_tensor;
  TF_CHECK_OK(tensorflow::Tensor::BuildTensor(DT_INT32, {}, &input_tensor));
  input_tensor.scalar<int32_t>()() = 1234;

  tsl::core::RefCountPtr<Var> variable(new Var(DT_INT32));
  *variable->tensor() = input_tensor;
  variable->is_initialized = true;
  ASSERT_OK(
      fallback_state_->device_manager().HostCPU()->resource_manager()->Create(
          std::string(kContainer), std::string(kSharedName), &(*variable)));

  std::vector<mlrt::Value> args;
  std::vector<uint8_t> last_uses;
  std::vector<mlrt::Value> results;
  results.resize(2);

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
  auto returned_future = results[1].Get<mlrt::Future>();
  ASSERT_TRUE(returned_future.IsReady());
  EXPECT_THAT(returned_future.Get<tfrt_stub::FallbackTensor>().tensor(),
              TensorEq(input_tensor));
}

TEST_F(KernelTest, IfrtLoadVariableOp) {
  auto buffer = CreateExecutableForIfrtLoadVariableOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::LoadedExecutable loaded_executable(executable, registry_);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(execution_work_queue_.get());

  execution_context.AddUserContext(std::move(tf_context_));

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
  TF_ASSERT_OK(ifrt_model_context_->GetRestoreTensorRegistry().TryRegister(
      kVariableRuntimeName, restore_tensor_info));

  std::vector<mlrt::Value> args;
  std::vector<uint8_t> last_uses;
  std::vector<mlrt::Value> results;
  results.resize(2);

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
  auto returned_future = results[1].Get<mlrt::Future>();
  ASSERT_TRUE(returned_future.IsReady());
  // Returned is an empty tensor since it is not used by host.
  EXPECT_THAT(returned_future.Get<tfrt_stub::FallbackTensor>().tensor(),
              TensorEq(tensorflow::Tensor()));
}

TEST_F(KernelTest, DuplicateIfrtLoadVariableOpShallSucceed) {
  auto buffer = CreateExecutableForIfrtLoadVariableOp(
      /*redundant_ifrt_load_variable_op=*/true);

  mlrt::bc::Executable executable(buffer.data());

  mlrt::LoadedExecutable loaded_executable(executable, registry_);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(execution_work_queue_.get());
  execution_context.AddUserContext(std::move(tf_context_));

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
  TF_ASSERT_OK(ifrt_model_context_->GetRestoreTensorRegistry().TryRegister(
      kVariableRuntimeName, restore_tensor_info));

  std::vector<mlrt::Value> args;
  std::vector<uint8_t> last_uses;
  std::vector<mlrt::Value> results;
  results.resize(2);

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

  auto returned_future = results[1].Get<mlrt::Future>();
  ASSERT_TRUE(returned_future.IsReady());
  // Returned is an empty tensor since it is not used by host.
  EXPECT_THAT(returned_future.Get<tfrt_stub::FallbackTensor>().tensor(),
              TensorEq(tensorflow::Tensor()));
}

TEST_F(KernelTest, IfrtRestoreVariableOp) {
  std::string checkpoint_prefix =
      tensorflow::GetDataDependencyFilepath(
          "tensorflow/core/tfrt/mlrt/kernel/testdata/"
          "gen_checkpoint_data/variables") +
      "/variables";

  auto buffer = CreateExecutableForIfrtRestoreVariableOp();

  mlrt::bc::Executable executable(buffer.data());

  mlrt::LoadedExecutable loaded_executable(executable, registry_);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(execution_work_queue_.get());

  execution_context.AddUserContext(std::move(tf_context_));

  xla::ifrt::Future<tensorflow::Tensor> uninitialized_entry =
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          kVariableRuntimeName);
  ASSERT_TRUE(uninitialized_entry.IsReady());
  EXPECT_THAT(uninitialized_entry.Await().status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kNotFound));

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
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          absl::StrCat(kVariableRuntimeName, 0));
  absl::StatusOr<tensorflow::Tensor> restored_tensor = restored_future.Await();
  TF_ASSERT_OK(restored_tensor.status());
  EXPECT_THAT(*restored_tensor, TensorEq(AsTensor<int16_t>({1, 2, 3}, {3})));
}

TEST_F(KernelTest, IfrtRestoreVariableOp4Variables) {
  std::string checkpoint_prefix =
      tensorflow::GetDataDependencyFilepath(
          "tensorflow/core/tfrt/mlrt/kernel/testdata/"
          "gen_checkpoint_data/variables") +
      "/variables";

  static constexpr int kNumVariables = 4;
  auto buffer = CreateExecutableForIfrtRestoreVariableOp(kNumVariables);

  mlrt::bc::Executable executable(buffer.data());

  mlrt::LoadedExecutable loaded_executable(executable, registry_);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(execution_work_queue_.get());

  execution_context.AddUserContext(std::move(tf_context_));

  xla::ifrt::Future<tensorflow::Tensor> uninitialized_entry =
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          kVariableRuntimeName);
  ASSERT_TRUE(uninitialized_entry.IsReady());
  EXPECT_THAT(uninitialized_entry.Await().status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kNotFound));

  std::vector<mlrt::Value> args;
  args.resize(3);

  tensorflow::Tensor prefix_tensor =
      AsTensor<tsl::tstring>({tsl::tstring(checkpoint_prefix)});
  args.at(0).Set(tfrt_stub::FallbackTensor(std::move(prefix_tensor)));

  tensorflow::Tensor name_tensor =
      AsTensor<tsl::tstring>({tsl::tstring("w/.ATTRIBUTES/VARIABLE_VALUE"),
                              tsl::tstring("w1/.ATTRIBUTES/VARIABLE_VALUE"),
                              tsl::tstring("w2/.ATTRIBUTES/VARIABLE_VALUE"),
                              tsl::tstring("w3/.ATTRIBUTES/VARIABLE_VALUE")});
  args.at(1).Set(tfrt_stub::FallbackTensor(std::move(name_tensor)));

  tensorflow::Tensor slice_tensor = AsTensor<tsl::tstring>(
      {tsl::tstring(""), tsl::tstring(""), tsl::tstring(""), tsl::tstring("")});
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
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          absl::StrCat(kVariableRuntimeName, 0));
  absl::StatusOr<tensorflow::Tensor> restored_tensor = restored_future.Await();
  TF_ASSERT_OK(restored_tensor.status());
  EXPECT_THAT(*restored_tensor, TensorEq(AsTensor<int16_t>({1, 2, 3}, {3})));

  xla::ifrt::Future<tensorflow::Tensor> restored_future1 =
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          absl::StrCat(kVariableRuntimeName, 1));
  absl::StatusOr<tensorflow::Tensor> restored_tensor1 =
      restored_future1.Await();
  TF_ASSERT_OK(restored_tensor1.status());
  EXPECT_THAT(*restored_tensor1, TensorEq(AsTensor<int16_t>({4, 5, 6}, {3})));

  xla::ifrt::Future<tensorflow::Tensor> restored_future2 =
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          absl::StrCat(kVariableRuntimeName, 2));
  absl::StatusOr<tensorflow::Tensor> restored_tensor2 =
      restored_future2.Await();
  TF_ASSERT_OK(restored_tensor2.status());
  EXPECT_THAT(*restored_tensor2, TensorEq(AsTensor<int16_t>({7, 8, 9}, {3})));

  xla::ifrt::Future<tensorflow::Tensor> restored_future3 =
      ifrt_model_context_->GetRestoreTensorRegistry().GetRestoredTensor(
          absl::StrCat(kVariableRuntimeName, 3));
  absl::StatusOr<tensorflow::Tensor> restored_tensor3 =
      restored_future3.Await();
  TF_ASSERT_OK(restored_tensor3.status());
  EXPECT_THAT(*restored_tensor3,
              TensorEq(AsTensor<int16_t>({10, 11, 12}, {3})));
}

}  // namespace
}  // namespace tf_mlrt
}  // namespace tensorflow
