/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/kernel/batch_kernel.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/tfrt/fallback/device_with_custom_allocator.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/interpreter_testutil.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {
namespace {
using tensorflow::test::ExpectEqual;

void PopulateBatchTestAttributesHelper(
    mlrt::testing::AttributeTable& attributes) {
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
                        key: "batch_padding_policy"
                        value { s: "PAD_UP" }
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
  PopulateBatchTestAttributesHelper(attributes);

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

  ExpectEqual(result.Get<tfrt_stub::FallbackTensor>().tensor(), expected);
}

mlrt::bc::Buffer CreateExecutableForBatchFunctionWithDeviceOp() {
  mlrt::bc::Buffer buffer;
  mlrt::bc::Allocator allocator(&buffer);

  auto executable_ctor = mlrt::bc::New<mlrt::bc::Executable>(&allocator);

  mlrt::testing::SymbolTable kernels;

  std::vector<std::string> kernel_names = {
      "tf_mlrt.createop", "tf_mlrt.executeop", "tf_mlrt.batch_function.device",
      "tf_mlrt.await", "return"};

  executable_ctor.construct_kernel_names(kernel_names.size())
      .Assign(kernel_names);
  kernels.Def(kernel_names);
  mlrt::testing::AttributeTable attributes(
      executable_ctor.construct_attributes(5));
  PopulateBatchTestAttributesHelper(attributes);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    mlrt::testing::SymbolTable regs;

    function_ctor.construct_input_regs(2).Assign(
        {regs.Def("custom_device"), regs.Def("input")});

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
      kernel_ctor.set_code(kernels.Use("tf_mlrt.batch_function.device"));
      kernel_ctor.construct_arguments(3).Assign(
          regs.Use({"custom_device", "input", "input"}));
      kernel_ctor.construct_results(1).Assign({regs.Def("result_future")});
      kernel_ctor.construct_attributes(3).Assign(
          {attributes.GetHandle("device"), attributes.GetHandle("func_idx"),
           attributes.GetHandle("batch_node_def_str")});
      kernel_ctor.construct_last_uses(3).Assign({0, 0, 0});
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

TEST(KernelTest, BatchFunctionWithDeviceOp) {
  auto buffer = CreateExecutableForBatchFunctionWithDeviceOp();

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

  class TestDevice : public tfrt_stub::DeviceWithCustomAllocator {
   public:
    using Base = tfrt_stub::DeviceWithCustomAllocator;
    using DeviceWithCustomAllocator::DeviceWithCustomAllocator;
    Allocator* GetAllocator(AllocatorAttributes attr) override {
      allocator_uses_++;
      return Base::GetAllocator(attr);
    }

    int allocator_uses() const { return allocator_uses_; }

   private:
    std::atomic<int> allocator_uses_ = 0;
  };

  std::vector<mlrt::Value> args;
  auto test_device = std::make_shared<TestDevice>(
      fallback_request_state.cpu_device(),
      fallback_request_state.cpu_device()->GetAllocator({}));
  args.push_back(mlrt::Value(std::shared_ptr<Device>(test_device)));

  tensorflow::Tensor input_tensor(tensorflow::DT_INT32, {1});
  input_tensor.flat<int32_t>()(0) = 100;
  args.push_back(
      mlrt::Value(tfrt_stub::FallbackTensor(std::move(input_tensor))));
  mlrt::Value result;

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::MakeSpan(args), absl::MakeSpan(&result, 1));
  mlrt::Execute(execution_context);

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());

  tensorflow::Tensor expected(tensorflow::DT_INT32, {1});
  expected.flat<int32_t>()(0) = 200;

  ExpectEqual(result.Get<tfrt_stub::FallbackTensor>().tensor(), expected);
  // Two input tensors; hence two allocator uses.
  EXPECT_EQ(test_device->allocator_uses(), 2);
}

}  // namespace
}  // namespace tf_mlrt
}  // namespace tensorflow
