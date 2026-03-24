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
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/async_handle.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/builtin_kernels.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/interpreter_testutil.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace mlrt {
namespace {

class AddI32Kernel : public KernelFrame {
 public:
  using KernelFrame::KernelFrame;

  int32_t arg0() const { return arguments()[kArg0Index].Get<int32_t>(); }
  int32_t arg1() const { return arguments()[kArg1Index].Get<int32_t>(); }

  void set_result(int32_t result) { results()[kResultIndex].Set(result); }

  void Invoke() { set_result(arg0() + arg1()); }

  static constexpr char kName[] = "add";

 private:
  static constexpr int kArg0Index = 0;
  static constexpr int kArg1Index = 1;

  static constexpr int kResultIndex = 0;
};

void AddI32Const(KernelFrame frame) {
  auto args = frame.arguments();
  int32_t constant = frame.attributes().GetAs<int32_t>(0);
  frame.results()[0].Set(args[0].Get<int32_t>() + constant);
}

bc::Buffer CreateSequentialAddExecutable(int num_add) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"add", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);

  testing::SymbolTable regs;

  function_ctor.construct_name("main");
  function_ctor.construct_input_regs(1).Assign({regs.Def("r0")});
  function_ctor.construct_output_last_uses(1).Assign({true});

  auto kernels_ctor = function_ctor.construct_kernels(num_add + 1);

  {
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("add"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"r0", "r0"}));
    kernel_ctor.construct_results(1).Assign({regs.Def("r1")});
  }

  for (int i = 1; i < num_add; ++i) {
    auto kernel_ctor = kernels_ctor.ConstructAt(i);
    kernel_ctor.set_code(kernels.Use("add"));
    kernel_ctor.construct_arguments(2).Assign(
        regs.Use({absl::StrCat("r", (i + 1) % 2 + 1), "r0"}));
    kernel_ctor.construct_results(1).Assign(
        {regs.Def(absl::StrCat("r", i % 2 + 1))});
  }

  auto kernel_ctor = kernels_ctor.ConstructAt(num_add);
  kernel_ctor.set_code(kernels.Use("return"));
  kernel_ctor.construct_arguments(1).Assign(
      {regs.Use(absl::StrCat("r", (num_add - 1) % 2 + 1))});

  function_ctor.construct_output_regs(1).Assign(
      {regs.Use(absl::StrCat("r", (num_add - 1) % 2 + 1))});
  function_ctor.set_num_regs(regs.size());

  return buffer;
}

bc::Buffer CreateSequentialAddAttributesExecutable(int num_add) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"add.const", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("op_key", 1);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);

  testing::SymbolTable regs;

  function_ctor.construct_name("main");
  function_ctor.construct_input_regs(1).Assign({regs.Def("r0")});

  auto kernels_ctor = function_ctor.construct_kernels(num_add + 1);
  for (int i = 0; i < num_add; ++i) {
    auto kernel_ctor = kernels_ctor.ConstructAt(i);
    kernel_ctor.set_code(kernels.Use("add.const"));
    kernel_ctor.construct_arguments(1).Assign({regs.Use(absl::StrCat("r", i))});
    kernel_ctor.construct_results(1).Assign(
        {regs.Def(absl::StrCat("r", i + 1))});
    kernel_ctor.construct_attributes(1).Assign(
        {attributes.GetHandle("op_key")});
  }

  auto kernel_ctor = kernels_ctor.ConstructAt(num_add);
  kernel_ctor.set_code(kernels.Use("return"));
  kernel_ctor.construct_arguments(1).Assign(
      {regs.Use(absl::StrCat("r", num_add))});

  function_ctor.construct_output_regs(1).Assign(
      {regs.Use(absl::StrCat("r", num_add))});
  function_ctor.set_num_regs(regs.size());

  return buffer;
}

TEST(InterpreterTest, SequentialAdd) {
  auto buffer = CreateSequentialAddExecutable(99);

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register<AddI32Kernel>();

  LoadedExecutable loaded_executable(executable, kernel_registry);

  absl::Notification notification;

  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_exit_handler([&]() { notification.Notify(); });

  int32_t v = 1;
  mlrt::Value arg(v);
  mlrt::Value result;

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&arg, 1),
                         absl::Span<Value>(&result, 1));
  Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_EQ(result.Get<int32_t>(), 100);
}

TEST(InterpreterTest, SequentialAddAttributes) {
  auto buffer = CreateSequentialAddAttributesExecutable(99);

  bc::Executable executable(buffer.Get(0));

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("add.const", &AddI32Const);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  absl::Notification notification;

  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_exit_handler([&]() { notification.Notify(); });

  int32_t v = 1;
  mlrt::Value arg(v);
  mlrt::Value result;

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&arg, 1),
                         absl::Span<Value>(&result, 1));
  Execute(execution_context);

  notification.WaitForNotification();

  EXPECT_EQ(result.Get<int32_t>(), 100);
}

bc::Buffer CreateCallExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("op_key", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"call", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    testing::SymbolTable regs;

    auto caller_ctor = functions_ctor.ConstructAt(0);
    caller_ctor.construct_name("caller");
    caller_ctor.construct_input_regs(1).Assign({regs.Def("arg")});

    auto kernels_ctor = caller_ctor.construct_kernels(2);
    {
      // Call
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("call"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("arg")});
      kernel_ctor.construct_last_uses(1).Assign({true});
      kernel_ctor.construct_results(1).Assign({regs.Def("result")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("op_key")});
    }

    {
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
    }

    caller_ctor.construct_output_regs(1).Assign({regs.Use("result")});
    caller_ctor.set_num_regs(regs.size());
  }

  {
    testing::SymbolTable regs;

    auto callee_ctor = functions_ctor.ConstructAt(1);
    callee_ctor.construct_name("callee");
    callee_ctor.construct_input_regs(1).Assign({regs.Def("arg")});

    {
      auto kernels_ctor = callee_ctor.construct_kernels(1);

      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("arg")});
    }

    callee_ctor.construct_output_regs(1).Assign({regs.Use("arg")});
    callee_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, Call) {
  auto buffer = CreateCallExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("caller");
  ASSERT_TRUE(function);

  Value input(123);
  Value output;

  std::vector<uint8_t> last_uses = {false};
  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output.Get<int>(), 123);
  EXPECT_TRUE(input.HasValue());
}

bc::Buffer CreateCondExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(2));

  attributes.Add("then_idx", 1);

  attributes.Add("else_idx", 2);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.cond", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(3);
  {
    auto caller_ctor = functions_ctor.ConstructAt(0);
    caller_ctor.construct_name("caller");

    testing::SymbolTable regs;

    caller_ctor.construct_input_regs(3).Assign(regs.Def({"cond", "x", "y"}));

    {
      auto kernels_ctor = caller_ctor.construct_kernels(2);
      {
        // mlrt.cond
        auto kernel_ctor = kernels_ctor.ConstructAt(0);
        kernel_ctor.set_code(kernels.Use("mlrt.cond"));
        kernel_ctor.construct_arguments(3).Assign(regs.Use({"cond", "x", "y"}));
        kernel_ctor.construct_last_uses(3).Assign({true, true, true});
        kernel_ctor.construct_results(1).Assign({regs.Def("z")});
        kernel_ctor.construct_attributes(2).Assign(
            {attributes.GetHandle("then_idx"),
             attributes.GetHandle("else_idx")});
      }

      {
        // Return
        auto kernel_ctor = kernels_ctor.ConstructAt(1);
        kernel_ctor.set_code(kernels.Use("return"));
        kernel_ctor.construct_arguments(1).Assign({regs.Use("z")});
      }
    }

    caller_ctor.set_num_regs(regs.size());
    caller_ctor.construct_output_regs(1).Assign({regs.Use("z")});
  }

  {
    auto then_ctor = functions_ctor.ConstructAt(1);
    then_ctor.construct_name("then");

    testing::SymbolTable regs;

    then_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"x", "y"}));

    {
      auto kernels_ctor = then_ctor.construct_kernels(1);

      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("x")});
    }

    then_ctor.set_num_regs(regs.size());
    then_ctor.construct_output_regs(1).Assign({regs.Use("x")});
  }

  {
    auto else_ctor = functions_ctor.ConstructAt(2);
    else_ctor.construct_name("else");

    testing::SymbolTable regs;

    else_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"x", "y"}));

    {
      auto kernels_ctor = else_ctor.construct_kernels(1);

      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("y")});
    }

    else_ctor.set_num_regs(regs.size());
    else_ctor.construct_output_regs(1).Assign({regs.Use("y")});
  }

  return buffer;
}

TEST(InterpreterTest, Cond) {
  auto buffer = CreateCondExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("caller");
  ASSERT_TRUE(function);

  Value inputs[3];
  inputs[0].Set(true);
  inputs[1].Set(100);
  inputs[2].Set(200);
  Value output;

  std::vector<uint8_t> last_uses = {true, false, false};
  execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output.Get<int>(), 100);

  ASSERT_TRUE(inputs[1].HasValue());
  ASSERT_TRUE(inputs[2].HasValue());
  ASSERT_EQ(inputs[1].Get<int>(), 100);
  ASSERT_EQ(inputs[2].Get<int>(), 200);

  inputs[0].Set(false);
  execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output.Get<int>(), 200);
}

bc::Buffer CreateNestedCallExecutable(int num_calls) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  for (int i = 0; i < num_calls; ++i) {
    attributes.Add(absl::StrCat("f_id_", i), i);
  }

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"call", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(num_calls);

  for (int i = 0; i < num_calls - 1; ++i) {
    testing::SymbolTable regs;

    auto caller_ctor = functions_ctor.ConstructAt(i);
    caller_ctor.construct_name(absl::StrCat("call_", i));
    caller_ctor.construct_input_regs(1).Assign({regs.Def("arg")});

    auto kernels_ctor = caller_ctor.construct_kernels(2);
    {
      // Call
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("call"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("arg")});
      kernel_ctor.construct_last_uses(1).Assign({true});
      kernel_ctor.construct_results(1).Assign({regs.Def("result")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle(absl::StrCat("f_id_", i + 1))});
    }

    {
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
    }

    caller_ctor.construct_output_regs(1).Assign({regs.Use("result")});
    caller_ctor.set_num_regs(regs.size());
  }

  {
    testing::SymbolTable regs;

    auto callee_ctor = functions_ctor.ConstructAt(num_calls - 1);
    callee_ctor.construct_name(absl::StrCat("call_", num_calls));
    callee_ctor.construct_input_regs(1).Assign({regs.Def("arg")});

    {
      auto kernels_ctor = callee_ctor.construct_kernels(1);

      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("arg")});
    }

    callee_ctor.construct_output_regs(1).Assign({regs.Use("arg")});
    callee_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, NestedCall) {
  auto buffer = CreateNestedCallExecutable(32);

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("call_0");
  ASSERT_TRUE(function);

  Value input(123);
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output.Get<int>(), 123);
}

bc::Buffer CreateFailExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"fail", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  auto kernels_ctor = function_ctor.construct_kernels(2);

  {
    // Fail
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("fail"));
  }

  {
    // Return
    auto kernel_ctor = kernels_ctor.ConstructAt(1);
    kernel_ctor.set_code(kernels.Use("return"));
  }

  return buffer;
}

void Fail(KernelFrame frame) {
  frame.execution_context().Fail(absl::InternalError("test error"));
}

TEST(InterpreterTest, Fail) {
  auto buffer = CreateFailExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("fail", &Fail);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  std::vector<uint8_t> last_uses;
  execution_context.Call(function, last_uses, absl::Span<Value>(),
                         absl::Span<Value>());
  Execute(execution_context);

  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

bc::Buffer CreateAwaitExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"await.i32", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  testing::SymbolTable regs;

  function_ctor.construct_input_regs(1).Assign({regs.Def("future")});

  auto kernels_ctor = function_ctor.construct_kernels(2);

  {
    // Await
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("await.i32"));
    kernel_ctor.construct_arguments(1).Assign({regs.Use("future")});
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

  return buffer;
}

void AwaitI32(KernelFrame frame) {
  auto& future = frame.arguments()[0].Get<Future>();

  frame.execution_context().Await<int32_t>(future, &frame.results()[0]);
}

TEST(InterpreterTest, Await) {
  auto buffer = CreateAwaitExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("await.i32", &AwaitI32);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();

  Value input(promise.GetFuture());
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  std::move(promise).Set<int32_t>(100);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output.Get<int32_t>(), 100);
}

struct TestPayload {
  TestPayload() = default;
  TestPayload(const TestPayload& other)
      : copy(other.copy + 1), move(other.move) {}
  TestPayload& operator=(const TestPayload& other) {
    copy = other.copy + 1;
    move = other.move;
    return *this;
  }
  TestPayload(TestPayload&& other) : copy(other.copy), move(other.move + 1) {}
  TestPayload& operator=(TestPayload&& other) {
    copy = other.copy;
    move = other.move + 1;
    return *this;
  }

  int copy = 0;
  int move = 0;
};

void AwaitTestPayload(KernelFrame frame) {
  auto& future = frame.arguments()[0].Get<Future>();

  frame.execution_context().Await<TestPayload>(std::move(future),
                                               &frame.results()[0]);
}

TEST(InterpreterTest, AwaitMove) {
  auto buffer = CreateAwaitExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("await.i32", &AwaitTestPayload);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  {
    absl::Notification notification;
    execution_context.set_exit_handler(
        [&notification]() { notification.Notify(); });

    auto promise = Promise::Allocate<TestPayload>();

    Value input(promise.GetFuture());
    Value output;

    std::vector<uint8_t> last_uses = {true};
    execution_context.Call(executable.functions()[0], last_uses,
                           absl::Span<Value>(&input, 1),
                           absl::Span<Value>(&output, 1));
    Execute(execution_context);

    std::move(promise).Set<TestPayload>(TestPayload{});

    notification.WaitForNotification();
    TF_ASSERT_OK(execution_context.status());

    EXPECT_EQ(output.Get<TestPayload>().copy, 0);
    EXPECT_EQ(output.Get<TestPayload>().move, 4);
  }

  {
    absl::Notification notification;
    execution_context.set_exit_handler(
        [&notification]() { notification.Notify(); });

    auto promise = Promise::Allocate<TestPayload>();

    Value input(promise.GetFuture());
    Value output;

    std::vector<uint8_t> last_uses = {true};
    execution_context.Call(executable.functions()[0], last_uses,
                           absl::Span<Value>(&input, 1),
                           absl::Span<Value>(&output, 1));
    std::move(promise).Set<TestPayload>(TestPayload{});

    Execute(execution_context);

    notification.WaitForNotification();
    TF_ASSERT_OK(execution_context.status());

    EXPECT_EQ(output.Get<TestPayload>().copy, 0);
    EXPECT_EQ(output.Get<TestPayload>().move, 4);
  }
}

TEST(InterpreterTest, AwaitError) {
  auto buffer = CreateAwaitExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("await.i32", &AwaitI32);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();

  Value input(promise.GetFuture());
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(executable.functions()[0], last_uses,
                         absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  std::move(promise).SetError(absl::InternalError("test error"));

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

bc::Buffer CreateAwaitAllExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"await_all.i32", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  testing::SymbolTable regs;

  function_ctor.construct_input_regs(2).Assign(
      regs.Def(absl::Span<const std::string>{"f0", "f1"}));

  auto kernels_ctor = function_ctor.construct_kernels(2);

  {
    // await_all.i32
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("await_all.i32"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"f0", "f1"}));
    kernel_ctor.construct_last_uses(2).Assign({true, true});
    kernel_ctor.construct_results(2).Assign(
        regs.Def(absl::Span<const std::string>{"r0", "r1"}));
  }

  {
    // Return
    auto kernel_ctor = kernels_ctor.ConstructAt(1);
    kernel_ctor.set_code(kernels.Use("return"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"r0", "r1"}));
  }

  function_ctor.set_num_regs(regs.size());
  function_ctor.construct_output_regs(2).Assign(regs.Use({"r0", "r1"}));

  return buffer;
}

void AwaitAllI32(KernelFrame frame) {
  RegisterValueSpan<Future> futures(frame.arguments());
  frame.execution_context().AwaitAll<int32_t>(futures, frame.results());
}

TEST(InterpreterTest, AwaitAll) {
  auto buffer = CreateAwaitAllExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("await_all.i32", &AwaitAllI32);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto p0 = Promise::Allocate<int32_t>();
  auto p1 = Promise::Allocate<int32_t>();

  std::vector<Value> inputs(2);
  inputs[0].Set(p0.GetFuture());
  inputs[1].Set(p1.GetFuture());
  std::vector<Value> outputs(2);

  std::vector<uint8_t> last_uses = {true, true};

  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(inputs), absl::MakeSpan(outputs));
  Execute(execution_context);

  std::move(p0).Set<int32_t>(100);
  std::move(p1).Set<int32_t>(200);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(outputs[0].Get<int32_t>(), 100);
  EXPECT_EQ(outputs[1].Get<int32_t>(), 200);
}

void AwaitAllSharedPtrI32(KernelFrame frame) {
  RegisterValueSpan<Future> futures(frame.arguments());
  frame.execution_context().AwaitAll<std::shared_ptr<int32_t>>(futures,
                                                               frame.results());

  for (int i = 0; i < futures.size(); ++i) {
    if (frame.last_uses()[i]) {
      futures.Destroy(i);
    }
  }
}

TEST(InterpreterTest, AwaitAllSingleProducerMultiConsumers) {
  auto buffer = CreateAwaitAllExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("await_all.i32", &AwaitAllSharedPtrI32);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto p = Promise::Allocate<std::shared_ptr<int32_t>>();

  std::vector<Value> inputs(2);
  inputs[0].Set(p.GetFuture());
  inputs[1].Set(p.GetFuture());
  std::vector<Value> outputs(2);

  std::vector<uint8_t> last_uses = {true, true};

  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(inputs), absl::MakeSpan(outputs));
  Execute(execution_context);
  work_queue->AddTask([p = std::move(p)]() mutable {
    std::move(p).Set<std::shared_ptr<int32_t>>(std::make_shared<int32_t>(123));
  });

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(*outputs[0].Get<std::shared_ptr<int32_t>>(), 123);
  EXPECT_EQ(*outputs[1].Get<std::shared_ptr<int32_t>>(), 123);
}

TEST(InterpreterTest, AwaitAllError) {
  auto buffer = CreateAwaitAllExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("await_all.i32", &AwaitAllI32);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto p0 = Promise::Allocate<int32_t>();
  auto p1 = Promise::Allocate<int32_t>();

  std::vector<Value> inputs(2);
  inputs[0].Set(p0.GetFuture());
  inputs[1].Set(p1.GetFuture());
  std::vector<Value> outputs(2);

  std::vector<uint8_t> last_uses = {true, true};

  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(inputs), absl::MakeSpan(outputs));
  Execute(execution_context);

  std::move(p0).Set<int32_t>(100);

  // The execution must be blocked on a tf_mlrt.await_all at this moment,
  // because `p1` has not been fulfilled yet.
  ASSERT_FALSE(notification.HasBeenNotified());

  std::move(p1).SetError(absl::InternalError("test error"));

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

struct TestState : UserContext<TestState> {
  int* state = nullptr;
};

void WriteState(KernelFrame frame) {
  auto& test_state = frame.execution_context().GetUserContext<TestState>();
  CHECK(test_state.state);
  *test_state.state = frame.arguments()[0].Get<int>();
}

void ReadState(KernelFrame frame) {
  auto& test_state = frame.execution_context().GetUserContext<TestState>();
  CHECK(test_state.state);
  frame.results()[0].Set(*test_state.state);
}

bc::Buffer CreateAwaitControlExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.await_control", "return",
                                    "read_state"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  testing::SymbolTable regs;
  function_ctor.construct_input_regs(1).Assign({regs.Def("control_future")});

  auto kernels_ctor = function_ctor.construct_kernels(3);

  {
    // mlrt.await_control
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("mlrt.await_control"));
    kernel_ctor.construct_arguments(1).Assign({regs.Use("control_future")});
  }

  {
    // read_state
    auto kernel_ctor = kernels_ctor.ConstructAt(1);
    kernel_ctor.set_code(kernels.Use("read_state"));
    kernel_ctor.construct_results(1).Assign({regs.Def("result")});
  }

  {
    // Return
    auto kernel_ctor = kernels_ctor.ConstructAt(2);
    kernel_ctor.set_code(kernels.Use("return"));
    kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
  }

  function_ctor.set_num_regs(regs.size());
  function_ctor.construct_output_regs(1).Assign({regs.Use("result")});

  return buffer;
}

TEST(InterpreterTest, AwaitControl) {
  auto buffer = CreateAwaitControlExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("read_state", &ReadState);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  int state = 0;
  auto test_state = std::make_unique<TestState>();
  test_state->state = &state;
  execution_context.AddUserContext(std::move(test_state));

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<Control>();

  Value input(promise.GetFuture());
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  state = 100;
  std::move(promise).Set<Control>();

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());
  ASSERT_TRUE(output.HasValue());
  EXPECT_EQ(output.Get<int>(), 100);
}

bc::Buffer CreateAwaitAllControlExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.await_all_control", "return",
                                    "read_state"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(1);
  auto function_ctor = functions_ctor.ConstructAt(0);
  function_ctor.construct_name("main");

  testing::SymbolTable regs;
  function_ctor.construct_input_regs(2).Assign(
      regs.Def(absl::Span<const std::string>{"f0", "f1"}));

  auto kernels_ctor = function_ctor.construct_kernels(3);

  {
    // mlrt.await_all_control
    auto kernel_ctor = kernels_ctor.ConstructAt(0);
    kernel_ctor.set_code(kernels.Use("mlrt.await_all_control"));
    kernel_ctor.construct_arguments(2).Assign(regs.Use({"f0", "f1"}));
  }

  {
    // read_state
    auto kernel_ctor = kernels_ctor.ConstructAt(1);
    kernel_ctor.set_code(kernels.Use("read_state"));
    kernel_ctor.construct_results(1).Assign({regs.Def("result")});
  }

  {
    // Return
    auto kernel_ctor = kernels_ctor.ConstructAt(2);
    kernel_ctor.set_code(kernels.Use("return"));
    kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
  }

  function_ctor.set_num_regs(regs.size());
  function_ctor.construct_output_regs(1).Assign({regs.Use("result")});

  return buffer;
}

TEST(InterpreterTest, AwaitAllControl) {
  auto buffer = CreateAwaitAllControlExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("read_state", &ReadState);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  int state = 0;
  auto test_state = std::make_unique<TestState>();
  test_state->state = &state;
  execution_context.AddUserContext(std::move(test_state));

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto p0 = Promise::Allocate<Control>();
  auto p1 = Promise::Allocate<Control>();

  std::vector<Value> inputs(2);
  inputs[0].Set(p0.GetFuture());
  inputs[1].Set(p1.GetFuture());
  Value output;

  std::vector<uint8_t> last_uses = {true, true};
  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(inputs), absl::Span<Value>(&output, 1));
  Execute(execution_context);

  state = 100;

  ASSERT_FALSE(notification.HasBeenNotified());

  std::move(p0).Set<Control>();

  ASSERT_FALSE(notification.HasBeenNotified());

  std::move(p1).Set<Control>();

  notification.WaitForNotification();

  TF_ASSERT_OK(execution_context.status());
  ASSERT_TRUE(output.HasValue());
  EXPECT_EQ(output.Get<int>(), 100);
}

TEST(InterpreterTest, AwaitAllControlError) {
  auto buffer = CreateAwaitAllControlExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("read_state", &ReadState);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  int state = 0;
  auto test_state = std::make_unique<TestState>();
  test_state->state = &state;
  execution_context.AddUserContext(std::move(test_state));

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto p0 = Promise::Allocate<Control>();
  auto p1 = Promise::Allocate<Control>();

  std::vector<Value> inputs(2);
  inputs[0].Set(p0.GetFuture());
  inputs[1].Set(p1.GetFuture());
  Value output;

  std::vector<uint8_t> last_uses = {true, true};
  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(inputs), absl::Span<Value>(&output, 1));
  Execute(execution_context);

  state = 100;

  ASSERT_FALSE(notification.HasBeenNotified());

  std::move(p1).Set<Control>();

  ASSERT_FALSE(notification.HasBeenNotified());

  std::move(p0).SetError(absl::InternalError("test error"));

  notification.WaitForNotification();

  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

class AddInPlaceI32 : public KernelFrame {
 public:
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "add_inplace";

  int32_t arg0() const { return arguments()[kArg0Index].Get<int32_t>(); }
  int32_t arg1() const { return arguments()[kArg1Index].Get<int32_t>(); }
  int32_t& arg2() const { return *arguments()[kArg2Index].Get<int32_t*>(); }

  void Invoke() { arg2() = arg0() + arg1(); }

 private:
  static constexpr int kArg0Index = 0;
  static constexpr int kArg1Index = 1;
  static constexpr int kArg2Index = 2;
};

bc::Buffer CreateAsyncExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("func_idx", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.async", "mlrt.await_handle",
                                    "add_inplace", "return"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(regs.Def({"x", "y", "z_ptr"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z_ptr"}));
      kernel_ctor.construct_last_uses(3).Assign({false, false, false});
      kernel_ctor.construct_results(1).Assign({regs.Def("handle")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // mlrt.await_handle
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.await_handle"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("handle")});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("callee");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(regs.Def({"x", "y", "z_ptr"}));

    auto kernels_ctor = function_ctor.construct_kernels(2);
    {
      // add_inplace
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("add_inplace"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z_ptr"}));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, Async) {
  auto buffer = CreateAsyncExecutable();
  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register<AddInPlaceI32>();
  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  int32_t output = 0;
  std::vector<mlrt::Value> args(3);
  args[0].Set<int32_t>(1);
  args[1].Set<int32_t>(2);
  // The output parameter will be moved into execution.
  args[2].Set<int32_t*>(&output);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(loaded_executable.GetFunction("main"), last_uses,
                         absl::MakeSpan(args), absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output, 3);
}

void AddInPlaceI32Error(KernelFrame frame) {
  frame.execution_context().Fail(absl::InternalError("test error"));
}

TEST(InterpreterTest, AsyncError) {
  auto buffer = CreateAsyncExecutable();
  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("add_inplace", &AddInPlaceI32Error);
  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  int32_t output = 0;
  std::vector<mlrt::Value> args(3);
  args[0].Set<int32_t>(1);
  args[1].Set<int32_t>(2);
  args[2].Set<int32_t*>(&output);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(function, last_uses, absl::MakeSpan(args),
                         absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

bc::Buffer CreateNestedAsyncExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(2));

  std::string constant_str(sizeof(uint32_t), '\0');

  attributes.Add("async_callee_index", 1);

  attributes.Add("callee_index", 2);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.async", "mlrt.await_handle",
                                    "add_inplace", "return"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(3);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(regs.Def({"x", "y", "z"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z"}));
      kernel_ctor.construct_last_uses(3).Assign({false, false, false});
      kernel_ctor.construct_results(1).Assign({regs.Def("handle")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("async_callee_index")});
    }

    {
      // mlrt.await_handle
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.await_handle"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("handle")});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("async_callee");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(regs.Def({"x", "y", "z"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z"}));
      kernel_ctor.construct_last_uses(3).Assign({false, false, false});
      kernel_ctor.construct_results(1).Assign({regs.Def("handle")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("callee_index")});
    }

    {
      // mlrt.await_handle
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.await_handle"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("handle")});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(2);
    function_ctor.construct_name("callee");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(regs.Def({"x", "y", "z"}));

    auto kernels_ctor = function_ctor.construct_kernels(2);
    {
      // add_inplace
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("add_inplace"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z"}));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, NestedAsync) {
  auto buffer = CreateNestedAsyncExecutable();
  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register<AddInPlaceI32>();
  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  int32_t output = 0;
  std::vector<mlrt::Value> args(3);
  args[0].Set<int32_t>(1);
  args[1].Set<int32_t>(2);
  args[2].Set<int32_t*>(&output);  // output parameter

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(function, last_uses, absl::MakeSpan(args),
                         absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(output, 3);
}

TEST(InterpreterTest, NestedAsyncError) {
  auto buffer = CreateNestedAsyncExecutable();
  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("add_inplace", &AddInPlaceI32Error);
  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  int32_t output = 0;
  std::vector<mlrt::Value> args(3);
  args[0].Set<int32_t>(1);
  args[1].Set<int32_t>(2);
  args[2].Set<int32_t*>(&output);  // output parameter

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true};
  execution_context.Call(function, last_uses, absl::MakeSpan(args),
                         absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "test error"));
}

bc::Buffer CreateAsyncControlPromiseAwaitExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(2));

  attributes.Add("func_idx", 1);

  attributes.Add("num_futures", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.async",
                                    "mlrt.await_handle",
                                    "mlrt.allocate_control_futures",
                                    "mlrt.await_control",
                                    "mlrt.promise_control",
                                    "return",
                                    "write_state",
                                    "read_state"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    testing::SymbolTable regs;

    function_ctor.construct_input_regs(1).Assign({regs.Def("input")});

    auto kernels_ctor = function_ctor.construct_kernels(6);
    {
      // mlrt.allocate_control_futures
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.allocate_control_futures"));
      kernel_ctor.construct_results(2).Assign(regs.Def(
          absl::Span<const std::string>{"control_promise", "control_future"}));
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("num_futures")});
    }

    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(2).Assign(
          regs.Use({"input", "control_promise"}));
      kernel_ctor.construct_last_uses(2).Assign({false, true});
      kernel_ctor.construct_results(1).Assign({regs.Def("handle")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // mlrt.await_control
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("mlrt.await_control"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("control_future")});
    }

    {
      // read_state
      auto kernel_ctor = kernels_ctor.ConstructAt(3);
      kernel_ctor.set_code(kernels.Use("read_state"));
      kernel_ctor.construct_results(1).Assign({regs.Def("output")});
    }

    {
      // mlrt.await_handle
      auto kernel_ctor = kernels_ctor.ConstructAt(4);
      kernel_ctor.set_code(kernels.Use("mlrt.await_handle"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("handle")});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(5);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("output")});
    }

    function_ctor.set_num_regs(regs.size());
    function_ctor.construct_output_regs(1).Assign({regs.Use("output")});
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("callee");
    testing::SymbolTable regs;

    function_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"input", "control_promise"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);
    {
      // write_state
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("write_state"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("input")});
    }

    {
      // mlrt.promise_control
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.promise_control"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("control_promise")});
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, AsyncControlPromiseAwait) {
  auto buffer = CreateAsyncControlPromiseAwaitExecutable();
  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("read_state", &ReadState);
  kernel_registry.Register("write_state", &WriteState);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  int state = 0;
  auto test_state = std::make_unique<TestState>();
  test_state->state = &state;
  execution_context.AddUserContext(std::move(test_state));

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  Value input(200);
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  ASSERT_TRUE(output.HasValue());
  EXPECT_EQ(output.Get<int>(), 200);
}

bc::Buffer CreateAwaitAllHandleExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("func_idx", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.async", "mlrt.await_all_handle",
                                    "add_inplace", "return"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(4).Assign(
        regs.Def({"x", "y", "z_ptr", "w_ptr"}));

    auto kernels_ctor = function_ctor.construct_kernels(4);
    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z_ptr"}));
      kernel_ctor.construct_last_uses(3).Assign({false, false, false});
      kernel_ctor.construct_results(1).Assign({regs.Def("h0")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "w_ptr"}));
      kernel_ctor.construct_last_uses(3).Assign({false, false, false});
      kernel_ctor.construct_results(1).Assign({regs.Def("h1")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // mlrt.await_all_handle
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("mlrt.await_all_handle"));
      kernel_ctor.construct_arguments(2).Assign(regs.Use({"h0", "h1"}));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(3);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("callee");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(3).Assign(regs.Def({"x", "y", "z_ptr"}));

    auto kernels_ctor = function_ctor.construct_kernels(2);
    {
      // add_inplace
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("add_inplace"));
      kernel_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z_ptr"}));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, AwaitAllhandle) {
  auto buffer = CreateAwaitAllHandleExecutable();
  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register<AddInPlaceI32>();
  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  int32_t z = 0, w = 0;
  std::vector<mlrt::Value> args(4);
  args[0].Set<int32_t>(1);
  args[1].Set<int32_t>(2);
  args[2].Set<int32_t*>(&z);
  args[3].Set<int32_t*>(&w);

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  std::vector<uint8_t> last_uses = {true, true, true, true};
  execution_context.Call(function, last_uses, absl::MakeSpan(args),
                         absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  TF_ASSERT_OK(execution_context.status());

  EXPECT_EQ(z, 3);
  EXPECT_EQ(w, 3);
}

bc::Buffer CreateWhileExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("body_idx", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.while", "return", "test_while_body"};
  executable_ctor.construct_kernel_names(3).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);
  {
    auto caller_ctor = functions_ctor.ConstructAt(0);
    caller_ctor.construct_name("main");

    testing::SymbolTable regs;

    caller_ctor.construct_input_regs(4).Assign(
        regs.Def({"init_bool", "x", "y", "z"}));

    {
      auto kernels_ctor = caller_ctor.construct_kernels(2);
      {
        // mlrt.while
        auto kernel_ctor = kernels_ctor.ConstructAt(0);
        kernel_ctor.set_code(kernels.Use("mlrt.while"));
        kernel_ctor.construct_arguments(4).Assign(
            regs.Use({"init_bool", "x", "y", "z"}));
        kernel_ctor.construct_last_uses(4).Assign({true, true, true, true});
        kernel_ctor.construct_results(4).Assign(regs.Def({"r", "s", "t", "u"}));
        kernel_ctor.construct_attributes(1).Assign(
            {attributes.GetHandle("body_idx")});
      }

      {
        // Return
        auto kernel_ctor = kernels_ctor.ConstructAt(1);
        kernel_ctor.set_code(kernels.Use("return"));
        kernel_ctor.construct_arguments(1).Assign({regs.Use("t")});
      }
    }

    caller_ctor.set_num_regs(regs.size());
    caller_ctor.construct_output_regs(1).Assign({regs.Use("t")});
  }

  {
    auto body_ctor = functions_ctor.ConstructAt(1);
    body_ctor.construct_name("body");

    testing::SymbolTable regs;

    body_ctor.construct_input_regs(3).Assign(
        regs.Def(absl::Span<const std::string>{"x", "y", "z"}));

    {
      auto kernels_ctor = body_ctor.construct_kernels(2);
      auto predicate_ctor = kernels_ctor.ConstructAt(0);
      predicate_ctor.set_code(kernels.Use("test_while_body"));
      predicate_ctor.construct_arguments(3).Assign(regs.Use({"x", "y", "z"}));
      predicate_ctor.construct_results(4).Assign(
          regs.Def({"u", "v", "w", "p"}));

      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(4).Assign(regs.Use({"u", "v", "w", "p"}));
    }

    body_ctor.set_num_regs(regs.size());
    body_ctor.construct_output_regs(4).Assign(regs.Use({"u", "v", "w", "p"}));
  }

  return buffer;
}

// A test while loop body.
// Pseudo code
// out[0] = in[0] + 1 --> Loop count increment
// out[1] = in[1]     --> Loop count stop value
// out[2] = in[2] + 2 (Step)  --> Increment value
// out[3] = in[0] < in[1]
constexpr int32_t kValueIncrementStep = 2;
void TestWhileBody(KernelFrame frame) {
  ASSERT_EQ(frame.arguments().size(), 3);
  ASSERT_EQ(frame.results().size(), 4);
  frame.results()[0].Set(frame.arguments()[0].Get<int32_t>() + 1);
  frame.results()[1].Set(frame.arguments()[1].Get<int32_t>());
  frame.results()[2].Set(frame.arguments()[2].Get<int32_t>() +
                         kValueIncrementStep);
  frame.results()[3].Set(frame.arguments()[0].Get<int32_t>() <
                         frame.arguments()[1].Get<int32_t>());
}

TEST(KernelTest, While) {
  auto buffer = CreateWhileExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry registry;
  RegisterBuiltinKernels(registry);
  registry.Register("test_while_body", &TestWhileBody);
  LoadedExecutable loaded_executable(executable, registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  Value inputs[4];

  constexpr int32_t kStart = 0;
  constexpr int32_t kEnd = 2;
  constexpr int32_t kInitialValue = 6;

  inputs[0].Set(true);
  inputs[1].Set(kStart);
  inputs[2].Set(kEnd);
  inputs[3].Set(kInitialValue);
  Value output;

  std::vector<uint8_t> last_uses = {false, false, false, false};
  execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                         absl::Span<Value>(&output, 1));

  Execute(execution_context);

  ASSERT_TRUE(output.HasValue());
  EXPECT_EQ(kInitialValue + kValueIncrementStep * (kEnd - kStart + 1),
            output.Get<int32_t>());
}

TEST(KernelTest, WhileWithInitialFalseCondition) {
  auto buffer = CreateWhileExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry registry;
  RegisterBuiltinKernels(registry);
  registry.Register("test_while_body", &TestWhileBody);
  LoadedExecutable loaded_executable(executable, registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  Value inputs[4];

  constexpr int32_t kStart = 0;
  constexpr int32_t kEnd = 2;
  constexpr int32_t kInitialValue = 6;

  inputs[0].Set(false);
  inputs[1].Set(kStart);
  inputs[2].Set(kEnd);
  inputs[3].Set(kInitialValue);
  Value output;

  std::vector<uint8_t> last_uses = {false, false, false, false};
  execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                         absl::Span<Value>(&output, 1));

  Execute(execution_context);

  ASSERT_TRUE(output.HasValue());
  EXPECT_EQ(kInitialValue, output.Get<int32_t>());

  // Should have no side effect on input.
  EXPECT_EQ(inputs[0].Get<bool>(), false);
  EXPECT_EQ(inputs[1].Get<int32_t>(), kStart);
  EXPECT_EQ(inputs[2].Get<int32_t>(), kEnd);
  EXPECT_EQ(inputs[3].Get<int32_t>(), kInitialValue);
}

bc::Buffer CreateUnwindExecutable(int num_regs = 1) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  auto kernel_names_ctor = executable_ctor.construct_kernel_names(2);
  kernel_names_ctor.ConstructAt(0, "cancel");
  kernel_names_ctor.ConstructAt(1, "return");

  auto functions_ctor = executable_ctor.construct_functions(1);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");
    function_ctor.set_num_regs(num_regs);

    std::vector<uint32_t> reg_indices(num_regs);
    std::iota(reg_indices.begin(), reg_indices.end(), 0);

    function_ctor.construct_input_regs(num_regs).Assign(reg_indices.begin(),
                                                        reg_indices.end());
    function_ctor.construct_output_regs(num_regs).Assign(reg_indices.begin(),
                                                         reg_indices.end());

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      // cancel
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(0);
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(1);
      kernel_ctor.construct_arguments(num_regs).Assign(reg_indices.begin(),
                                                       reg_indices.end());
    }
  }

  return buffer;
}

void Cancel(KernelFrame frame) {
  frame.execution_context().Fail(absl::CancelledError("test cancel"));
}

TEST(InterpreterTest, UnwindPromise) {
  auto buffer = CreateUnwindExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  ExecutionContext execution_context(&loaded_executable);

  auto promise = Promise::Allocate<int32_t>();
  auto future = promise.GetFuture();

  Value input(std::move(promise));
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
}

TEST(InterpreterTest, UnwindInvalidPromise) {
  auto buffer = CreateUnwindExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  ExecutionContext execution_context(&loaded_executable);

  auto promise = Promise::Allocate<int32_t>();
  auto future = promise.GetFuture();
  std::move(promise).Set<int32_t>(100);

  Value input(std::move(promise));  // NOLINT(bugprone-use-after-move)
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
  EXPECT_EQ(future.Get<int32_t>(), 100);
}

TEST(InterpreterTest, UnwindFuture) {
  auto buffer = CreateUnwindExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();

  Value input(promise.GetFuture());
  Value output;

  std::vector<uint8_t> last_uses = {false};

  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  std::move(promise).Set<int32_t>(100);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
  EXPECT_EQ(input.Get<Future>().Get<int32_t>(), 100);
}

TEST(InterpreterTest, UnwindPromiseAndFuture) {
  auto buffer = CreateUnwindExecutable(/*num_regs=*/2);

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();
  auto future = promise.GetFuture();

  // If both the promise and the future are going to be unwinded before the
  // promise is set or passed to another thread, the promise must be unwinded
  // first before the future to avoid deadlock.
  std::vector<Value> inputs(2);
  inputs[0].Set(std::move(promise));
  inputs[1].Set(future);

  std::vector<Value> outputs(2);

  std::vector<uint8_t> last_uses = {true, true};
  execution_context.Call(function, last_uses, absl::Span<Value>(inputs),
                         absl::Span<Value>(outputs));
  Execute(execution_context);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
  EXPECT_THAT(
      future.GetError(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
}

TEST(InterpreterTest, UnwindAsyncHandle) {
  auto buffer = CreateUnwindExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto [promise, handle] = AsyncHandle::Allocate(execution_context);

  Value input(std::move(handle));
  Value output;

  std::vector<uint8_t> last_uses = {true};
  execution_context.Call(function, last_uses, absl::Span<Value>(&input, 1),
                         absl::Span<Value>(&output, 1));
  Execute(execution_context);

  ASSERT_FALSE(notification.HasBeenNotified());

  std::move(promise).Finish(absl::OkStatus());

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
}

bc::Buffer CreateCaseExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  {
    bc::Buffer attr_buffer;
    bc::Allocator attr_allocator(&attr_buffer);
    bc::New<bc::Vector<int32_t>>(&attr_allocator, std::vector<int32_t>{1, 2});
    attributes.Add("function_indices",
                   absl::string_view(attr_buffer.data(), attr_buffer.size()));
  }

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.case", "return"};
  executable_ctor.construct_kernel_names(2).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(3);
  {
    auto caller_ctor = functions_ctor.ConstructAt(0);
    caller_ctor.construct_name("main");

    testing::SymbolTable regs;

    caller_ctor.construct_input_regs(3).Assign(
        regs.Def({"branch_idx", "in_0", "in_1"}));

    {
      auto kernels_ctor = caller_ctor.construct_kernels(2);
      {
        // mlrt.case
        auto kernel_ctor = kernels_ctor.ConstructAt(0);
        kernel_ctor.set_code(kernels.Use("mlrt.case"));
        kernel_ctor.construct_arguments(3).Assign(
            regs.Use({"branch_idx", "in_0", "in_1"}));
        kernel_ctor.construct_last_uses(3).Assign({true, true, true});
        kernel_ctor.construct_results(1).Assign({regs.Def("result")});
        kernel_ctor.construct_attributes(1).Assign(
            {attributes.GetHandle("function_indices")});
      }

      {
        // Return
        auto kernel_ctor = kernels_ctor.ConstructAt(1);
        kernel_ctor.set_code(kernels.Use("return"));
        kernel_ctor.construct_arguments(1).Assign({regs.Use("result")});
      }
    }

    caller_ctor.set_num_regs(regs.size());
    caller_ctor.construct_output_regs(1).Assign({regs.Use("result")});
  }

  {
    auto callee_ctor = functions_ctor.ConstructAt(1);
    callee_ctor.construct_name("callee0");

    testing::SymbolTable regs;

    callee_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"x", "y"}));

    {
      auto kernels_ctor = callee_ctor.construct_kernels(1);
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign(regs.Use({"x"}));
    }

    callee_ctor.set_num_regs(regs.size());
    callee_ctor.construct_output_regs(1).Assign(regs.Use({"x"}));
  }

  {
    auto callee_ctor = functions_ctor.ConstructAt(2);
    callee_ctor.construct_name("calle1");

    testing::SymbolTable regs;

    callee_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"x", "y"}));

    {
      auto kernels_ctor = callee_ctor.construct_kernels(1);
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("return"));
      kernel_ctor.construct_arguments(1).Assign(regs.Use({"y"}));
    }

    callee_ctor.set_num_regs(regs.size());
    callee_ctor.construct_output_regs(1).Assign(regs.Use({"y"}));
  }

  return buffer;
}

bc::Buffer CreateUnwindComplexExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("func_idx", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"mlrt.async", "mlrt.await_control",
                                    "cancel", "return"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(2).Assign(
        regs.Def(absl::Span<const std::string>{"p", "f"}));

    auto kernels_ctor = function_ctor.construct_kernels(3);

    {
      // async
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.async"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("f")});
      kernel_ctor.construct_last_uses(1).Assign({true});
      kernel_ctor.construct_results(1).Assign({regs.Def("handle")});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // cancel
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("cancel"));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(2);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("callee");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(1).Assign({regs.Def("f")});

    auto kernels_ctor = function_ctor.construct_kernels(2);
    {
      // mlrt.await_control
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("mlrt.await_control"));
      kernel_ctor.construct_arguments(1).Assign(regs.Use({"f"}));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, UnwindComplex) {
  // Test unwinding a function that launches an async function and uses
  // promise/future to communicate with the async function. Interpreter should
  // handle error correctly in this situation by setting error in promises and
  // wait until the async function to finish.

  auto buffer = CreateUnwindComplexExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();
  auto future = promise.GetFuture();

  std::vector<Value> inputs(2);
  inputs[0].Set(std::move(promise));
  inputs[1].Set(future);

  std::vector<uint8_t> last_uses = {true, true};

  execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                         absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
  EXPECT_THAT(
      future.GetError(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
}

bc::Buffer CreateUnwindNestedExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("func_idx", 1);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"call", "cancel", "return"};
  executable_ctor.construct_kernel_names(names.size()).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(3);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("main");

    testing::SymbolTable regs;
    function_ctor.construct_input_regs(1).Assign({regs.Def("p")});

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      // Call
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("call"));
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(1);
    function_ctor.construct_name("callee");

    testing::SymbolTable regs;

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      // cancel
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("cancel"));
    }

    {
      // return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(InterpreterTest, UnwindNested) {
  auto buffer = CreateUnwindNestedExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("cancel", &Cancel);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);
  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  execution_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();
  auto future = promise.GetFuture();

  Value input(std::move(promise));
  Value output;

  execution_context.CallByMove(function, absl::Span<Value>(&input, 1),
                               absl::Span<Value>());
  Execute(execution_context);

  notification.WaitForNotification();
  EXPECT_THAT(
      execution_context.status(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
  EXPECT_THAT(
      future.GetError(),
      absl_testing::StatusIs(absl::StatusCode::kCancelled, "test cancel"));
}

TEST(KernelTest, Case) {
  auto buffer = CreateCaseExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry registry;
  RegisterBuiltinKernels(registry);
  LoadedExecutable loaded_executable(executable, registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  Value inputs[3];

  constexpr int32_t kBranch0In = 123;
  constexpr int32_t kBranch1In = 456;

  // Test Branch 0
  {
    inputs[0].Set<uint32_t>(0);
    inputs[1].Set(kBranch0In);
    inputs[2].Set(kBranch1In);
    Value output;

    std::vector<uint8_t> last_uses = {true, true, true};
    execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                           absl::Span<Value>(&output, 1));

    Execute(execution_context);

    ASSERT_TRUE(output.HasValue());
    EXPECT_EQ(kBranch0In, output.Get<int32_t>());
  }
  {
    // Test Branch 1
    inputs[0].Set<uint32_t>(1);
    inputs[1].Set(kBranch0In);
    inputs[2].Set(kBranch1In);
    Value output;

    std::vector<uint8_t> last_uses = {true, true, true};
    execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                           absl::Span<Value>(&output, 1));

    Execute(execution_context);

    ASSERT_TRUE(output.HasValue());
    EXPECT_EQ(kBranch1In, output.Get<int32_t>());
  }
}

TEST(KernelTest, CaseInvalidBranchIndexShallChooseLastBranch) {
  auto buffer = CreateCaseExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry registry;
  RegisterBuiltinKernels(registry);
  LoadedExecutable loaded_executable(executable, registry);

  ExecutionContext execution_context(&loaded_executable);

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  Value inputs[3];

  constexpr int32_t kBranch0In = 123;
  constexpr int32_t kBranch1In = 456;

  // Test Invalid Branch 10
  {
    inputs[0].Set<uint32_t>(10);
    inputs[1].Set(kBranch0In);
    inputs[2].Set(kBranch1In);
    Value output;

    std::vector<uint8_t> last_uses = {true, true, true};
    execution_context.Call(function, last_uses, absl::MakeSpan(inputs),
                           absl::Span<Value>(&output, 1));

    Execute(execution_context);

    ASSERT_TRUE(output.HasValue());
    EXPECT_EQ(kBranch1In, output.Get<int32_t>());
  }
}

struct TestPromiseReturnOp : PromiseReturnOpBase<TestPromiseReturnOp> {
  using PromiseReturnOpBase::PromiseReturnOpBase;

  static constexpr char kName[] = "test_promise_return";

  Promise& promise() const { return arguments()[0].Get<Promise>(); }
  int32_t value() const { return arguments()[1].Get<int32_t>(); }
  bool value_last_use() const { return last_uses()[1]; }
};

bc::Buffer CreatePromiseReturnExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"await.i32", "test_promise_return",
                                    "return"};
  executable_ctor.construct_kernel_names(3).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(2);
  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("consumer");

    testing::SymbolTable regs;

    function_ctor.construct_input_regs(1).Assign({regs.Def("future")});

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      // Await
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("await.i32"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("future")});
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

    testing::SymbolTable regs;

    function_ctor.construct_input_regs(2).Assign(
        {regs.Def("promise"), regs.Def("value")});

    auto kernels_ctor = function_ctor.construct_kernels(1);

    {
      // test_promise_return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("test_promise_return"));
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

  bc::Executable executable(buffer.data());

  KernelRegistry registry;
  RegisterBuiltinKernels(registry);
  registry.Register<TestPromiseReturnOp>();
  registry.Register("await.i32", &AwaitI32);

  LoadedExecutable loaded_executable(executable, registry);
  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/1, /*num_blocking_threads=*/1);
  ExecutionContext consumer_context(&loaded_executable);
  consumer_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  consumer_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();

  Value output;
  {
    Value input(promise.GetFuture());

    std::vector<uint8_t> last_uses = {true};
    consumer_context.Call(loaded_executable.GetFunction("consumer"), last_uses,
                          absl::Span<Value>(&input, 1),
                          absl::Span<Value>(&output, 1));
    Execute(consumer_context);
  }

  {
    Value inputs[2];
    inputs[0].Set(std::move(promise));
    inputs[1].Set(100);

    ExecutionContext producer_context(&loaded_executable);
    std::vector<uint8_t> last_uses = {true, true};
    producer_context.Call(loaded_executable.GetFunction("producer"), last_uses,
                          absl::Span<Value>(inputs), absl::Span<Value>());
    Execute(producer_context);
  }

  notification.WaitForNotification();
  EXPECT_EQ(output.Get<int32_t>(), 100);
}

bc::Buffer CreatePromiseReturnNotScheduleImmediatelyExecutable() {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  testing::AttributeTable attributes(executable_ctor.construct_attributes(1));

  attributes.Add("func_idx", 2);

  testing::SymbolTable kernels;
  std::vector<std::string> names = {"await.i32", "call", "test_promise_return",
                                    "return"};
  executable_ctor.construct_kernel_names(4).Assign(names);
  kernels.Def(names);

  auto functions_ctor = executable_ctor.construct_functions(3);

  {
    auto function_ctor = functions_ctor.ConstructAt(0);
    function_ctor.construct_name("consumer");

    testing::SymbolTable regs;

    function_ctor.construct_input_regs(1).Assign({regs.Def("future")});

    auto kernels_ctor = function_ctor.construct_kernels(2);

    {
      // Await
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("await.i32"));
      kernel_ctor.construct_arguments(1).Assign({regs.Use("future")});
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
    testing::SymbolTable regs;

    auto caller_ctor = functions_ctor.ConstructAt(1);
    caller_ctor.construct_name("producer");
    caller_ctor.construct_input_regs(2).Assign(
        {regs.Def("promise"), regs.Def("value")});

    auto kernels_ctor = caller_ctor.construct_kernels(2);
    {
      // Call
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("call"));
      kernel_ctor.construct_arguments(2).Assign(
          {regs.Use("promise"), regs.Use("value")});
      kernel_ctor.construct_last_uses(2).Assign({true, true});
      kernel_ctor.construct_attributes(1).Assign(
          {attributes.GetHandle("func_idx")});
    }

    {
      // Return
      auto kernel_ctor = kernels_ctor.ConstructAt(1);
      kernel_ctor.set_code(kernels.Use("return"));
    }

    caller_ctor.set_num_regs(regs.size());
  }

  {
    auto function_ctor = functions_ctor.ConstructAt(2);
    function_ctor.construct_name("producer_callee");

    testing::SymbolTable regs;

    function_ctor.construct_input_regs(2).Assign(
        {regs.Def("promise"), regs.Def("value")});

    auto kernels_ctor = function_ctor.construct_kernels(1);

    {
      // test_promise_return
      auto kernel_ctor = kernels_ctor.ConstructAt(0);
      kernel_ctor.set_code(kernels.Use("test_promise_return"));
      kernel_ctor.construct_arguments(2).Assign(
          {regs.Use("promise"), regs.Use("value")});
      kernel_ctor.construct_last_uses(2).Assign({true, true});
    }

    function_ctor.set_num_regs(regs.size());
  }

  return buffer;
}

TEST(KernelTest, PromiseReturnNotScheduleImmediately) {
  auto buffer = CreatePromiseReturnNotScheduleImmediatelyExecutable();

  bc::Executable executable(buffer.data());

  KernelRegistry registry;
  RegisterBuiltinKernels(registry);
  registry.Register<TestPromiseReturnOp>();
  registry.Register("await.i32", &AwaitI32);

  LoadedExecutable loaded_executable(executable, registry);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  ExecutionContext consumer_context(&loaded_executable);
  consumer_context.set_work_queue(work_queue.get());

  absl::Notification notification;
  consumer_context.set_exit_handler(
      [&notification]() { notification.Notify(); });

  auto promise = Promise::Allocate<int32_t>();

  Value output;
  {
    Value input(promise.GetFuture());

    std::vector<uint8_t> last_uses = {true};
    consumer_context.Call(loaded_executable.GetFunction("consumer"), last_uses,
                          absl::Span<Value>(&input, 1),
                          absl::Span<Value>(&output, 1));
    Execute(consumer_context);
  }

  {
    Value inputs[2];
    inputs[0].Set(std::move(promise));
    inputs[1].Set(100);

    ExecutionContext producer_context(&loaded_executable);
    consumer_context.set_work_queue(work_queue.get());

    std::vector<uint8_t> last_uses = {true, true};
    producer_context.Call(loaded_executable.GetFunction("producer"), last_uses,
                          absl::Span<Value>(inputs), absl::Span<Value>());
    Execute(producer_context);
  }

  notification.WaitForNotification();
  EXPECT_EQ(output.Get<int32_t>(), 100);
}

void BM_SequentialAdd(::testing::benchmark::State& state) {
  auto buffer = CreateSequentialAddExecutable(99);

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register<AddI32Kernel>();

  LoadedExecutable loaded_executable(executable, kernel_registry);

  absl::Notification notification;

  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_exit_handler([&]() { notification.Notify(); });

  int32_t v = 1;
  Value arg(v);
  Value result;

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  std::vector<uint8_t> last_uses = {false};
  execution_context.Call(function, last_uses, absl::Span<Value>(&arg, 1),
                         absl::Span<Value>(&result, 1));

  Execute(execution_context);
  notification.WaitForNotification();
  CHECK_EQ(result.Get<int32_t>(), 100);

  for (auto s : state) {
    absl::Notification notification;

    ExecutionContext execution_context(&loaded_executable);
    execution_context.set_exit_handler([&]() { notification.Notify(); });

    execution_context.Call(function, last_uses, absl::Span<Value>(&arg, 1),
                           absl::Span<Value>(&result, 1));
    Execute(execution_context);
    notification.WaitForNotification();
  }
}
BENCHMARK(BM_SequentialAdd);

void BM_SequentialAddAttributes(::testing::benchmark::State& state) {
  auto buffer = CreateSequentialAddAttributesExecutable(99);

  bc::Executable executable(buffer.data());

  KernelRegistry kernel_registry;
  RegisterBuiltinKernels(kernel_registry);
  kernel_registry.Register("add.const", &AddI32Const);

  LoadedExecutable loaded_executable(executable, kernel_registry);

  absl::Notification notification;

  ExecutionContext execution_context(&loaded_executable);
  execution_context.set_exit_handler([&]() { notification.Notify(); });

  int32_t v = 1;
  mlrt::Value arg(v);
  mlrt::Value result;

  auto function = loaded_executable.GetFunction("main");
  ASSERT_TRUE(function);

  std::vector<uint8_t> last_uses = {false};
  execution_context.Call(function, last_uses, absl::Span<Value>(&arg, 1),
                         absl::Span<Value>(&result, 1));
  Execute(execution_context);
  notification.WaitForNotification();
  CHECK_EQ(result.Get<int32_t>(), 100);

  for (auto s : state) {
    absl::Notification notification;

    ExecutionContext execution_context(&loaded_executable);
    execution_context.set_exit_handler([&]() { notification.Notify(); });

    std::vector<uint8_t> last_uses = {false};
    execution_context.Call(function, last_uses, absl::Span<Value>(&arg, 1),
                           absl::Span<Value>(&result, 1));
    Execute(execution_context);
    notification.WaitForNotification();
  }
}
BENCHMARK(BM_SequentialAddAttributes);

}  // namespace
}  // namespace mlrt
