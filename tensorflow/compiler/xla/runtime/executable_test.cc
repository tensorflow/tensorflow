/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/tests/testlib_pipeline.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/results.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

using absl::StatusOr;

//===----------------------------------------------------------------------===//
// A helper function that compiles the given `module` to an XLA runtime
// executable and runs the module's `test` function with the given arguments.
// Results are returned to the caller via the user-provided result converter.
//===----------------------------------------------------------------------===//

static AsyncTaskRunner* NoAsyncTaskRunner() {
  return reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);
}

static absl::Status CompileAndExecute(
    std::string_view module, ArgumentsRef args, ResultConverter& results,
    AsyncTaskRunner* async_task_runner = NoAsyncTaskRunner()) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.register_dialects = RegisterXlaRuntimeTestlibDialects;
  opts.compiler.create_compilation_pipeline = CreateXlaRuntimeTestlibPipeline;

  StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(module, "test", opts);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError()) return executable.GetError();

  Executable::CallFrame call_frame;
  auto initialized = executable->InitializeCallFrame(args, &call_frame);
  if (!initialized.ok()) return initialized;

  Executable::ExecuteOpts execute_opts;
  execute_opts.async_task_runner = async_task_runner;

  executable->Execute(call_frame, execute_opts);
  if (call_frame.is_error) return absl::InternalError(call_frame.error);

  return executable->ReturnResults(results, &call_frame);
}

//===----------------------------------------------------------------------===//

static void AssertNoError(const absl::Status& status) {
  assert(false && "Unexpected call to `ReturnError`");
}

struct ReturnI32 {
  LogicalResult operator()(unsigned result_index, const Type* type,
                           const Type* runtime_type, void* ret) const {
    auto* scalar = llvm::dyn_cast<ScalarType>(type);
    if (scalar && scalar->type() == PrimitiveType::S32) {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(ret, sizeof(int32_t));
      *ptr = *reinterpret_cast<int32_t*>(ret);
      return success();
    }
    return failure();
  }

  int32_t* ptr = nullptr;
};

// Execute all tasks in the caller thread immediately.
class InlineAsyncTaskRunner : public AsyncTaskRunner {
 public:
  void Schedule(Task task) final { (task(), num_executed_++); }
  size_t num_executed() const { return num_executed_; }

 private:
  size_t num_executed_ = 0;
};

//===----------------------------------------------------------------------===//

TEST(ExecutableTest, ReturnScalar) {
  absl::string_view module = R"(
    func.func @test() -> i32 {
      %0 = arith.constant 42 : i32
      return %0 : i32
    }
  )";

  int32_t result = 0;
  ResultConverterSet converter(AssertNoError, ReturnI32{&result});

  ASSERT_TRUE(CompileAndExecute(module, {}, converter).ok());
  EXPECT_EQ(result, 42);
}

TEST(ExecutableTest, ScalarArgs) {
  absl::string_view module = R"(
    func.func @test(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
  )";

  int32_t result = 0;
  ResultConverterSet converter(AssertNoError, ReturnI32{&result});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  ASSERT_TRUE(CompileAndExecute(module, {arg0, arg1}, converter).ok());
  EXPECT_EQ(result, 42);
}

TEST(ExecutableTest, AsyncExecuteAndAwait) {
  absl::string_view module = R"(
    func.func @test(%arg0: i32, %arg1: i32) -> i32 {
      %token, %result = async.execute -> !async.value<i32> {
        %0 = arith.addi %arg0, %arg1 : i32
        async.yield %0 : i32
      }
      %1 = async.await %result : !async.value<i32>
      return %1 : i32
    }
  )";

  int32_t result = 0;
  ResultConverterSet converter(AssertNoError, ReturnI32{&result});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  InlineAsyncTaskRunner runner;

  ASSERT_TRUE(CompileAndExecute(module, {arg0, arg1}, converter, &runner).ok());
  EXPECT_EQ(runner.num_executed(), 1);
  EXPECT_EQ(result, 42);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

using benchmark::State;

static void CompileAndBenchmark(
    State& state, std::string_view module, ArgumentsRef args,
    ResultConverter& results,
    AsyncTaskRunner* async_task_runner = NoAsyncTaskRunner()) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.register_dialects = RegisterXlaRuntimeTestlibDialects;
  opts.compiler.create_compilation_pipeline = CreateXlaRuntimeTestlibPipeline;

  StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(module, "test", opts);
  CHECK(jit_executable.ok()) << jit_executable.status().message();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  CHECK(!executable.IsError()) << executable.GetError().message();

  Executable::CallFrame call_frame;
  auto initialized = executable->InitializeCallFrame(args, &call_frame);
  CHECK(initialized.ok()) << initialized.message();

  Executable::ExecuteOpts execute_opts;
  execute_opts.async_task_runner = async_task_runner;

  for (auto _ : state) {
    call_frame.args[0] = nullptr;  // reset execution context
    executable->Execute(call_frame, execute_opts);
    CHECK(!call_frame.is_error) << call_frame.error;
    absl::Status returned = executable->ReturnResults(results, &call_frame);
    CHECK(returned.ok()) << returned.message();
  }
}

void BM_AsyncExecuteAndAwait(State& state) {
  absl::string_view module = R"(
    func.func @test(%arg0: i32, %arg1: i32) -> i32 {
      %token, %result = async.execute -> !async.value<i32> {
        %0 = arith.addi %arg0, %arg1 : i32
        async.yield %0 : i32
      }
      %1 = async.await %result : !async.value<i32>
      return %1 : i32
    }
  )";

  int32_t result = 0;
  ResultConverterSet converter(AssertNoError, ReturnI32{&result});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  InlineAsyncTaskRunner runner;
  CompileAndBenchmark(state, module, {arg0, arg1}, converter, &runner);
}

BENCHMARK(BM_AsyncExecuteAndAwait);

}  // namespace runtime
}  // namespace xla
