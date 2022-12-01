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
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/transforms/tests/testlib_pipeline.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/async_runtime_api.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/results.h"
#include "tensorflow/compiler/xla/runtime/types.h"
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

static AsyncTaskRunner* NoRunner() {
  return reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);
}

static absl::StatusOr<JitExecutable> Compile(
    std::string_view module, absl::Span<const std::string_view> exported) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.register_dialects = RegisterXlaRuntimeTestlibDialects;
  opts.compiler.create_compilation_pipeline = CreateXlaRuntimeTestlibPipeline;

  return JitExecutable::Instantiate(module, opts, exported);
}

static absl::Status Execute(JitExecutable& jit_executable, unsigned ordinal,
                            ArgumentsRef args, ResultConverter& results,
                            AsyncTaskRunner* async_task_runner = NoRunner()) {
  AsyncValuePtr<Executable> executable = jit_executable.DefaultExecutable();
  if (executable.IsError()) return executable.GetError();

  Executable::ExecuteOpts execute_opts;
  execute_opts.async_task_runner = async_task_runner;

  FunctionRef function_ref = executable->function_ref(ordinal);
  return function_ref(args, results, execute_opts);
}

static absl::Status CompileAndExecute(
    std::string_view module, ArgumentsRef args, ResultConverter& results,
    AsyncTaskRunner* async_task_runner = NoRunner()) {
  StatusOr<JitExecutable> jit_executable = Compile(module, {"test"});
  if (!jit_executable.ok()) return jit_executable.status();

  return Execute(*jit_executable, 0, args, results, async_task_runner);
}

//===----------------------------------------------------------------------===//

static void AssertNoError(const absl::Status& status) {
  assert(false && "Unexpected call to `ReturnError`");
}

static void IgnoreError(const absl::Status& status) {}

void Emplace(void* int_ptr, AsyncValue* dst) {
  auto& v = dst->get<int32_t>();
  v = *reinterpret_cast<int32_t*>(int_ptr);
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

struct ReturnMemref {
  LogicalResult operator()(unsigned result_index, const Type* type,
                           const Type* runtime_type, void* ret) const {
    auto* memref = llvm::dyn_cast<MemrefType>(runtime_type);
    if (!memref) return failure();

    auto desc = ConvertReturnedMemref<MemrefDesc>(*this, memref, ret);
    if (failed(desc)) return failure();

    *ptr = std::move(*desc);
    return success();
  }

  MemrefDesc operator()(PrimitiveType element_type, void* base_ptr,
                        void* data_ptr, int64_t offset,
                        absl::Span<const int64_t> sizes,
                        absl::Span<const int64_t> strides) const {
    return MemrefDesc(element_type, base_ptr, offset, sizes, strides);
  }

  std::optional<MemrefDesc>* ptr = nullptr;
};

struct ReturnAsyncToken {
  LogicalResult operator()(unsigned result_index, const Type* type,
                           const Type* runtime_type, void* result_ptr) const {
    if (!llvm::isa<AsyncTokenType>(type)) return failure();

    // Load the pointer to the async token from a pointer to result storage.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
    void* ret = *reinterpret_cast<void**>(result_ptr);
    auto* token = static_cast<mlir::runtime::AsyncToken*>(ret);
    auto* async_value = AsyncRuntime::GetAsyncValue(token);
    CHECK(async_value->IsAvailable());
    chain.SetStateConcrete();
    AsyncRuntime::DropRef(AsyncRuntime::ToAsyncRuntimeObject(token));
    return success();
  }

  AsyncValuePtr<Chain> chain;
};

struct ReturnAsyncI32 {
  LogicalResult operator()(unsigned result_index, const Type* type,
                           const Type* runtime_type, void* result_ptr) const {
    auto* value_type = llvm::dyn_cast<AsyncValueType>(type);
    if (!value_type) return mlir::failure();

    // Load the pointer to the async value from a pointer to result storage.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
    void* ret = *reinterpret_cast<void**>(result_ptr);
    auto* value = static_cast<mlir::runtime::AsyncValue*>(ret);
    auto* scalar = llvm::dyn_cast<ScalarType>(&value_type->value_type());
    if (scalar && scalar->type() == PrimitiveType::S32) {
      ExtractAsyncValue(value, ptr.value(), Emplace);
      return success();
    }
    return failure();
  }

  AsyncValuePtr<int32_t> ptr;
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

TEST(ExecutableTest, ReturnMemref) {
  absl::string_view module = R"(
    func.func @test() -> memref<?x?xf32> {
      %0 = arith.constant 1 : index
      %1 = arith.constant 2 : index
      %2 = memref.alloc(%0, %1) : memref<?x?xf32>
      return %2 : memref<?x?xf32>
    }
  )";

  std::optional<MemrefDesc> result;
  ResultConverterSet converter(AssertNoError, ReturnMemref{&result});

  ASSERT_TRUE(CompileAndExecute(module, {}, converter).ok());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rank(), 2);
  EXPECT_EQ(result->size(0), 1);
  EXPECT_EQ(result->size(1), 2);

  // Result converter passed onwership of the underlying buffer to MemrefDesc.
  std::free(result->data());
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

TEST(ExecutableTest, MultipleFunctions) {
  absl::string_view module = R"(
    func.func @add(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }

    func.func @mul(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.muli %arg0, %arg1 : i32
      return %0 : i32
    }
  )";

  absl::StatusOr<JitExecutable> compiled = Compile(module, {"add", "mul"});
  ASSERT_TRUE(compiled.ok());
  EXPECT_EQ(compiled->num_functions(), 2);

  int32_t result = 0;
  ResultConverterSet converter(AssertNoError, ReturnI32{&result});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  ASSERT_TRUE(Execute(*compiled, /*ordinal=*/0, {arg0, arg1}, converter).ok());
  EXPECT_EQ(result, 20 + 22);

  ASSERT_TRUE(Execute(*compiled, /*ordinal=*/1, {arg0, arg1}, converter).ok());
  EXPECT_EQ(result, 20 * 22);
}

TEST(ExecutableTest, AssertionFailure) {
  absl::string_view module = R"(
    func.func @test(%arg0: i32) {
      %c42 = arith.constant 42 : i32
      %0 = arith.cmpi ne, %c42, %arg0 : i32
      cf.assert %0, "Oops, argument can't be 42"
      return
    }
  )";

  NoResultConverter converter;

  {
    ScalarArg arg0(int32_t{20});
    EXPECT_TRUE(CompileAndExecute(module, {arg0}, converter).ok());
  }

  {
    ScalarArg arg0(int32_t{42});
    auto executed = CompileAndExecute(module, {arg0}, converter);
    EXPECT_FALSE(executed.ok());
    EXPECT_EQ(executed.message(), "run time error: Oops, argument can't be 42");
  }
}

TEST(ExecutableTest, AssertionFailureOrResult) {
  absl::string_view module = R"(
    func.func @test(%arg0: i32) -> i32 {
      %c42 = arith.constant 42 : i32
      %0 = arith.cmpi ne, %c42, %arg0 : i32
      cf.assert %0, "Oops, argument can't be 42"
      %1 = arith.addi %arg0, %c42 : i32
      return %1 : i32
    }
  )";

  {
    int32_t result = 0;
    ResultConverterSet converter(AssertNoError, ReturnI32{&result});

    ScalarArg arg0(int32_t{20});
    EXPECT_TRUE(CompileAndExecute(module, {arg0}, converter).ok());
    EXPECT_EQ(result, 62);
  }

  {
    int32_t result = 0;
    ResultConverterSet converter(IgnoreError, ReturnI32{&result});

    ScalarArg arg0(int32_t{42});
    auto executed = CompileAndExecute(module, {arg0}, converter);
    EXPECT_FALSE(executed.ok());
    EXPECT_EQ(executed.message(), "run time error: Oops, argument can't be 42");
    EXPECT_EQ(result, 0);
  }
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

TEST(ExecutableTest, AsyncTokenRet) {
  absl::string_view module = R"(
    async.func @test() -> !async.token {
      return
    }
  )";

  AsyncValueRef<Chain> result = MakeConstructedAsyncValueRef<Chain>();
  ResultConverterSet converter(AssertNoError, ReturnAsyncToken{result.AsPtr()});

  ASSERT_TRUE(CompileAndExecute(module, {}, converter).ok());
  EXPECT_EQ(result.IsAvailable(), true);
}

TEST(ExecutableTest, AsyncScalarRet) {
  absl::string_view module = R"(
    async.func @test(%arg0: i32, %arg1: i32) -> !async.value<i32> {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
  )";

  AsyncValueRef<int32_t> result = MakeConstructedAsyncValueRef<int32_t>();
  ResultConverterSet converter(AssertNoError, ReturnAsyncI32{result.AsPtr()});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  ASSERT_TRUE(CompileAndExecute(module, {arg0, arg1}, converter).ok());
  EXPECT_EQ(result.get(), 42);
}

TEST(ExecutableTest, AsyncWaiting) {
  absl::string_view module = R"(
    async.func @test2(%arg0: i32, %arg1: i32) -> !async.value<i32> {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
    async.func @test(%arg0: i32, %arg1:i32) -> !async.value<i32> {
      %0 = async.call @test2(%arg0, %arg1) : (i32, i32) -> !async.value<i32>
      %1 = async.await %0 : !async.value<i32>
      return %1 : i32
    }
  )";

  AsyncValueRef<int32_t> result = MakeConstructedAsyncValueRef<int32_t>();
  ResultConverterSet converter(AssertNoError, ReturnAsyncI32{result.AsPtr()});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  ASSERT_TRUE(CompileAndExecute(module, {arg0, arg1}, converter).ok());
  EXPECT_EQ(result.get(), 42);
}
//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

static void CompileAndBenchmark(
    benchmark::State& state, std::string_view module, ArgumentsRef args,
    ResultConverter& results, AsyncTaskRunner* async_task_runner = NoRunner()) {
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

void BM_AsyncExecuteAndAwait(benchmark::State& state) {
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

void BM_AsyncFunc(benchmark::State& state) {
  absl::string_view module = R"(
    async.func @test(%arg0: i32, %arg1: i32) -> !async.value<i32> {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
  )";

  AsyncValueRef<int32_t> result = MakeConstructedAsyncValueRef<int32_t>();
  ResultConverterSet converter(AssertNoError, ReturnAsyncI32{result.AsPtr()});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  InlineAsyncTaskRunner runner;
  CompileAndBenchmark(state, module, {arg0, arg1}, converter, &runner);
}

void BM_AsyncFuncCall(benchmark::State& state) {
  absl::string_view module = R"(
    async.func @test2(%arg0: i32, %arg1: i32) -> !async.value<i32> {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
    async.func @test(%arg0: i32, %arg1:i32) -> !async.value<i32> {
      %0 = async.call @test2(%arg0, %arg1) : (i32, i32) -> !async.value<i32>
      %1 = async.await %0 : !async.value<i32>
      return %1 : i32
    }
  )";

  AsyncValueRef<int32_t> result = MakeConstructedAsyncValueRef<int32_t>();
  ResultConverterSet converter(AssertNoError, ReturnAsyncI32{result.AsPtr()});

  ScalarArg arg0(static_cast<int32_t>(20));
  ScalarArg arg1(static_cast<int32_t>(22));

  InlineAsyncTaskRunner runner;
  CompileAndBenchmark(state, module, {arg0, arg1}, converter, &runner);
}

BENCHMARK(BM_AsyncExecuteAndAwait);
BENCHMARK(BM_AsyncFunc);
BENCHMARK(BM_AsyncFuncCall);

}  // namespace runtime
}  // namespace xla
