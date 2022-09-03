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

#include "tensorflow/compiler/xla/runtime/custom_call.h"

#include <string_view>
#include <utility>

#include "tensorflow/compiler/xla/mlir/transforms/runtime/compilation_pipeline.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/diagnostics.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// A helper function that compiles `module` to XLA runtime executable and runs
// `test` function with the given arguments. Caller can also register custom
// calls (direct or dynamic) and custom types.
//===----------------------------------------------------------------------===//

static absl::StatusOr<JitExecutable> Compile(
    std::string_view module, DirectCustomCallRegistry direct_custom_calls,
    TypeIDNameRegistry::RegistrationFn types) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding = ToSymbolsBinding(direct_custom_calls, types);

  opts.compiler.register_dialects = [&](mlir::DialectRegistry& registry) {
    RegisterDefaultXlaRuntimeDialects(registry);
  };

  opts.compiler.create_compilation_pipeline = [&](mlir::PassManager& pm) {
    CompilationPipelineOptions copts;
    CreateDefaultXlaRuntimeCompilationPipeline(pm, copts);
  };

  return JitExecutable::Instantiate(module, "test", opts);
}

static absl::Status CompileAndExecute(
    std::string_view module, ArgumentsRef args,
    DynamicCustomCallRegistry dynamic_custom_calls = {},
    DirectCustomCallRegistry direct_custom_calls = {},
    TypeIDNameRegistry::RegistrationFn types = {}) {
  absl::StatusOr<JitExecutable> jit_executable =
      Compile(module, direct_custom_calls, types);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError())
    return absl::InternalError(executable.GetError().message);

  // Prepare the call frame outside of a benchmark loop.
  Executable::CallFrame call_frame;
  auto initialized = executable->InitializeCallFrame(args, &call_frame);
  if (!initialized.ok()) return initialized;

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &dynamic_custom_calls;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  executable->Execute(call_frame, execute_opts);
  if (call_frame.is_error) return absl::InternalError(call_frame.error);

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

// Static counter to observe side effects of direct custom call.
static int32_t custom_call_counter = 0;

// Direct custom call linked with XLA runtime executable at compile (link) time.
static bool CustomCallFn(ExecutionContext* ctx, void** args, void** attrs,
                         void** rets) {
  auto handler = CustomCall::Bind("test.custom_call")
                     .Arg<int32_t>()
                     .To([&](int32_t arg) -> LogicalResult {
                       custom_call_counter += arg;
                       return success();
                     });

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

TEST(CustomCallTest, DirectCustomCall) {
  absl::string_view module = R"(
    func.func private @custom_call(%arg0: i32)
      attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i32
      call @custom_call(%0) : (i32) -> ()
      return
    }
  )";

  DirectCustomCallRegistry registry;
  registry.Register("test.custom_call", CustomCallFn);

  ASSERT_EQ(custom_call_counter, 0);
  ASSERT_TRUE(CompileAndExecute(module, {}, {}, std::move(registry)).ok());
  EXPECT_EQ(custom_call_counter, 42);
}

TEST(CustomCallTest, ScalarArgs) {
  absl::string_view module = R"(
    func.func private @custom_call(%arg0: i1, %arg1: i32, %arg2: i64,
                                   %arg3: f32, %arg4: f64)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant false
      %1 = arith.constant 42 : i32
      %2 = arith.constant 42 : i64
      %3 = arith.constant 42.0 : f32
      %4 = arith.constant 42.0 : f64
      call @custom_call(%0, %1, %2, %3, %4) : (i1, i32, i64, f32, f64) -> ()
      return
    }
  )";

  bool i1 = true;
  int32_t i32 = 0;
  int64_t i64 = 0;
  float f32 = 0.0;
  double f64 = 0.0;

  auto f = [&](bool arg0, int32_t arg1, int64_t arg2, float arg3, double arg4) {
    (i1 = arg0, i32 = arg1, i64 = arg2, f32 = arg3, f64 = arg4);
    return success();
  };

  DynamicCustomCallRegistry registry;
  registry.Register(CustomCall::Bind("test.custom_call")
                        .Arg<bool>()
                        .Arg<int32_t>()
                        .Arg<int64_t>()
                        .Arg<float>()
                        .Arg<double>()
                        .To(f));

  ASSERT_TRUE(CompileAndExecute(module, /*args=*/{}, std::move(registry)).ok());

  EXPECT_EQ(i1, false);
  EXPECT_EQ(i32, 42);
  EXPECT_EQ(i64, 42);
  EXPECT_EQ(f32, 42.0);
  EXPECT_EQ(f64, 42.0);
}

TEST(CustomCallTest, ScalarRets) {
  absl::string_view module = R"(
    func.func private @custom_call_result() -> (i1, i32, i64, f32, f64)
      attributes { rt.custom_call = "test.custom_call_result" }

    func.func private @custom_call(%arg0: i1, %arg1: i32, %arg2: i64,
                                   %arg3: f32, %arg4: f64)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0, %1, %2, %3, %4 = call @custom_call_result()
        : () -> (i1, i32, i64, f32, f64)
      call @custom_call(%0, %1, %2, %3, %4) : (i1, i32, i64, f32, f64) -> ()
      return
    }
  )";

  bool i1 = true;
  int32_t i32 = 0;
  int64_t i64 = 0;
  float f32 = 0.0;
  double f64 = 0.0;

  auto f_result = [&](Result<bool> ret0, Result<int32_t> ret1,
                      Result<int64_t> ret2, Result<float> ret3,
                      Result<double> ret4) {
    ret0.Set(false);
    ret1.Set(42);
    ret2.Set(42);
    ret3.Set(42.0);
    ret4.Set(42.0);
    return success();
  };

  auto f = [&](bool arg0, int32_t arg1, int64_t arg2, float arg3, double arg4) {
    (i1 = arg0, i32 = arg1, i64 = arg2, f32 = arg3, f64 = arg4);
    return success();
  };

  DynamicCustomCallRegistry registry;
  registry.Register(CustomCall::Bind("test.custom_call_result")
                        .Ret<bool>()
                        .Ret<int32_t>()
                        .Ret<int64_t>()
                        .Ret<float>()
                        .Ret<double>()
                        .To(f_result));

  registry.Register(CustomCall::Bind("test.custom_call")
                        .Arg<bool>()
                        .Arg<int32_t>()
                        .Arg<int64_t>()
                        .Arg<float>()
                        .Arg<double>()
                        .To(f));

  ASSERT_TRUE(CompileAndExecute(module, /*args=*/{}, std::move(registry)).ok());

  EXPECT_EQ(i1, false);
  EXPECT_EQ(i32, 42);
  EXPECT_EQ(i64, 42);
  EXPECT_EQ(f32, 42.0);
  EXPECT_EQ(f64, 42.0);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

using DirectCustomCall = DirectCustomCallRegistry::DirectCustomCall;
using RuntimeChecks = CustomCall::RuntimeChecks;

// Give short aliases to enums for benchmarks pretty printing.
static constexpr RuntimeChecks all = RuntimeChecks::kDefault;
static constexpr RuntimeChecks types = RuntimeChecks::kTypes;
static constexpr RuntimeChecks none = RuntimeChecks::kNone;

static void BenchmarkCustomCall(benchmark::State& state,
                                std::string_view module, ArgumentsRef args,
                                std::string_view name,
                                DirectCustomCall custom_call,
                                TypeIDNameRegistry::RegistrationFn types = {}) {
  // Wrap benchmarked custom call into a direct custom call registry.
  DirectCustomCallRegistry custom_calls;
  custom_calls.Register(name, custom_call);

  absl::StatusOr<JitExecutable> jit_executable =
      Compile(module, custom_calls, types);
  CHECK(jit_executable.ok()) << jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  CHECK(!executable.IsError()) << executable.GetError().message;

  // Prepare the call frame outside of a benchmark loop.
  Executable::CallFrame call_frame;
  CHECK(executable->InitializeCallFrame(args, &call_frame).ok());

  Executable::ExecuteOpts execute_opts;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  DiagnosticEngine diagnostic_engine;
  execute_opts.diagnostic_engine = &diagnostic_engine;

  for (auto _ : state) {
    call_frame.args[0] = nullptr;  // reset execution context
    executable->Execute(call_frame, execute_opts);
    CHECK(!call_frame.is_error) << call_frame.error;
  }
}

//===----------------------------------------------------------------------===//
// Custom call with a single i32 argument.
//===----------------------------------------------------------------------===//

template <RuntimeChecks checks>
static bool I32X1(ExecutionContext* ctx, void** args, void** attrs,
                  void** rets) {
  static auto* handler = CustomCall::Bind("test.custom_call")
                             .Arg<int32_t>()
                             .To<checks>([](int32_t arg0) { return success(); })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void I32X1(benchmark::State& state) {
  absl::string_view module = R"(
    func.func private @custom_call(%arg0: i32)
      attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 0 : i32
      call @custom_call(%0) : (i32) -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, module, {}, "test.custom_call", &I32X1<checks>);
}

static void BM_I32X1All(benchmark::State& state) { I32X1<all>(state); }
static void BM_I32X1Types(benchmark::State& state) { I32X1<types>(state); }
static void BM_I32X1None(benchmark::State& state) { I32X1<none>(state); }

BENCHMARK(BM_I32X1All);
BENCHMARK(BM_I32X1Types);
BENCHMARK(BM_I32X1None);

}  // namespace runtime
}  // namespace xla
