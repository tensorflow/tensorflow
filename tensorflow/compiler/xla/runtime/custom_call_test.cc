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

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/mlir/ir/runtime/tests/testlib.h"
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

struct TestOpts {
  DynamicCustomCallRegistry dynamic_custom_calls;
  DirectCustomCallRegistry direct_custom_calls;
  DiagnosticEngine diagnostic_engine;

  // Register type id names fro non-canonical types.
  TypeIDNameRegistry::RegistrationFn types;

  // Encoding for non-canonical custom call operands.
  std::function<void(CustomCallArgEncodingSet&)> populate_arg_encodings;
  std::function<void(CustomCallAttrEncodingSet&)> populate_attr_encodings;
};

static absl::StatusOr<JitExecutable> Compile(std::string_view module,
                                             const TestOpts& test_opts) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding =
      ToSymbolsBinding(test_opts.direct_custom_calls, test_opts.types);

  opts.compiler.register_dialects = [&](mlir::DialectRegistry& registry) {
    registry.insert<TestlibDialect>();
    RegisterDefaultXlaRuntimeDialects(registry);
  };

  opts.compiler.create_compilation_pipeline = [&](mlir::PassManager& pm) {
    CompilationPipelineOptions copts;
    copts.populate_type_id_names = test_opts.types;
    copts.populate_arg_encodings = test_opts.populate_arg_encodings;
    copts.populate_attr_encodings = test_opts.populate_attr_encodings;
    CreateDefaultXlaRuntimeCompilationPipeline(pm, copts);
  };

  return JitExecutable::Instantiate(module, "test", opts);
}

static absl::Status CompileAndExecute(std::string_view module,
                                      ArgumentsRef args,
                                      const TestOpts& test_opts) {
  absl::StatusOr<JitExecutable> jit_executable = Compile(module, test_opts);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError())
    return absl::InternalError(executable.GetError().message());

  // Prepare the call frame outside of a benchmark loop.
  Executable::CallFrame call_frame;
  auto initialized = executable->InitializeCallFrame(args, &call_frame);
  if (!initialized.ok()) return initialized;

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &test_opts.dynamic_custom_calls;
  execute_opts.diagnostic_engine = &test_opts.diagnostic_engine;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  executable->Execute(call_frame, execute_opts);
  if (call_frame.is_error) return absl::InternalError(call_frame.error);

  return absl::OkStatus();
}

// Diagnostic engine that appends all emitted diagnostic to the `error` string.
static DiagnosticEngine CollectDiagnostic(std::string* error) {
  DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([=](Diagnostic& diagnostic) -> LogicalResult {
    error->append(diagnostic.status().message());
    return success();
  });
  return diagnostic_engine;
}

// No-Op custom call with a single `i32` argument.
static std::unique_ptr<CustomCall> I32NoOp() {
  return CustomCall::Bind("test.custom_call").Arg<int32_t>().To([](int32_t) {
    return success();
  });
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

  TestOpts opts;
  opts.direct_custom_calls.Register("test.custom_call", CustomCallFn);

  ASSERT_EQ(custom_call_counter, 0);
  ASSERT_TRUE(CompileAndExecute(module, {}, opts).ok());
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

  TestOpts opts;
  opts.dynamic_custom_calls.Register(CustomCall::Bind("test.custom_call")
                                         .Arg<bool>()
                                         .Arg<int32_t>()
                                         .Arg<int64_t>()
                                         .Arg<float>()
                                         .Arg<double>()
                                         .To(f));

  ASSERT_TRUE(CompileAndExecute(module, /*args=*/{}, opts).ok());

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

  TestOpts opts;
  opts.dynamic_custom_calls.Register(CustomCall::Bind("test.custom_call_result")
                                         .Ret<bool>()
                                         .Ret<int32_t>()
                                         .Ret<int64_t>()
                                         .Ret<float>()
                                         .Ret<double>()
                                         .To(f_result));

  opts.dynamic_custom_calls.Register(CustomCall::Bind("test.custom_call")
                                         .Arg<bool>()
                                         .Arg<int32_t>()
                                         .Arg<int64_t>()
                                         .Arg<float>()
                                         .Arg<double>()
                                         .To(f));

  ASSERT_TRUE(CompileAndExecute(module, /*args=*/{}, opts).ok());

  EXPECT_EQ(i1, false);
  EXPECT_EQ(i32, 42);
  EXPECT_EQ(i64, 42);
  EXPECT_EQ(f32, 42.0);
  EXPECT_EQ(f64, 42.0);
}

TEST(CustomCallTest, ArgSizeCheck) {
  // Try to pass two argument to a custom call that expects one.
  absl::string_view module = R"(
    func.func private @custom_call(%arg0: i32, %arg1: i32)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i32
      call @custom_call(%0, %0) : (i32, i32) -> ()
      return
    }
  )";

  std::string error = "";

  TestOpts opts;
  opts.dynamic_custom_calls.Register(I32NoOp());
  opts.diagnostic_engine = CollectDiagnostic(&error);

  auto status = CompileAndExecute(module, /*args=*/{}, opts);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "custom call 'test.custom_call' failed");
  EXPECT_EQ(error, "Wrong number of arguments: expected 1 got 2");
}

TEST(CustomCallTest, ArgTypeCheck) {
  // Try to pass `i64` argument to a custom call that expects `i32`.
  absl::string_view module = R"(
    func.func private @custom_call(%arg1: i64)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i64
      call @custom_call(%0) : (i64) -> ()
      return
    }
  )";

  std::string error = "";

  TestOpts opts;
  opts.dynamic_custom_calls.Register(I32NoOp());
  opts.diagnostic_engine = CollectDiagnostic(&error);

  auto status = CompileAndExecute(module, /*args=*/{}, opts);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "custom call 'test.custom_call' failed");
  EXPECT_EQ(error, "Failed to decode all custom call operands");
}

// Register custom call attribute decoding for `testlib.enum_type`.
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(EnumType);

TEST(CustomCallTest, EnumAttr) {
  absl::string_view module = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { enum = #testlib.enum_type<Baz> }: () -> ()
      return
    }
  )";

  std::vector<EnumType> enums;

  auto type_id_name = [](TypeIDNameRegistry& registry) {
    registry.Register<Tagged<EnumType>>("__type_id_testlib_enum");
  };

  auto encode = [](CustomCallAttrEncodingSet& encoding) {
    encoding.Add<EnumAttrEncoding<EnumTypeAttr, EnumType>>();
  };

  auto handler = [&](EnumType value) -> LogicalResult {
    enums.push_back(value);
    return success();
  };

  TestOpts opts;
  opts.types = type_id_name;
  opts.populate_attr_encodings = encode;
  opts.dynamic_custom_calls.Register(
      CustomCall::Bind("test.custom_call").Attr<EnumType>("enum").To(handler));

  EXPECT_TRUE(CompileAndExecute(module, /*args=*/{}, opts).ok());
  ASSERT_EQ(enums.size(), 1);
  EXPECT_EQ(enums.front(), EnumType::Baz);
}

// Map enum defined by MLIR to a custom enum class.
enum class MyEnumType : uint32_t { kFoo, kBar, kBaz };

MyEnumType FromEnumType(EnumType value) {
  switch (value) {
    case EnumType::Foo:
      return MyEnumType::kFoo;
    case EnumType::Bar:
      return MyEnumType::kBar;
    case EnumType::Baz:
      return MyEnumType::kBaz;
  }
}

XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(MyEnumType);

TEST(CustomCallTest, MappedEnumAttr) {
  absl::string_view module = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { enum = #testlib.enum_type<Baz> }: () -> ()
      return
    }
  )";

  std::vector<MyEnumType> enums;

  auto type_id_name = [](TypeIDNameRegistry& registry) {
    registry.Register<Tagged<MyEnumType>>("__type_id_my_enum");
  };

  auto encode = [](CustomCallAttrEncodingSet& encoding) {
    encoding.Add<EnumAttrEncoding<EnumTypeAttr, EnumType, MyEnumType>>(
        FromEnumType);
  };

  auto handler = [&](MyEnumType value) -> LogicalResult {
    enums.push_back(value);
    return success();
  };

  TestOpts opts;
  opts.types = type_id_name;
  opts.populate_attr_encodings = encode;
  opts.dynamic_custom_calls.Register(CustomCall::Bind("test.custom_call")
                                         .Attr<MyEnumType>("enum")
                                         .To(handler));

  EXPECT_TRUE(CompileAndExecute(module, /*args=*/{}, opts).ok());
  ASSERT_EQ(enums.size(), 1);
  EXPECT_EQ(enums.front(), MyEnumType::kBaz);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

using benchmark::State;

using DirectCustomCall = DirectCustomCallRegistry::DirectCustomCall;
using RuntimeChecks = CustomCall::RuntimeChecks;

// Give short aliases to enums for benchmarks pretty printing.
static constexpr RuntimeChecks all = RuntimeChecks::kDefault;
static constexpr RuntimeChecks types = RuntimeChecks::kTypes;
static constexpr RuntimeChecks none = RuntimeChecks::kNone;

static void BenchmarkCustomCall(State& state, std::string_view module,
                                ArgumentsRef args, std::string_view name,
                                DirectCustomCall custom_call,
                                TypeIDNameRegistry::RegistrationFn types = {}) {
  // Wrap benchmarked custom call into a direct custom call registry.
  TestOpts opts;
  opts.direct_custom_calls.Register(name, custom_call);
  opts.types = std::move(types);

  absl::StatusOr<JitExecutable> jit_executable = Compile(module, opts);
  CHECK(jit_executable.ok()) << jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  CHECK(!executable.IsError()) << executable.GetError().message();

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
                             .To<checks>([](int32_t arg0) {
                               benchmark::DoNotOptimize(arg0);
                               return success();
                             })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void I32X1(State& state) {
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

static void BM_I32X1All(State& s) { I32X1<all>(s); }
static void BM_I32X1None(State& s) { I32X1<none>(s); }

BENCHMARK(BM_I32X1All);
BENCHMARK(BM_I32X1None);

//===----------------------------------------------------------------------===//
// Custom call with twelve i32 argument.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks>
static bool I32X12(ExecutionContext* ctx, void** args, void** attrs,
                   void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .To<checks>([](int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3,
                         int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg7,
                         int32_t arg8, int32_t arg9, int32_t arg10,
                         int32_t arg11) {
            benchmark::DoNotOptimize(arg0 + arg1 + arg2 + arg3 + arg4 + arg5 +
                                     arg6 + arg7 + arg8 + arg9 + arg10 + arg11);
            return success();
          })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks>
static void I32X12(State& state) {
  absl::string_view module = R"(
    func.func private @custom_call(%arg0: i32, %arg1: i32, %arg2: i32,
                                   %arg3: i32, %arg4: i32, %arg5: i32,
                                   %arg6: i32, %arg7: i32, %arg8: i32,
                                   %arg9: i32, %arg10: i32, %arg11: i32)
      attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2 : i32
      %3 = arith.constant 3 : i32
      %4 = arith.constant 4 : i32
      %5 = arith.constant 5 : i32
      %6 = arith.constant 6 : i32
      %7 = arith.constant 7 : i32
      %8 = arith.constant 8 : i32
      %9 = arith.constant 9 : i32
      %10 = arith.constant 10 : i32
      %11 = arith.constant 11 : i32
      call @custom_call(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11)
        : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.return
    }
  )";

  BenchmarkCustomCall(state, module, {}, "test.custom_call", &I32X12<checks>);
}

static void BM_I32X12All(State& s) { I32X12<all>(s); }
static void BM_I32X12None(State& s) { I32X12<none>(s); }

BENCHMARK(BM_I32X12All);
BENCHMARK(BM_I32X12None);

//===----------------------------------------------------------------------===//
// Custom call with a single memref argument.
//===----------------------------------------------------------------------===//

using Flat = FlatMemrefView;
using Strided = StridedMemrefView;

template <CustomCall::RuntimeChecks checks, typename MemrefType>
static bool MemrefX1(ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("test.custom_call")
                             .Arg<MemrefType>()
                             .template To<checks>([](MemrefType arg0) {
                               benchmark::DoNotOptimize(arg0);
                               return success();
                             })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks, typename MemrefType>
static void MemrefX1(State& state) {
  absl::string_view module = R"(
    func.func private @custom_call(%arg0: memref<4x4xf32>)
      attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      %0 = memref.alloca() : memref<4x4xf32>
      call @custom_call(%0) : (memref<4x4xf32>) -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, module, {}, "test.custom_call",
                      &MemrefX1<checks, MemrefType>);
}

static void BM_FlatMemrefX1All(State& s) { MemrefX1<all, Flat>(s); }
static void BM_FlatMemrefX1None(State& s) { MemrefX1<none, Flat>(s); }
static void BM_MemrefX1All(State& s) { MemrefX1<all, MemrefView>(s); }
static void BM_MemrefX1None(State& s) { MemrefX1<none, MemrefView>(s); }
static void BM_StridedMemrefX1All(State& s) { MemrefX1<all, Strided>(s); }
static void BM_StridedMemrefX1None(State& s) { MemrefX1<none, Strided>(s); }

BENCHMARK(BM_FlatMemrefX1All);
BENCHMARK(BM_FlatMemrefX1None);

BENCHMARK(BM_MemrefX1All);
BENCHMARK(BM_MemrefX1None);

BENCHMARK(BM_StridedMemrefX1All);
BENCHMARK(BM_StridedMemrefX1None);

//===----------------------------------------------------------------------===//
// Custom call with twelve memref argument.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks, typename MemrefType>
static bool MemrefX12(ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template To<checks>(
              [](MemrefType arg0, MemrefType arg1, MemrefType arg2,
                 MemrefType arg3, MemrefType arg4, MemrefType arg5,
                 MemrefType arg6, MemrefType arg7, MemrefType arg8,
                 MemrefType arg9, MemrefType arg10, MemrefType arg11) {
                benchmark::DoNotOptimize(arg0);
                benchmark::DoNotOptimize(arg1);
                benchmark::DoNotOptimize(arg2);
                benchmark::DoNotOptimize(arg3);
                benchmark::DoNotOptimize(arg4);
                benchmark::DoNotOptimize(arg5);
                benchmark::DoNotOptimize(arg6);
                benchmark::DoNotOptimize(arg7);
                benchmark::DoNotOptimize(arg8);
                benchmark::DoNotOptimize(arg9);
                benchmark::DoNotOptimize(arg10);
                benchmark::DoNotOptimize(arg11);
                return success();
              })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks, typename MemrefType>
static void MemrefX12(State& state) {
  absl::string_view module = R"(
    func.func private @custom_call(
      %arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>,
      %arg3: memref<4x4xf32>, %arg4: memref<4x4xf32>, %arg5: memref<4x4xf32>,
      %arg6: memref<4x4xf32>, %arg7: memref<4x4xf32>, %arg8: memref<4x4xf32>,
      %arg9: memref<4x4xf32>, %arg10: memref<4x4xf32>, %arg11: memref<4x4xf32>
    ) attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      %0 = memref.alloca() : memref<4x4xf32>
      call @custom_call(%0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0)
        : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>,
           memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>,
           memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
          ) -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, module, {}, "test.custom_call",
                      &MemrefX12<checks, MemrefType>);
}

static void BM_FlatMemrefX12All(State& s) { MemrefX12<all, Flat>(s); }
static void BM_FlatMemrefX12None(State& s) { MemrefX12<none, Flat>(s); }
static void BM_MemrefX12All(State& s) { MemrefX12<all, MemrefView>(s); }
static void BM_MemrefX12None(State& s) { MemrefX12<none, MemrefView>(s); }
static void BM_StridedMemrefX12All(State& s) { MemrefX12<all, Strided>(s); }
static void BM_StridedMemrefX12None(State& s) { MemrefX12<none, Strided>(s); }

BENCHMARK(BM_FlatMemrefX12All);
BENCHMARK(BM_FlatMemrefX12None);

BENCHMARK(BM_MemrefX12All);
BENCHMARK(BM_MemrefX12None);

BENCHMARK(BM_StridedMemrefX12All);
BENCHMARK(BM_StridedMemrefX12None);

//===----------------------------------------------------------------------===//
// Custom call with a single i32 attribute.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks>
static bool I32AttrX1(ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler = CustomCall::Bind("test.custom_call")
                             .Attr<int32_t>("attr0")
                             .To<checks>([](int32_t attr0) {
                               benchmark::DoNotOptimize(attr0);
                               return success();
                             })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks>
static void I32AttrX1(State& state) {
  absl::string_view module = R"(
    func.func private @custom_call()
      attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { attr0 = 42 : i32 }: () -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, module, {}, "test.custom_call",
                      &I32AttrX1<checks>);
}

static void BM_I32AttrX1All(State& s) { I32AttrX1<all>(s); }
static void BM_I32AttrX1None(State& s) { I32AttrX1<none>(s); }
static void BM_I32AttrX1Types(State& s) { I32AttrX1<types>(s); }

BENCHMARK(BM_I32AttrX1All);
BENCHMARK(BM_I32AttrX1Types);
BENCHMARK(BM_I32AttrX1None);

//===----------------------------------------------------------------------===//
// Custom call with twelve i32 attributes.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks>
static bool I32AttrX12(ExecutionContext* ctx, void** args, void** attrs,
                       void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .Attr<int32_t>("attr0")
          .Attr<int32_t>("attr1")
          .Attr<int32_t>("attr2")
          .Attr<int32_t>("attr3")
          .Attr<int32_t>("attr4")
          .Attr<int32_t>("attr5")
          .Attr<int32_t>("attr6")
          .Attr<int32_t>("attr7")
          .Attr<int32_t>("attr8")
          .Attr<int32_t>("attr9")
          .Attr<int32_t>("attr10")
          .Attr<int32_t>("attr11")
          .To<checks>([](int32_t attr0, int32_t attr1, int32_t attr2,
                         int32_t attr3, int32_t attr4, int32_t attr5,
                         int32_t attr6, int32_t attr7, int32_t attr8,
                         int32_t attr9, int32_t attr10, int32_t attr11) {
            benchmark::DoNotOptimize(attr0 + attr1 + attr2 + attr3 + attr4 +
                                     attr5 + attr6 + attr7 + attr8 + attr9 +
                                     attr10 + attr11);
            return success();
          })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks>
static void I32AttrX12(State& state) {
  absl::string_view module = R"(
    func.func private @custom_call()
      attributes { rt.direct_custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call()
       { "attr0" = 0 : i32, "attr1" = 1 : i32, "attr2" = 2 : i32,
         "attr3" = 3 : i32, "attr4" = 4 : i32, "attr5" = 5 : i32,
         "attr6" = 6 : i32, "attr7" = 7 : i32, "attr8" = 8 : i32,
         "attr9" = 9 : i32, "attr10" = 10 : i32, "attr11" = 11 : i32
       } : () -> ()
      func.return
    }
  )";

  BenchmarkCustomCall(state, module, {}, "test.custom_call",
                      &I32AttrX12<checks>);
}

static void BM_I32AttrX12All(State& s) { I32AttrX12<all>(s); }
static void BM_I32AttrX12None(State& s) { I32AttrX12<none>(s); }
static void BM_I32AttrX12Types(State& s) { I32AttrX12<types>(s); }

BENCHMARK(BM_I32AttrX12All);
BENCHMARK(BM_I32AttrX12Types);
BENCHMARK(BM_I32AttrX12None);

}  // namespace runtime
}  // namespace xla
