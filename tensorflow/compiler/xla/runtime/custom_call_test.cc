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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/diagnostics.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/module.h"
#include "tensorflow/compiler/xla/runtime/state.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

using absl::StatusOr;

//===----------------------------------------------------------------------===//
// A helper function that compiles `module` to XLA runtime executable and runs
// `test` function with the given arguments. Caller can also register custom
// calls (direct or dynamic) and custom types.
//===----------------------------------------------------------------------===//

struct CustomCallRegistry {
  std::function<void(DynamicCustomCallRegistry&)> dynamic_custom_calls;
  std::function<void(DirectCustomCallRegistry&)> direct_custom_calls;
};

static absl::StatusOr<JitExecutable> Compile(
    std::string_view source, const CustomCallRegistry& registry,
    const CompilationPipelineOptions& copts,
    const TypeConverter& type_converter = {},
    absl::Span<const std::string_view> exported = {"test"}) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding = ToSymbolsBinding(
      registry.direct_custom_calls, copts.populate_type_id_names);
  opts.compiler.type_converter = type_converter;

  opts.compiler.register_dialects = [&](DialectRegistry& dialects) {
    RegisterTestlibDialect(dialects);
    RegisterDefaultXlaGpuRuntimeDialects(dialects);
  };

  opts.compiler.create_compilation_pipeline = [=](PassManager& passes) {
    CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
  };

  return JitExecutable::Instantiate(source, opts, exported);
}

static absl::Status CompileAndExecute(
    std::string_view source, ArgumentsRef args,
    const CustomCallRegistry& registry,
    const CompilationPipelineOptions& copts = {},
    const TypeConverter& type_converter = {},
    absl::Span<const std::string_view> exported = {"test"},
    CustomCall::UserData user_data = {}) {
  StatusOr<JitExecutable> jit_executable =
      Compile(source, registry, copts, type_converter, exported);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError())
    return absl::InternalError(executable.GetError().message());

  // Register all dynamic custom calls.
  DynamicCustomCallRegistry dynamic_custom_calls;
  if (registry.dynamic_custom_calls)
    registry.dynamic_custom_calls(dynamic_custom_calls);

  // Always add a pointer to `self` to user data.
  user_data.insert(&executable.get());

  // Collect all emitted diangostics to a string;
  std::string error;
  DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([&](Diagnostic& diagnostic) -> LogicalResult {
    error.append(diagnostic.status().message());
    return success();
  });

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &dynamic_custom_calls;
  execute_opts.diagnostic_engine = &diagnostic_engine;
  execute_opts.custom_call_data = &user_data;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  // We do not support returning results from tests.
  NoResultConverter converter;

  auto executed = executable->Execute(args, converter, execute_opts);
  if (!executed.ok())
    return absl::InternalError(
        absl::StrFormat("%s: %s", executed.message(), error));

  return absl::OkStatus();
}

template <typename State>
static absl::StatusOr<std::unique_ptr<State>> CompileAndExecute(
    std::string_view source, ArgumentsRef args, const StatefulModule<State>& m,
    absl::Span<const std::string_view> exported = {"test"}) {
  CustomCallRegistry registry = {
      [&](DynamicCustomCallRegistry& registry) { m.Export(registry); },
      [&](DirectCustomCallRegistry& registry) { m.Export(registry); },
  };
  auto state = m.CreateModuleState();
  if (!state.ok()) return state.status();

  CustomCall::UserData user_data;
  auto initialized = m.InitializeUserData(state->get(), user_data);
  if (!initialized.ok()) return initialized;

  auto executed = CompileAndExecute(source, args, registry, /*copts=*/{},
                                    /*type_converter=*/{}, exported, user_data);
  if (!executed.ok()) return executed;

  return state;
}

// No-Op custom call with a single `i32` argument.
static void I32NoOp(DynamicCustomCallRegistry& registry) {
  registry.Register(
      CustomCall::Bind("test.custom_call").Arg<int32_t>().To([](int32_t) {
        return success();
      }));
}

//===----------------------------------------------------------------------===//
// A test for stateful module with a direct custom call.
//===----------------------------------------------------------------------===//

struct Counter : public Module::State {
  int32_t value = 0;
};

// Package custom call that updates a `Counter` as a runtime module.
struct CounterModule : public StatefulModule<Counter> {
  CounterModule() : StatefulModule<Counter>("counter") {}

  static bool Inc(ExecutionContext* e, void** args, void** attrs, void** rets) {
    auto impl = CustomCall::Bind("test.increment")
                    .UserData<Counter*>()  // counter
                    .Arg<int32_t>()        // value
                    .To([](Counter* counter, int32_t value) {
                      return success(counter->value += value);
                    });
    return succeeded(Executable::Call(e, *impl, args, attrs, rets));
  }

  void Export(DirectCustomCallRegistry& registry) const final {
    registry.Register("test.increment", Inc);
  }

  absl::StatusOr<std::unique_ptr<Counter>> CreateModuleState() const final {
    return std::make_unique<Counter>();
  }

  absl::Status InitializeUserData(Counter* state,
                                  CustomCall::UserData& user_data) const final {
    user_data.insert(state);
    return absl::OkStatus();
  }
};

TEST(CustomCallTest, DirectCustomCall) {
  absl::string_view source = R"(
    func.func private @increment(%arg0: i32)
      attributes { rt.custom_call = "test.increment" }

    func.func @test() {
      %0 = arith.constant 42 : i32
      call @increment(%0) : (i32) -> ()
      return
    }
  )";

  auto counter = CompileAndExecute(source, /*args=*/{}, CounterModule());
  ASSERT_TRUE(counter.ok());
  EXPECT_EQ((*counter)->value, 42);
}

//===----------------------------------------------------------------------===//
// All other tests use dynamic custom calls and do not use modules.
//===----------------------------------------------------------------------===//

TEST(CustomCallTest, ScalarArgs) {
  absl::string_view source = R"(
    func.func private @custom_call(%arg0: i1, %arg1: i32, %arg2: i64,
                                   %arg3: f32, %arg4: f64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

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

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call")
                          .Arg<bool>()
                          .Arg<int32_t>()
                          .Arg<int64_t>()
                          .Arg<float>()
                          .Arg<double>()
                          .To(f));
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());

  EXPECT_EQ(i1, false);
  EXPECT_EQ(i32, 42);
  EXPECT_EQ(i64, 42);
  EXPECT_EQ(f32, 42.0);
  EXPECT_EQ(f64, 42.0);
}

TEST(CustomCallTest, ScalarRets) {
  absl::string_view source = R"(
    func.func private @custom_call_result() -> (i1, i32, i64, f32, f64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call_result" }

    func.func private @custom_call(%arg0: i1, %arg1: i32, %arg2: i64,
                                   %arg3: f32, %arg4: f64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

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

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
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
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());

  EXPECT_EQ(i1, false);
  EXPECT_EQ(i32, 42);
  EXPECT_EQ(i64, 42);
  EXPECT_EQ(f32, 42.0);
  EXPECT_EQ(f64, 42.0);
}

TEST(CustomCallTest, StatusOrRet) {
  absl::string_view source = R"(
    func.func private @custom_call_return(%arg0: i32) -> (i64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call_return" }

    func.func private @custom_call(%arg64 : i64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i32
      %1 = call @custom_call_return(%0) : (i32) -> (i64)
      call @custom_call(%1) : (i64) -> ()
      return
    }
  )";

  int64_t i64 = 0;
  auto f_result = [](int32_t arg) -> absl::StatusOr<int64_t> { return arg; };
  auto f = [&](int64_t arg) {
    i64 = arg;
    return success();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call_return")
                          .Arg<int32_t>()
                          .Ret<int64_t>()
                          .To(f_result));

    registry.Register(
        CustomCall::Bind("test.custom_call").Arg<int64_t>().To(f));
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());
  EXPECT_EQ(i64, 42);
}

TEST(CustomCallTest, StatusOrTupleRets) {
  absl::string_view source = R"(
    func.func private @custom_call_return(%arg0 : i64, %arg1 : i64) -> (i64,
                                                                        i64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call_return" }

    func.func private @custom_call(%arg0 : i64, %arg1 : i64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i64
      %1 = arith.constant 43 : i64
      %2, %3 = call @custom_call_return(%0, %1) : (i64, i64) -> (i64, i64)
      call @custom_call(%2, %3) : (i64, i64) -> ()
      return
    }
  )";

  int64_t a = 0;
  int64_t b = 0;
  auto f_result =
      [](int64_t arg0,
         int64_t arg1) -> absl::StatusOr<std::tuple<int64_t, int64_t>> {
    return std::make_tuple(arg0, arg1);
  };
  auto f = [&](int64_t arg0, int64_t arg1) {
    a = arg0;
    b = arg1;
    return success();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call_return")
                          .Arg<int64_t>()
                          .Ret<int64_t>()
                          .Arg<int64_t>()
                          .Ret<int64_t>()
                          .To(f_result));

    registry.Register(CustomCall::Bind("test.custom_call")
                          .Arg<int64_t>()
                          .Arg<int64_t>()
                          .To(f));
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());
  EXPECT_EQ(a, 42);
  EXPECT_EQ(b, 43);
}

TEST(CustomCallTest, OpaqueArgs) {
  absl::string_view source = R"(
    func.func private @use(%arg0: !rt.opaque)
      attributes { rt.dynamic, rt.custom_call = "test.use" }

    func.func @test(%arg0: !rt.opaque) {
      call @use(%arg0) : (!rt.opaque) -> ()
      return
    }
  )";

  // We'll pass around an opaque pointer to this string in our custom calls.
  std::string message = "";

  auto use = [&](void* arg0) {
    std::string* str = reinterpret_cast<std::string*>(arg0);
    (*str) += "foo";
    return success();
  };

  OpaqueArg arg0(&message);

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.use").Arg<void*>().To(use));
  }};

  ASSERT_TRUE(CompileAndExecute(source, {arg0}, registry).ok());
  EXPECT_EQ(message, "foo");
}

TEST(CustomCallTest, OpaqueArgsAndRets) {
  absl::string_view source = R"(
    func.func private @make() -> (!rt.opaque)
      attributes { rt.dynamic, rt.custom_call = "test.make" }

    func.func private @use(%arg0: !rt.opaque)
      attributes { rt.dynamic, rt.custom_call = "test.use" }

    func.func @test() {
      %0 = call @make() : () -> (!rt.opaque)
      call @use(%0) : (!rt.opaque) -> ()
      return
    }
  )";

  // We'll pass around an opaque pointer to this string in our custom calls.
  std::string message = "";

  auto make = [&](Result<void*> res) {
    res.Set(&message);
    return success();
  };

  auto use = [&](void* arg0) {
    std::string* str = reinterpret_cast<std::string*>(arg0);
    (*str) += "foo";
    return success();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.make").Ret<void*>().To(make));
    registry.Register(CustomCall::Bind("test.use").Arg<void*>().To(use));
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());
  EXPECT_EQ(message, "foo");
}

// Instead of passing a pointer to value of underlying type we pass it wrapped
// into a typed reference, for example this would allow to automatically cast
// type-erased `AsyncValue *` to typed `AsyncValuePtr<T>`.
struct ValueRef {
  std::string* value = nullptr;
};

// Register decoding for `ValueRef` (!testlib.value) arguments and results.
XLA_RUNTIME_REGISTER_OPAQUE_ARG_DECODING(ValueRef, std::string*);
XLA_RUNTIME_REGISTER_OPAQUE_RET_DECODING(ValueRef, std::string*);

// Register mapping from custom type id to its unique symbol name.
static void RegisterTypeName(TypeIDNameRegistry& registry) {
  registry.Register<Tagged<ValueRef>>("__type_id_testlib_value");
}

// Register custom call argument encoding for a custom value type.
static void RegisterArgEncoding(CustomCallArgEncodingSet& encoding) {
  encoding.Add<OpaqueArgEncoding>(OpaqueArgEncoding::Match<ValueType>(),
                                  TypeID::get<Tagged<ValueRef>>());
}

// Register custom call result encoding for a custom value type.
static void RegisterRetEncoding(CustomCallRetEncodingSet& encoding) {
  encoding.Add<OpaqueRetEncoding>(OpaqueRetEncoding::Match<ValueType>(),
                                  TypeID::get<Tagged<ValueRef>>());
}

// Conversion from argument compile-time type to the argument run-time types.
static std::unique_ptr<Type> ConvertArgTypeToOpaqueArg(ValueType arg) {
  return std::make_unique<OpaqueOperandType>();
}

// Compilation pipeline options with `testlib` and custom args/rets support.
CompilationPipelineOptions TestlibCopts() {
  CompilationPipelineOptions copts;
  copts.populate_type_id_names = RegisterTypeName;
  copts.populate_arg_encodings = RegisterArgEncoding;
  copts.populate_ret_encodings = RegisterRetEncoding;
  copts.populate_type_conversions = AddTestlibTypeConversions;
  return copts;
}

TEST(CustomCallTest, CustomArgAsOpaqueArg) {
  absl::string_view source = R"(
    func.func private @use(%arg0: !testlib.value)
      attributes { rt.dynamic, rt.custom_call = "test.use" }

    func.func @test(%arg0: !testlib.value) {
      call @use(%arg0) : (!testlib.value) -> ()
      return
    }
  )";

  // We'll pass around an opaque pointer to this string in our custom calls.
  std::string message = "";

  auto use = [&](ValueRef arg0) {
    (*arg0.value) += "foo";
    return success();
  };

  OpaqueArg arg0(&message);

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.use").Arg<ValueRef>().To(use));
  }};

  CompilationPipelineOptions copts = TestlibCopts();
  TypeConverter type_converter(ConvertArgTypeToOpaqueArg);

  ASSERT_TRUE(
      CompileAndExecute(source, {arg0}, registry, copts, type_converter).ok());
  EXPECT_EQ(message, "foo");
}

// In the test above we relied on the conversion of custom argument type to
// opaque type and opaque argument. In this test we introduce a custom type and
// argument to preserve the type information at run time.
struct ValueArgType : public llvm::RTTIExtends<ValueArgType, Type> {
  static constexpr char ID = 0;  // NOLINT
  StatusOr<ArgumentAbi> AsArgument() const final { return ArgumentAbi{1}; }
  std::string ToString() const final { return "!testlib.value"; }
};

// Value argument passed as a single pointer to the XLA executable.
struct ValueArg final : public llvm::RTTIExtends<ValueArg, Argument> {
  static constexpr char ID = 0;  // NOLINT

  explicit ValueArg(std::string* ptr) : ptr(ptr) {}

  absl::Status Verify(const Type& type) const final {
    return llvm::isa<ValueArgType>(type)
               ? absl::OkStatus()
               : absl::InvalidArgumentError("unsupported type");
  }

  void Pack(absl::Span<void*> args) const final {
    args[0] = const_cast<void*>(reinterpret_cast<const void*>(&ptr));
  }

  std::string ToString() const final { return "!testlib.value"; }

  std::string* ptr;
};

// Converts `!testlib.value` type to the `ValueArgType` run-time type.
static std::unique_ptr<Type> ConvertArgTypeToValueArg(ValueType arg) {
  return std::make_unique<ValueArgType>();
}

TEST(CustomCallTest, CustomArg) {
  absl::string_view source = R"(
    func.func private @use(%arg0: !testlib.value)
      attributes { rt.dynamic, rt.custom_call = "test.use" }

    func.func @test(%arg0: !testlib.value) {
      call @use(%arg0) : (!testlib.value) -> ()
      return
    }
  )";

  // We'll pass around an opaque pointer to this string in our custom calls.
  std::string message = "";

  auto use = [&](ValueRef arg0) {
    (*arg0.value) += "bar";
    return success();
  };

  ValueArg arg0(&message);

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.use").Arg<ValueRef>().To(use));
  }};

  CompilationPipelineOptions copts = TestlibCopts();
  TypeConverter type_converter(ConvertArgTypeToValueArg);

  ASSERT_TRUE(
      CompileAndExecute(source, {arg0}, registry, copts, type_converter).ok());
  EXPECT_EQ(message, "bar");
}

TEST(CustomCallTest, CustomArgsAndRets) {
  absl::string_view source = R"(
    func.func private @make() -> (!testlib.value)
      attributes { rt.dynamic, rt.custom_call = "test.make" }

    func.func private @use(%arg0: !testlib.value)
      attributes { rt.dynamic, rt.custom_call = "test.use" }

    func.func @test() {
      %0 = call @make() : () -> (!testlib.value)
      call @use(%0) : (!testlib.value) -> ()
      return
    }
  )";

  // Our `!testlib.value` type at run time will be just a pointer to a string,
  // and it will be encoded similar to the `!rt.opaque` test above.
  std::string message = "";

  auto make = [&](Result<ValueRef> res) {
    res.Set(&message);
    return success();
  };

  auto use = [&](ValueRef arg0) {
    (*arg0.value) += "foo";
    return success();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.make").Ret<ValueRef>().To(make));
    registry.Register(CustomCall::Bind("test.use").Arg<ValueRef>().To(use));
  }};

  CompilationPipelineOptions copts = TestlibCopts();

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry, copts).ok());
  EXPECT_EQ(message, "foo");
}

TEST(CustomCallTest, MemRefRets) {
  absl::string_view source = R"(
    func.func private @custom_call_result() -> memref<2x2xf32>
      attributes { rt.dynamic, rt.custom_call = "test.custom_call_result" }

    func.func private @custom_call(%arg0: memref<2x2xf32>)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = call @custom_call_result() : () -> (memref<2x2xf32>)
      call @custom_call(%0) : (memref<2x2xf32>) -> ()
      return
    }
  )";

  // Allocate storage for arguments.
  std::vector<float> input = {1.0, 2.0, 3.0, 4.0};

  // Observe returned memref by capturing memref argument shape and data.
  std::vector<int64_t> arg_shape;
  std::vector<float> arg_data;

  auto f_result = [&](Result<MemrefView> ret0) {
    std::vector<int64_t> dims = {ret0.GetDims().begin(), ret0.GetDims().end()};
    ret0.Set({ret0.GetDType(), input.data(), dims});
    return success();
  };

  auto f = [&](MemrefView arg0) {
    llvm::ArrayRef<float> data = {reinterpret_cast<float*>(arg0.data), 4};
    arg_shape = {arg0.sizes.begin(), arg0.sizes.end()};
    arg_data = {data.begin(), data.end()};
    return success();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call_result")
                          .Ret<MemrefView>()  // ret0
                          .To(f_result));

    registry.Register(CustomCall::Bind("test.custom_call")
                          .Arg<MemrefView>()  // arg0
                          .To(f));
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());
  EXPECT_EQ(arg_shape, std::vector<int64_t>({2, 2}));
  EXPECT_EQ(arg_data, input);
}

TEST(CustomCallTest, ArgSizeCheck) {
  // Try to pass two argument to a custom call that expects one.
  absl::string_view source = R"(
    func.func private @custom_call(%arg0: i32, %arg1: i32)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i32
      call @custom_call(%0, %0) : (i32, i32) -> ()
      return
    }
  )";

  std::string error = "";

  CustomCallRegistry registry = {I32NoOp};

  auto status = CompileAndExecute(source, /*args=*/{}, registry);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(),
            "run time error: custom call 'test.custom_call' failed: Wrong "
            "number of arguments: expected 1 got 2");
}

TEST(CustomCallTest, ArgTypeCheck) {
  // Try to pass `i64` argument to a custom call that expects `i32`.
  absl::string_view source = R"(
    func.func private @custom_call(%arg1: i64)
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 42 : i64
      call @custom_call(%0) : (i64) -> ()
      return
    }
  )";

  std::string error = "";

  CustomCallRegistry registry = {I32NoOp};

  auto status = CompileAndExecute(source, /*args=*/{}, registry);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(),
            "run time error: custom call 'test.custom_call' failed: Failed to "
            "decode all custom call operands");
}

// Register custom call attribute decoding for `testlib.enum_type`.
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(EnumType);

TEST(CustomCallTest, EnumAttr) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { enum = #testlib.enum_type<Baz> }: () -> ()
      return
    }
  )";

  std::vector<EnumType> enums;

  auto handler = [&](EnumType value) -> LogicalResult {
    enums.push_back(value);
    return success();
  };

  auto types = [](TypeIDNameRegistry& registry) {
    registry.Register<Tagged<EnumType>>("__type_id_testlib_enum");
  };

  auto attrs = [](CustomCallAttrEncodingSet& encoding) {
    encoding.Add<EnumAttrEncoding<EnumTypeAttr, EnumType>>();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call")
                          .Attr<EnumType>("enum")
                          .To(handler));
  }};

  CompilationPipelineOptions copts;
  copts.populate_type_id_names = types;
  copts.populate_attr_encodings = attrs;

  EXPECT_TRUE(CompileAndExecute(source, /*args=*/{}, registry, copts).ok());
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
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { enum = #testlib.enum_type<Baz> }: () -> ()
      return
    }
  )";

  std::vector<MyEnumType> enums;

  auto handler = [&](MyEnumType value) -> LogicalResult {
    enums.push_back(value);
    return success();
  };

  auto types = [](TypeIDNameRegistry& registry) {
    registry.Register<Tagged<MyEnumType>>("__type_id_my_enum");
  };

  auto attrs = [](CustomCallAttrEncodingSet& encoding) {
    encoding.Add<EnumAttrEncoding<EnumTypeAttr, EnumType, MyEnumType>>(
        FromEnumType);
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call")
                          .Attr<MyEnumType>("enum")
                          .To(handler));
  }};

  CompilationPipelineOptions copts;
  copts.populate_type_id_names = types;
  copts.populate_attr_encodings = attrs;

  EXPECT_TRUE(CompileAndExecute(source, /*args=*/{}, registry, copts).ok());
  ASSERT_EQ(enums.size(), 1);
  EXPECT_EQ(enums.front(), MyEnumType::kBaz);
}

// Structure corresponding to the MLIR attribute.
struct PairOfDims {
  int64_t rank;
  llvm::ArrayRef<int64_t> a;
  llvm::ArrayRef<int64_t> b;
};

// Register aggregate attribute decoding.
XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    PairOfDims, AggregateMember<int64_t>("rank"),
    AggregateMember<llvm::ArrayRef<int64_t>>("a"),
    AggregateMember<llvm::ArrayRef<int64_t>>("b"));

TEST(CustomCallTest, StructAttr) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() {
        dims = #testlib.pair_of_dims<2, [1, 1], [2, 2]>
      }: () -> ()
      return
    }
  )";

  int64_t rank = 0;
  std::vector<int64_t> a;
  std::vector<int64_t> b;

  auto handler = [&](PairOfDims value) -> LogicalResult {
    rank = value.rank;
    a.assign(value.a.begin(), value.a.end());
    b.assign(value.b.begin(), value.b.end());
    return success();
  };

  auto types = [](TypeIDNameRegistry& registry) {
    registry.Register<Tagged<PairOfDims>>("__type_id_pair_of_dims");
  };

  auto attrs = [](CustomCallAttrEncodingSet& encoding) {
    encoding.Add<AggregateAttrEncoding<PairOfDimsAttr, PairOfDims>>(
        encoding, AggregateAttrDef<PairOfDimsAttr>()
                      .Add("rank", &PairOfDimsAttr::getRank)
                      .Add("a", &PairOfDimsAttr::getA)
                      .Add("b", &PairOfDimsAttr::getB));
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call")
                          .Attr<PairOfDims>("dims")
                          .To(handler));
  }};

  CompilationPipelineOptions copts;
  copts.populate_type_id_names = types;
  copts.populate_attr_encodings = attrs;

  EXPECT_TRUE(CompileAndExecute(source, /*args=*/{}, registry, copts).ok());
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(a, std::vector<int64_t>(2, 1));
  EXPECT_EQ(b, std::vector<int64_t>(2, 2));
}

TEST(CustomCallTest, FunctionOrdinalAttr) {
  using FunctionOrdinal = CustomCall::FunctionOrdinal;

  absl::string_view source = R"(
    func.func private @init()
      attributes { rt.dynamic, rt.custom_call = "test.init" }

    func.func private @custom_call()
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    // We use a nested call to `@init` custom call as a simple way of proving
    // that `@call_init` was called from `@custom_call` handler.
    func.func @call_init() {
      call @init() : () -> ()
      return
    }

    func.func @test() {
      call @custom_call() { func = @call_init }: () -> ()
      return
    }
  )";

  bool called_init = false;

  // Custom call handler for `@init` custom call.
  auto init = [&]() {
    called_init = true;
    return success();
  };

  // Dynamic custom call registry for resolving nested custom calls.
  DynamicCustomCallRegistry nested_registry;
  nested_registry.Register(CustomCall::Bind("test.init").To(init));

  // Execute options for nested custom calls.
  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &nested_registry;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  // Custom call handler for `@custom_call` custom call.
  auto handler = [&](Executable* executable, FunctionOrdinal exported) {
    FunctionRef fn = executable->function_ref(exported.ordinal);
    return success(fn({}, NoResultConverter{}, execute_opts).ok());
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.init").To(init));
    registry.Register(CustomCall::Bind("test.custom_call")
                          .UserData<Executable*>()
                          .Attr<FunctionOrdinal>("func")
                          .To(handler));
  }};

  std::vector<std::string_view> exported = {"test", "call_init"};
  EXPECT_TRUE(CompileAndExecute(source, /*args=*/{}, registry, /*copts=*/{},
                                /*type_converter=*/{}, exported)
                  .ok());
  EXPECT_TRUE(called_init);
}

TEST(CustomCallTest, OptionalAttr) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { attr0, attr1 = 42 : i64 }: () -> ()
      return
    }
  )";

  std::vector<std::optional<int64_t>> attrs;

  auto handler = [&](std::optional<int64_t> attr0,
                     std::optional<int64_t> attr1) -> LogicalResult {
    attrs.push_back(attr0);
    attrs.push_back(attr1);
    return success();
  };

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call")
                          .Attr<std::optional<int64_t>>("attr0")
                          .Attr<std::optional<int64_t>>("attr1")
                          .To(handler));
  }};

  EXPECT_TRUE(CompileAndExecute(source, /*args=*/{}, registry).ok());
  ASSERT_EQ(attrs.size(), 2);
  EXPECT_EQ(attrs[0], std::nullopt);
  EXPECT_EQ(attrs[1], 42);
}

TEST(CustomCallTest, StateArg) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.dynamic, rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { id = 0 : i64 } : () -> ()
      return
    }
  )";

  auto handler = [](int64_t id, State<int32_t> state0, State<int64_t> state1) {
    state0.GetOrCreate([] { return 42; }).IgnoreError();
    state1.GetOrCreate([] { return 42; }).IgnoreError();
    return success();
  };

  StateVector<int32_t> state_i32;
  StateVector<int64_t> state_i64;

  StateVector<int32_t>::Snapshot snapshot_i32 = state_i32.snapshot();
  StateVector<int64_t>::Snapshot snapshot_i64 = state_i64.snapshot();
  CustomCall::UserData user_data(&snapshot_i32, &snapshot_i64);

  CustomCallRegistry registry = {[&](DynamicCustomCallRegistry& registry) {
    registry.Register(CustomCall::Bind("test.custom_call")
                          .Attr<int64_t>("id")
                          .State<int32_t>("id")
                          .State<int64_t>("id")
                          .To(handler));
  }};

  ASSERT_TRUE(CompileAndExecute(source, /*args=*/{}, registry, /*copts=*/{},
                                /*type_converter=*/{}, {"test"}, user_data)
                  .ok());
  ASSERT_EQ(*state_i32[0], 42);
  ASSERT_EQ(*state_i64[0], 42);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

namespace bm = ::testing::benchmark;

using DirectCustomCall = DirectCustomCallRegistry::DirectCustomCall;
using RuntimeChecks = CustomCall::RuntimeChecks;

// Give short aliases to enums for benchmarks pretty printing.
static constexpr RuntimeChecks all = RuntimeChecks::kDefault;
static constexpr RuntimeChecks less = RuntimeChecks::kLess;
static constexpr RuntimeChecks none = RuntimeChecks::kNone;

static void BenchmarkCustomCall(
    bm::State& state, std::string_view module, ArgumentsRef args,
    std::string_view name, DirectCustomCall custom_call,
    std::function<void(TypeIDNameRegistry&)> types = {},
    std::function<void(CustomCallAttrEncodingSet&)> attrs = {},
    const CustomCall::UserData& user_data = {}) {
  CustomCallRegistry registry;

  // Wrap benchmarked custom call into a direct custom call registry.
  registry.direct_custom_calls = [&](DirectCustomCallRegistry& registry) {
    registry.Register(name, custom_call);
  };

  CompilationPipelineOptions copts;
  copts.populate_type_id_names = std::move(types);
  copts.populate_attr_encodings = std::move(attrs);

  StatusOr<JitExecutable> jit_executable = Compile(module, registry, copts);
  CHECK(jit_executable.ok()) << jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  CHECK(!executable.IsError()) << executable.GetError().message();

  // Prepare the call frame outside of a benchmark loop.
  Executable::CallFrame call_frame;
  CHECK(executable->InitializeCallFrame(args, &call_frame).ok());

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_data = &user_data;
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
static void I32X1(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call(%arg0: i32)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = arith.constant 0 : i32
      call @custom_call(%0) : (i32) -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, source, {}, "test.custom_call", &I32X1<checks>);
}

static void BM_I32X1All(bm::State& s) { I32X1<all>(s); }
static void BM_I32X1None(bm::State& s) { I32X1<none>(s); }

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
static void I32X12(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call(%arg0: i32, %arg1: i32, %arg2: i32,
                                   %arg3: i32, %arg4: i32, %arg5: i32,
                                   %arg6: i32, %arg7: i32, %arg8: i32,
                                   %arg9: i32, %arg10: i32, %arg11: i32)
      attributes { rt.custom_call = "test.custom_call" }

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

  BenchmarkCustomCall(state, source, {}, "test.custom_call", &I32X12<checks>);
}

static void BM_I32X12All(bm::State& s) { I32X12<all>(s); }
static void BM_I32X12None(bm::State& s) { I32X12<none>(s); }

BENCHMARK(BM_I32X12All);
BENCHMARK(BM_I32X12None);

//===----------------------------------------------------------------------===//
// Custom call with a single i32 result.
//===----------------------------------------------------------------------===//

template <RuntimeChecks checks>
static bool RetI32X1(ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .Ret<int32_t>()
          .To<checks>([]() -> absl::StatusOr<int32_t> { return 42; })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void RetI32X1(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call() -> i32
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = call @custom_call() : () -> (i32)
      return
    }
  )";

  BenchmarkCustomCall(state, source, {}, "test.custom_call", &RetI32X1<checks>);
}

static void BM_RetI32X1All(bm::State& s) { RetI32X1<all>(s); }
static void BM_RetI32X1None(bm::State& s) { RetI32X1<none>(s); }

BENCHMARK(BM_RetI32X1All);
BENCHMARK(BM_RetI32X1None);

//===----------------------------------------------------------------------===//
// Custom call with twelve i32 results.
//===----------------------------------------------------------------------===//

template <RuntimeChecks checks>
static bool RetI32X12(ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .Ret<int32_t>()
          .To<checks>(
              []() -> absl::StatusOr<std::tuple<
                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                       int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>> {
                return std::make_tuple(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
              })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void RetI32X12(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call()
      -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11 = call @custom_call()
        : () -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
      return
    }
  )";

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &RetI32X12<checks>);
}

static void BM_RetI32X12All(bm::State& s) { RetI32X12<all>(s); }
static void BM_RetI32X12None(bm::State& s) { RetI32X12<none>(s); }

BENCHMARK(BM_RetI32X12All);
BENCHMARK(BM_RetI32X12None);

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
static void MemrefX1(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call(%arg0: memref<4x4xf32>)
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = memref.alloca() : memref<4x4xf32>
      call @custom_call(%0) : (memref<4x4xf32>) -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &MemrefX1<checks, MemrefType>);
}

static void BM_FlatMemrefX1All(bm::State& s) { MemrefX1<all, Flat>(s); }
static void BM_FlatMemrefX1None(bm::State& s) { MemrefX1<none, Flat>(s); }
static void BM_MemrefX1All(bm::State& s) { MemrefX1<all, MemrefView>(s); }
static void BM_MemrefX1None(bm::State& s) { MemrefX1<none, MemrefView>(s); }
static void BM_StridedMemrefX1All(bm::State& s) { MemrefX1<all, Strided>(s); }
static void BM_StridedMemrefX1None(bm::State& s) { MemrefX1<none, Strided>(s); }

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
static void MemrefX12(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call(
      %arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>,
      %arg3: memref<4x4xf32>, %arg4: memref<4x4xf32>, %arg5: memref<4x4xf32>,
      %arg6: memref<4x4xf32>, %arg7: memref<4x4xf32>, %arg8: memref<4x4xf32>,
      %arg9: memref<4x4xf32>, %arg10: memref<4x4xf32>, %arg11: memref<4x4xf32>
    ) attributes { rt.custom_call = "test.custom_call" }

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

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &MemrefX12<checks, MemrefType>);
}

static void BM_FlatMemrefX12All(bm::State& s) { MemrefX12<all, Flat>(s); }
static void BM_FlatMemrefX12None(bm::State& s) { MemrefX12<none, Flat>(s); }
static void BM_MemrefX12All(bm::State& s) { MemrefX12<all, MemrefView>(s); }
static void BM_MemrefX12None(bm::State& s) { MemrefX12<none, MemrefView>(s); }
static void BM_StridedMemrefX12All(bm::State& s) { MemrefX12<all, Strided>(s); }
static void BM_StridedMemrefX12None(bm::State& s) {
  MemrefX12<none, Strided>(s);
}

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
static void I32AttrX1(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() { attr0 = 42 : i32 }: () -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &I32AttrX1<checks>);
}

static void BM_I32AttrX1All(bm::State& s) { I32AttrX1<all>(s); }
static void BM_I32AttrX1None(bm::State& s) { I32AttrX1<none>(s); }
static void BM_I32AttrX1Less(bm::State& s) { I32AttrX1<less>(s); }

BENCHMARK(BM_I32AttrX1All);
BENCHMARK(BM_I32AttrX1Less);
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
static void I32AttrX12(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "test.custom_call" }

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

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &I32AttrX12<checks>);
}

static void BM_I32AttrX12All(bm::State& s) { I32AttrX12<all>(s); }
static void BM_I32AttrX12None(bm::State& s) { I32AttrX12<none>(s); }
static void BM_I32AttrX12Types(bm::State& s) { I32AttrX12<less>(s); }

BENCHMARK(BM_I32AttrX12All);
BENCHMARK(BM_I32AttrX12Types);
BENCHMARK(BM_I32AttrX12None);

//===----------------------------------------------------------------------===//
// Custom call with a single PairOfDims attribute.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks>
static bool AggregateAttrX1(ExecutionContext* ctx, void** args, void** attrs,
                            void** rets) {
  static auto* handler = CustomCall::Bind("test.custom_call")
                             .Attr<PairOfDims>("attr0")
                             .To<checks>([](PairOfDims attr0) {
                               benchmark::DoNotOptimize(attr0);
                               return success();
                             })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks>
static void AggregateAttrX1(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() {
        attr0 = #testlib.pair_of_dims<2, [1, 1], [2, 2]>
      }: () -> ()
      return
    }
  )";

  auto types = [](TypeIDNameRegistry& registry) {
    registry.Register<Tagged<PairOfDims>>("__type_id_pair_of_dims");
  };

  auto attrs = [](CustomCallAttrEncodingSet& encoding) {
    encoding.Add<AggregateAttrEncoding<PairOfDimsAttr, PairOfDims>>(
        encoding, AggregateAttrDef<PairOfDimsAttr>()
                      .Add("rank", &PairOfDimsAttr::getRank)
                      .Add("a", &PairOfDimsAttr::getA)
                      .Add("b", &PairOfDimsAttr::getB));
  };

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &AggregateAttrX1<checks>, types, attrs);
}

static void BM_AggregateAttrX1All(bm::State& s) { AggregateAttrX1<all>(s); }
static void BM_AggregateAttrX1None(bm::State& s) { AggregateAttrX1<none>(s); }
static void BM_AggregateAttrX1Less(bm::State& s) { AggregateAttrX1<less>(s); }

BENCHMARK(BM_AggregateAttrX1All);
BENCHMARK(BM_AggregateAttrX1Less);
BENCHMARK(BM_AggregateAttrX1None);

//===----------------------------------------------------------------------===//
// Custom call with UserData arguments.
//===----------------------------------------------------------------------===//

// Use std::integral_constant to fake multiple unique UserData types.
template <int value>
using Data = std::integral_constant<int, value>;

// Benchmark how long it takes to prepare UserData.
static void BM_PrepareUserData(bm::State& state) {
  Data<0> data0;
  Data<1> data1;
  Data<2> data2;
  Data<3> data3;
  Data<4> data4;
  Data<5> data5;
  Data<6> data6;
  Data<7> data7;
  Data<8> data8;
  Data<9> data9;

  for (auto _ : state) {
    CustomCall::UserData user_data(&data0, &data1, &data2, &data3, &data4,
                                   &data5, &data6, &data7, &data8, &data9);
    benchmark::DoNotOptimize(user_data);
  }
}

BENCHMARK(BM_PrepareUserData);

template <CustomCall::RuntimeChecks checks>
static bool UserDataX12(ExecutionContext* ctx, void** args, void** attrs,
                        void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .UserData<Data<0>*>()
          .UserData<Data<1>*>()
          .UserData<Data<2>*>()
          .UserData<Data<3>*>()
          .UserData<Data<4>*>()
          .UserData<Data<5>*>()
          .UserData<Data<6>*>()
          .UserData<Data<7>*>()
          .UserData<Data<8>*>()
          .UserData<Data<9>*>()
          .UserData<Data<10>*>()
          .UserData<Data<11>*>()
          .To<checks>([](Data<0>* data0, Data<1>* data1, Data<2>* data2,
                         Data<3>* data3, Data<4>* data4, Data<5>* data5,
                         Data<6>* data6, Data<7>* data7, Data<8>* data8,
                         Data<9>* data9, Data<10>* data10, Data<11>* data11) {
            benchmark::DoNotOptimize(data0);
            benchmark::DoNotOptimize(data1);
            benchmark::DoNotOptimize(data2);
            benchmark::DoNotOptimize(data3);
            benchmark::DoNotOptimize(data4);
            benchmark::DoNotOptimize(data5);
            benchmark::DoNotOptimize(data6);
            benchmark::DoNotOptimize(data7);
            benchmark::DoNotOptimize(data8);
            benchmark::DoNotOptimize(data9);
            benchmark::DoNotOptimize(data10);
            benchmark::DoNotOptimize(data11);
            return success();
          })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks>
static void UserDataX12(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      call @custom_call() : () -> ()
      return
    }
  )";

  Data<0> data0;
  Data<1> data1;
  Data<2> data2;
  Data<3> data3;
  Data<4> data4;
  Data<5> data5;
  Data<6> data6;
  Data<7> data7;
  Data<8> data8;
  Data<9> data9;
  Data<10> data10;
  Data<11> data11;

  CustomCall::UserData user_data;
  user_data.insert_all(&data0, &data1, &data2, &data3, &data4, &data5, &data6,
                       &data7, &data8, &data9, &data10, &data11);

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &UserDataX12<checks>, {}, {}, user_data);
}

static void BM_UserDataX12All(bm::State& s) { UserDataX12<all>(s); }
static void BM_UserDataX12None(bm::State& s) { UserDataX12<none>(s); }
static void BM_UserDataX12Less(bm::State& s) { UserDataX12<less>(s); }

BENCHMARK(BM_UserDataX12All);
BENCHMARK(BM_UserDataX12Less);
BENCHMARK(BM_UserDataX12None);

//===----------------------------------------------------------------------===//
// Benchmark memref encoding for a sequence of custom calls.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks>
static bool RemainingArgsSink(ExecutionContext* ctx, void** args, void** attrs,
                              void** rets) {
  static auto* handler =
      CustomCall::Bind("test.custom_call")
          .RemainingArgs()
          .To<checks>([](CustomCall::RemainingArgs) { return success(); })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CustomCall::RuntimeChecks checks>
static void MemrefEncoding(bm::State& state) {
  absl::string_view source = R"(
    func.func private @custom_call(
      %arg0: memref<4x4xf32>, %arg1: memref<5x5xf32>, %arg2: memref<6x6xf32>,
      %arg3: memref<4x4xf32>, %arg4: memref<5x5xf32>, %arg5: memref<6x6xf32>
    ) attributes { rt.custom_call = "test.custom_call" }

    func.func @test() {
      %0 = memref.alloca() : memref<4x4xf32>
      %1 = memref.alloca() : memref<5x5xf32>
      %2 = memref.alloca() : memref<6x6xf32>

      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      call @custom_call(%0, %1, %2, %0, %1, %2)
        : (memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>,
           memref<4x4xf32>, memref<5x5xf32>, memref<6x6xf32>) -> ()
      return
    }
  )";

  BenchmarkCustomCall(state, source, {}, "test.custom_call",
                      &RemainingArgsSink<checks>);
}

static void BM_MemrefEncoding(bm::State& s) { MemrefEncoding<none>(s); }

BENCHMARK(BM_MemrefEncoding);

}  // namespace runtime
}  // namespace xla

// Add explicit dense type ids for all data types passed as UserData to measure
// the effects of explicit type id declaration/definition.
#define DEFINE_DENSE_TYPE_ID(n)                                        \
  XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall, \
                                             xla::runtime::Data<n>);   \
  XLA_RUNTIME_DEFINE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall,  \
                                            xla::runtime::Data<n>)

DEFINE_DENSE_TYPE_ID(0);
DEFINE_DENSE_TYPE_ID(1);
DEFINE_DENSE_TYPE_ID(2);
DEFINE_DENSE_TYPE_ID(3);
DEFINE_DENSE_TYPE_ID(4);
DEFINE_DENSE_TYPE_ID(5);
DEFINE_DENSE_TYPE_ID(6);
DEFINE_DENSE_TYPE_ID(7);
DEFINE_DENSE_TYPE_ID(8);
DEFINE_DENSE_TYPE_ID(9);
DEFINE_DENSE_TYPE_ID(10);
DEFINE_DENSE_TYPE_ID(11);

#undef DEFINE_DENSE_TYPE_ID
