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
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace runtime {

// ===---------------------------------------------------------------------===//
// A helper function that compiles `module` to XLA runtime executable and runs
// `test` function with the given arguments. Caller can also register custom
// calls (direct or dynamic) and custom types.
// ===---------------------------------------------------------------------===//

static absl::Status CompileAndExecute(
    std::string_view module, ArgumentsRef args,
    CustomCallRegistry custom_call_registry = {},
    DirectCustomCallLibrary custom_call_library = {},
    TypeIDNameRegistry::RegistrationFn types = {}) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding = ToSymbolsBinding(custom_call_library, types);

  opts.compiler.register_dialects = [&](mlir::DialectRegistry& registry) {
    RegisterDefaultXlaRuntimeDialects(registry);
  };

  opts.compiler.create_compilation_pipeline = [&](mlir::PassManager& pm) {
    CompilationPipelineOptions copts;
    CreateDefaultXlaRuntimeCompilationPipeline(pm, copts);
  };

  absl::StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(module, "test", opts);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError())
    return absl::InternalError(executable.GetError().message);

  // Prepare the call frame outside of a benchmark loop.
  Executable::CallFrame call_frame;
  auto initialized = executable->InitializeCallFrame(args, &call_frame);
  if (!initialized.ok()) return initialized;

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &custom_call_registry;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  executable->Execute(call_frame, execute_opts);
  if (call_frame.is_error) return absl::InternalError(call_frame.error);

  return absl::OkStatus();
}

// ===---------------------------------------------------------------------===//

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

  CustomCallRegistry registry;
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

}  // namespace runtime
}  // namespace xla
