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

#include "tensorflow/compiler/xla/runtime/ffi.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_api.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_c_api.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/results.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace runtime {

// Diagnostic engine that appends all emitted diagnostics to the `error` string.
static DiagnosticEngine CollectDiagnostic(std::string* error) {
  DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([=](Diagnostic& diagnostic) -> LogicalResult {
    error->append(diagnostic.status().message());
    return success();
  });
  return diagnostic_engine;
}

static absl::StatusOr<JitExecutable> Compile(std::string_view module) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding = ToSymbolsBinding();
  opts.compiler.register_dialects = RegisterDefaultXlaGpuRuntimeDialects;
  opts.compiler.create_compilation_pipeline =
      [&](xla::runtime::PassManager& passes) {
        CompilationPipelineOptions copts;
        CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
      };

  return JitExecutable::Instantiate(module, opts, {"test"});
}

static absl::Status CompileAndExecute(std::string_view module,
                                      ArgumentsRef args) {
  absl::StatusOr<JitExecutable> jit_executable = Compile(module);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError())
    return absl::InternalError(executable.GetError().message());

  std::string diagnostic;
  DiagnosticEngine diagnostic_engine = CollectDiagnostic(&diagnostic);

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &ffi::FfiCustomCalls();
  execute_opts.diagnostic_engine = &diagnostic_engine;
  execute_opts.async_task_runner =
      reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  // Append emited diagnostic if execution failed.
  auto executed = executable->Execute(args, NoResultConverter{}, execute_opts);
  if (!executed.ok()) {
    return absl::InternalError(
        absl::StrFormat("%s: %s", executed.message(), diagnostic));
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

using ::xla::runtime::ffi::Ffi;
using ::xla::runtime::ffi::FfiStatus;

// Use these static variables to observe custom call side effects.
static int32_t i32 = 0;
static float f32 = 0;

// Typed XLA FFI function handler that will be registered with the runtime.
static FfiStatus TestFnImpl(int32_t arg0, ffi::BufferArg arg1, float attr0) {
  // Update static variables to observe side effects.
  i32 = arg0;
  f32 = attr0;

  // Write attribute value into the buffer argument.
  if (arg1.dtype != ffi::PrimitiveType::F32)
    return FfiStatus::InvalidArgument("Unsupported buffer type");
  if (arg1.sizes.size() != 2)
    return FfiStatus::InvalidArgument("Unsupported buffer rank");

  size_t size = arg1.sizes[0] * arg1.sizes[1];
  float* data = reinterpret_cast<float*>(arg1.data);
  std::fill(data, data + size, attr0);

  return FfiStatus::Ok();
}

// Bind `TestFn` function to `TestFnImpl` handler.
XLA_FFI_DEFINE_FUNCTION(TestFn, TestFnImpl,
                        ffi::Ffi::Bind("test.ffi")
                            .Arg<int32_t>()
                            .Arg<ffi::BufferArg>()
                            .Attr<float>("attr"));

TEST(FfiTest, ScalarAndBufferArgs) {
  const XLA_FFI_Api* api = GetXlaFfiApi();
  Ffi::Register(api, "test.ffi", TestFn);

  absl::string_view module = R"(
    func.func private @ffi(%arg0: i32, %arg1: memref<?x?xf32>)
      attributes { rt.dynamic, rt.custom_call = "test.ffi" }

    func.func @test(%arg0: memref<?x?xf32>) {
      %0 = arith.constant 42 : i32
      call @ffi(%0, %arg0) { attr = 42.0 : f32 } : (i32, memref<?x?xf32>) -> ()
      return
    }
  )";

  // Use vector as buffer storage.
  std::vector<float> buffer(16);

  // Use row major layout.
  std::array<int64_t, 2> sizes = {8, 2};
  std::array<int64_t, 2> strides = {2, 1};

  // Pass a single memref argument to the executable.
  std::vector<MemrefDesc> args;
  args.emplace_back(PrimitiveType::F32, buffer.data(), 0, sizes, strides);

  ASSERT_TRUE(CompileAndExecute(module, args).ok());

  // Check FFI handler side effects.
  EXPECT_EQ(i32, 42);
  EXPECT_EQ(f32, 42.0);
  EXPECT_EQ(buffer, std::vector<float>(16, 42.0));
}

}  // namespace runtime
}  // namespace xla
