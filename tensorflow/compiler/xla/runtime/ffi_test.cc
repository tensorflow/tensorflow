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
#include <memory>
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

static absl::StatusOr<JitExecutable> Compile(std::string_view source) {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding = ToSymbolsBinding();
  opts.compiler.register_dialects = RegisterDefaultXlaGpuRuntimeDialects;
  opts.compiler.create_compilation_pipeline =
      [&](xla::runtime::PassManager& passes) {
        CompilationPipelineOptions copts;
        CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
      };

  return JitExecutable::Instantiate(source, opts, {"test"});
}

static absl::Status CompileAndExecute(std::string_view source,
                                      ArgumentsRef args,
                                      const DynamicCustomCallRegistry& registry,
                                      CustomCall::UserData user_data) {
  absl::StatusOr<JitExecutable> jit_executable = Compile(source);
  if (!jit_executable.ok()) return jit_executable.status();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError())
    return absl::InternalError(executable.GetError().message());

  std::string diagnostic;
  DiagnosticEngine diagnostic_engine = CollectDiagnostic(&diagnostic);

  Executable::ExecuteOpts execute_opts;
  execute_opts.custom_call_registry = &registry;
  execute_opts.diagnostic_engine = &diagnostic_engine;
  execute_opts.custom_call_data = &user_data;
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

using ffi::FfiStatus;

// When FFI module is instantiated for an Xla runtime executable, it creates a
// state object whose lifetime is bound to the executable, and the state can be
// accessed from exported FFI functions.
struct TestModuleState {
  int32_t i32 = 0;
  int32_t f32 = 0;
};

// TestModule is a stateful FFI module with every exported function having
// access to the instance of `TestModuleState`. State is optional, it's ok to
// skip it in the FFI binding if it's not needed.
struct TestModule : public ffi::StatefulModule<TestModuleState> {
  using Base = ffi::StatefulModule<TestModuleState>;

  explicit TestModule(const XLA_FFI_Api* api)
      : Base(api, "ffi-module", {{"ffi.fill", FFI_Fill}}) {}

  // Creates a new TestModule state for each executable.
  std::unique_ptr<TestModuleState> CreateState() final {
    return std::make_unique<TestModuleState>();
  }

  // Prepare `Fill` function for export as `FFI_Fill` FFI function.
  XLA_FFI_DEFINE_FUNCTION(FFI_Fill, Fill,
                          ffi::Ffi::Bind("ffi.fill")
                              .State<TestModuleState>()  // state
                              .Arg<int32_t>()            // arg0
                              .Arg<ffi::BufferArg>()     // arg1
                              .Attr<float>("attr"));

  static FfiStatus Fill(TestModuleState* state, int32_t arg0,
                        ffi::BufferArg arg1, float attr0);
};

FfiStatus TestModule::Fill(TestModuleState* state, int32_t arg0,
                           ffi::BufferArg arg1, float attr0) {
  // Update state to observe side effects.
  state->i32 = arg0;
  state->f32 = attr0;

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

//----------------------------------------------------------------------------//

TEST(FfiTest, ScalarAndBufferArgs) {
  absl::string_view source = R"(
    func.func private @fill(%arg0: i32, %arg1: memref<?x?xf32>)
      attributes { rt.dynamic, rt.custom_call = "ffi.fill" }

    func.func @test(%arg0: memref<?x?xf32>) {
      %0 = arith.constant 42 : i32
      call @fill(%0, %arg0) { attr = 42.0 : f32 } : (i32, memref<?x?xf32>) -> ()
      return
    }
  )";

  // When module is instantiated it's automatically registered with the runtime.
  TestModule module(GetXlaFfiApi());

  // Check that it was registered with the runtime.
  std::vector<const Module*> modules = ffi::FfiModules();
  ASSERT_EQ(modules.size(), 1);
  EXPECT_EQ(modules[0]->name(), "ffi-module");

  // Export custom calls defined by FFI modules.
  DynamicCustomCallRegistry registry;
  ffi::ExportFfiModules(registry);
  EXPECT_TRUE(registry.Find("ffi.fill"));

  // Instantiate state for all registered FFI modules.
  auto state = ffi::FfiModulesState::Instantiate();
  ASSERT_TRUE(state.ok());

  // Add an FFI state vector to the UserData.
  ffi::FfiStateVector state_vector = state->state_vector();
  CustomCall::UserData user_data(&state_vector);
  ASSERT_EQ(state_vector.state.size(), 1);

  // Use vector as buffer storage.
  std::vector<float> buffer(16);

  // Use row major layout.
  std::array<int64_t, 2> sizes = {8, 2};
  std::array<int64_t, 2> strides = {2, 1};

  // Pass a single memref argument to the executable.
  std::vector<MemrefDesc> args;
  args.emplace_back(PrimitiveType::F32, buffer.data(), 0, sizes, strides);

  ASSERT_TRUE(CompileAndExecute(source, args, registry, user_data).ok());

  // Check that the FFI function updated the corresponding module state.
  auto* state_ptr = reinterpret_cast<TestModuleState*>(state_vector.state[0]);
  EXPECT_EQ(state_ptr->i32, 42);
  EXPECT_EQ(state_ptr->f32, 42.0);

  // Check that FFI function filled the buffer argument with data.
  EXPECT_EQ(buffer, std::vector<float>(16, 42.0));
}

}  // namespace runtime
}  // namespace xla
