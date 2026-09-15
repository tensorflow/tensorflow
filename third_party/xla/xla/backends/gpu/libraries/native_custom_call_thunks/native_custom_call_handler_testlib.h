/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_TESTLIB_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_TESTLIB_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Runs a custom-call thunk-folding handler over a module given as HLO text,
// without needing a GPU or a full compiler run.
//
//   ASSERT_OK_AND_ASSIGN(auto tester, NativeCustomCallHandlerTester::Create(R"(
//     ENTRY e {
//       ROOT c = f32[4] custom-call(),
//         custom_call_target="xla.gpu.my_target",
//         backend_config="{val = 3 : i32}"
//     }
//   )"));
//   ASSERT_OK_AND_ASSIGN(ThunkSequence thunks, tester->EmitThunks());
//
// The tester parses the module and runs a real buffer assignment over it, so
// the slices a handler sees are the ones it would see during compilation.
//
// The emitted thunks point into the module and the buffer assignment that the
// tester owns, so the tester must outlive them.
class NativeCustomCallHandlerTester {
 public:
  // XLA still builds as C++17, so populate this with positional list
  // initialization (`{/*gpu_model=*/GpuModel::H100_SXM,
  // /*instruction_name=*/"c0"}`) rather than designated initializers.
  struct Options {
    // The device the handler is told it is compiling for.
    GpuModel gpu_model = GpuModel::H100_SXM;

    // Name of the custom call to lower. Empty means the entry root, which must
    // then be a custom call.
    std::string instruction_name;

    // Debug options the handler sees. Defaults to the same values the compiler
    // would use in the absence of any flag.
    DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  };

  static absl::StatusOr<std::unique_ptr<NativeCustomCallHandlerTester>> Create(
      absl::string_view hlo_text);
  static absl::StatusOr<std::unique_ptr<NativeCustomCallHandlerTester>> Create(
      absl::string_view hlo_text, Options options);

  NativeCustomCallHandlerTester(const NativeCustomCallHandlerTester&) = delete;
  NativeCustomCallHandlerTester& operator=(
      const NativeCustomCallHandlerTester&) = delete;

  // Invokes the handler registered in the global registry for this custom
  // call's target. Returns `NotFound` if there is no such handler; note that
  // a handler is only registered if the library defining it is linked into the
  // test binary.
  absl::StatusOr<ThunkSequence> EmitThunks() const;

  // Invokes `handler` directly, bypassing the registry.
  absl::StatusOr<ThunkSequence> EmitThunksWith(
      NativeCustomCallHandlerRef handler) const;

  const HloCustomCallInstruction& instruction() const { return *instruction_; }
  const NativeCustomCallEmitterContext& context() const { return *context_; }
  const HloModule& module() const { return *module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }

  ~NativeCustomCallHandlerTester();

 private:
  class ContextImpl;

  NativeCustomCallHandlerTester();

  std::unique_ptr<HloModule> module_;
  AliasInfo alias_info_;
  std::unique_ptr<BufferAssignment> buffer_assignment_;
  std::unique_ptr<GpuTopology> topology_;
  const HloCustomCallInstruction* instruction_ = nullptr;
  std::unique_ptr<NativeCustomCallEmitterContext> context_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_TESTLIB_H_
