/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {
namespace gpu {

class TritonFusion : public FusionInterface {
 public:
  struct LaunchConfig {
    LaunchDimensions launch_dimensions;
    BlockLevelParameters block_level_parameters;
  };

  explicit TritonFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

  // A kernel thunk and the corresponding LLVM module if it was not cached.
  // This is a more concrete version of FusionEmissionResult that can be used in
  // places where we know we are dealing with Triton fusions.
  struct EmitResult {
    std::unique_ptr<KernelThunk> kernel_thunk;
    std::unique_ptr<llvm::Module> llvm_module;
  };
  // Overload of [Emit] that allows passing overrides for instructions
  // and unmanaged arguments.
  // - Instruction overloads are required when we forcibly form fusions for
  // instructions by extracting them into a separate HLO module. In this case
  // buffer assignments are still associated with the original instruction.
  // So the root of the fusion cannot be used to determine the emitted kernel
  // arguments.
  // - The unmanaged arguments are used for collectives which have extra
  // arguments (such as pointers to remote buffers, and metadata arguments).
  // Empty unmanaged arguments mean that all arguments have buffer slices
  // associated with them.
  //
  // TODO(b/461717780): Remove the instruction override once we form collective
  // based fusions earlier in the compiler pipeline.
  // Returns a pair of the kernel thunk and an llvm module. The local module
  // is only returned if the kernel was not cached.
  absl::StatusOr<EmitResult> Emit(
      IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
      const HloInstruction* instr_override,
      absl::Span<const Shape> unmanaged_arguments) const;

  // Returns the launch config for Triton fusions that have a block level fusion
  // config.
  // Not supported for MatMul fusions yet.
  //
  // If `thread_dims_override` is provided, it is used instead of the thread
  // dimensions calculated in the function.
  // Ideally, we should pass the values extracted from the Triton module after
  // compilation, but there are use-cases where we want to call the function
  // without compiling, e.g. during cost modelling. In that case, the function
  // calculates the values.
  std::optional<LaunchConfig> GetLaunchConfig(
      std::optional<se::ThreadDim> thread_dims_override = std::nullopt) const;

  // Generates a Triton kernel for the given fusion into the provided LLVM
  // module, and returns the `TritonWrapperResult` corresponding to the
  // generated kernel.
  absl::StatusOr<TritonWrapperResult> GenerateTritonKernelAndWrapper(
      const HloFusionInstruction& fusion, absl::string_view impl_fn_name,
      const se::DeviceDescription& device_info,
      const llvm::Triple& target_triple, const std::string& data_layout,
      llvm::LLVMContext* llvm_context, mlir::MLIRContext* mlir_context) const;

 private:
  const HloFusionAnalysis& analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_H_
