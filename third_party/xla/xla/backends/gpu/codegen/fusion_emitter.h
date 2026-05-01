/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_GPU_CODEGEN_FUSION_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_FUSION_EMITTER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/IR/AffineMap.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

struct FusionEmissionResult {
  AsyncThunkSequence thunks;
  std::unique_ptr<llvm::Module> module;
};

class FusionInterface {
 public:
  virtual ~FusionInterface() = default;

  virtual absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const = 0;
};

// Interface for fusions that are implemented using cuda kernels.
class KernelFusionInterface : public FusionInterface {
 public:
  ~KernelFusionInterface() override = default;

  // Returns the fusion's launch dimensions.
  virtual LaunchDimensions launch_dimensions() const = 0;
};

void CopySelectAttrs(const llvm::Function& src, llvm::Function& dst);
void AnnotateAttrsIfUnset(const emitters::KernelArguments& arguments,
                          llvm::Function& dst);

absl::StatusOr<llvm::Function*> BuildKernelPrototype(
    llvm::Module* llvm_module, const se::DeviceDescription& gpu_device_info,
    const std::string& impl_fn_name, const std::string& unique_kernel_name,
    const emitters::KernelArguments& arguments,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilderBase* builder);

absl::StatusOr<llvm::Function*> RemoveUnusedTritonAbiArguments(
    llvm::Module* llvm_module, IrEmitterContext& ir_emitter_context,
    const std::string& sanitized_kernel_name, bool keep_scratch = false);

absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, llvm::Function* kernel,
    llvm::Module* llvm_module);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_FUSION_EMITTER_H_
