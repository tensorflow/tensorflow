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
#include "mlir/IR/MLIRContext.h"
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
  std::vector<std::unique_ptr<Thunk>> thunks;
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
  virtual ~KernelFusionInterface() = default;

  // Returns the fusion's launch dimensions.
  virtual LaunchDimensions launch_dimensions() const = 0;

  // Computes an indexing map from thread to output element(s) of the **hero**.
  //
  // The dimensions in the resulting map are
  //   d0, d1, d2: threadIdx.{x,y,z}
  //   d3, d4, d5: blockIdx.{x,y,z}
  // If one thread computes multiple elements, this will be represented using a
  // symbol.
  //
  // Cases where the exact element cannot be statically determined are currently
  // unsupported (scatter, in-place DUS). Implementations will return nullopt.
  // Note: Work in progress, not implemented for all emitters.
  virtual std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const = 0;

  // Computes an indexing map from thread to input element(s) of the root's
  // **hero**. Note that in many cases this is not computable from the output
  // indexing. The indexing may only be known for some operands of the hero.
  virtual std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const = 0;

  static constexpr std::array<int, 3> kIndexingMapThreadIdxDims = {0, 1, 2};
  static constexpr std::array<int, 3> kIndexingMapBlockIdxDims = {3, 4, 5};

 protected:
  // Returns the default mapping for the given launch dimensions: linearizes
  // the thread index and then reshapes it into the given layout.
  // Populates the ranges for d0, d1, d2, d3, d4, d5 from the thread counts and
  // block sizes in the given launch dimensions.
  static IndexingMap GetDefaultThreadIdIndexingMap(
      const LaunchDimensions& launch_dims, int unroll_factor,
      const Shape& shape, mlir::MLIRContext* ctx);
};

absl::StatusOr<
    std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray /*inputs*/>,
               std::vector<llvm_ir::IrArray> /*outputs*/>>
BuildKernelPrototype(IrEmitterContext& ir_emitter_context,
                     const std::string& impl_fn_name,
                     const std::string& suggested_name,
                     absl::Span<const emitters::KernelArgument> arguments,
                     size_t num_inputs,
                     const LaunchDimensions& launch_dimensions,
                     llvm::IRBuilderBase* builder);
absl::StatusOr<
    std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray /*inputs*/>,
               std::vector<llvm_ir::IrArray> /*outputs*/>>
BuildKernelPrototypeFromUniqueName(
    IrEmitterContext& ir_emitter_context, const std::string& impl_fn_name,
    const std::string& unique_name,
    absl::Span<const emitters::KernelArgument> arguments, size_t num_inputs,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilderBase* builder);

// Compute the kernel name. The opcode string may contain "-" which cannot be
// in a PTX function name, so sanitize the name before uniquifying it.
std::string GetSanitizedUniqueName(IrEmitterContext& ir_emitter_context,
                                   const std::string& suggested_name);

absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, llvm::Function* kernel,
    llvm::Module* llvm_module);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_FUSION_EMITTER_H_
