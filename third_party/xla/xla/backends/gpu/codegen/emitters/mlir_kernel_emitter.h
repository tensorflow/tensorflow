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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_MLIR_KERNEL_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_MLIR_KERNEL_EMITTER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// MlirKernelEmitter is an abstract base class for emitters that generate
// MLIR(Multi-Level Intermediate Representation) from an HLO fusion.
//
// Subclasses of MlirKernelEmitter implement the specific logic for
// different types of fusions (e.g., LoopFusion, ReductionFusion), translating
// the high-level HLO operations within the fusion into an MLIR module.
// The primary method to override and implement is `EmitEntryFunction`,
// which populates the body of the main kernel function in the MLIR module.
// Other virtual methods allow customization of launch dimensions, indexing
// maps, and epilogues.
class MlirKernelEmitter {
 public:
  virtual ~MlirKernelEmitter() = default;

  virtual LaunchDimensions launch_dimensions() const = 0;
  virtual int unroll_factor() const { return 0; }

  absl::StatusOr<MlirKernelSource> Emit(
      mlir::MLIRContext* mlir_context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const;

  // Visible for testing. `buffer_assignment` is optional for testing (assigns
  // a different buffer to each tensor).
  virtual absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const;

  static mlir::DialectRegistry GetDialectRegistry();

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

  // Computes indexing maps from thread id to input elements of the root's
  // **hero**. Note that in many cases this is not computable from the output
  // indexing. The indexing may only be known for some operands of the hero.
  virtual std::optional<std::vector<IndexingMap>>
  ComputeThreadIdToInputIndexing(int64_t root_index,
                                 mlir::MLIRContext* ctx) const = 0;

 protected:
  // Returns the set of instructions that will be isolated in the partitioned,
  // i.e., they will get their own subgraph. We won't automatically emit
  // functions for these instructions.
  virtual std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const {
    return {};
  }

  // Creates an epilogue with the raw thread/block/symbol indices, as defined
  // by the fusion's thread->output mapping.
  emitters::EpilogueSpecification GetEpilogueForOutputIndexing(
      const HloFusionAnalysis& analysis,
      const std::vector<const HloInstruction*>& heroes,
      const std::vector<const HloInstruction*>& roots,
      mlir::MLIRContext* mlir_context) const;

  virtual absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const = 0;

  static std::array<uint64_t, 2> MaybeSplitGridDimensionX(
      uint64_t num_threads_x, uint64_t num_blocks_x,
      const se::DeviceDescription& info);

  mlir::Value EmitWorkGroupId(mlir::ImplicitLocOpBuilder& builder,
                              WorkGroupDimension dim) const;
  mlir::Value EmitBlockId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
  mlir::Value EmitThreadId(mlir::ImplicitLocOpBuilder& builder, int dim) const;
  llvm::SmallVector<mlir::Value> EmitThreadAndBlockIds(
      mlir::ImplicitLocOpBuilder& builder) const;

  // Returns the default mapping for the given launch dimensions: linearizes
  // the thread index and then reshapes it into the given layout.
  // Populates the ranges for d0, d1, d2, d3, d4, d5 from the thread counts and
  // block sizes in the given launch dimensions.
  static IndexingMap GetDefaultThreadIdIndexingMap(
      const LaunchDimensions& launch_dims, int unroll_factor,
      const Shape& shape, mlir::MLIRContext* mlir_context);

 private:
  // Emits MLIR for the given fusion. The entry function has one tensor argument
  // per fusion parameter and output and one tensor result per fusion output.
  // The fuson outputs may only be used with `tensor.insert` ops.a
  absl::Status EmitMlir(mlir::ModuleOp module,
                        mlir::func::FuncOp entry_function,
                        const HloFusionInstruction& fusion,
                        mlir::MLIRContext& mlir_context) const;
};

// MlirKernelFusion encapsulates an MlirKernelEmitter, which is responsible for
// generating an MLIR module from an HLO fusion. MlirKernelFusion then takes
// this MLIR module and handles the process of lowering it through various
// passes down to LLVM IR.
class MlirKernelFusion final : public KernelFusionInterface {
 public:
  explicit MlirKernelFusion(std::unique_ptr<MlirKernelEmitter> emitter)
      : emitter_(std::move(emitter)) {}

  LaunchDimensions launch_dimensions() const override {
    return emitter_->launch_dimensions();
  }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const {
    return emitter_->ComputeThreadIdToOutputIndexing(root_index, ctx);
  }

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const {
    return emitter_->ComputeThreadIdToInputIndexing(root_index, ctx);
  }

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

  // Visible for testing. `buffer_assignment` is optional for testing (assigns
  // a different buffer to each tensor).
  absl::StatusOr<std::unique_ptr<llvm::Module>> CreateLLVMModule(
      mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
      const se::DeviceDescription& device, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const;

  MlirKernelEmitter* mlir_kernel_emitter() { return emitter_.get(); }

  static constexpr std::array<int, 3> kIndexingMapThreadIdxDims = {0, 1, 2};
  static constexpr std::array<int, 3> kIndexingMapBlockIdxDims = {3, 4, 5};

 private:
  std::unique_ptr<MlirKernelEmitter> emitter_;
};

// Adds passes that transform XLA_GPU and SCF loops, e.g. peel, pipeline,
// vectorize.
void AddLoopTransformationPasses(mlir::OpPassManager& pm,
                                 const se::DeviceDescription& device,
                                 int max_unroll_factor = 0);

// Adds passes that lower transformed loops to LLVM.
void AddLoweringPasses(mlir::OpPassManager& pm,
                       const se::DeviceDescription& device);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_MLIR_KERNEL_EMITTER_H_
