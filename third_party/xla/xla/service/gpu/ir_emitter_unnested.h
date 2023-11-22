/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
#define XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter.h"
#include "xla/service/gpu/kernel_mapping_scheme.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

struct BufferSlice {
  // The root buffer to look at.
  BufferAllocation::Slice buffer_slice;

  // The global constant name of the buffer, if it's a constant.
  std::string constant_name;

  // The buffer is modified by the kernel.
  bool written = false;

  Shape shape;
};

// Emits LLVM IR for an "unnested computation".
//
// An unnested computation is an HloComputation which you run by executing one
// or more kernels for each HloInstruction it contains.  Examples of unnested
// computations:
//
//  - An HloModule's root computation,
//  - The body of an HLO while loop,
//  - The true/false computation of an HLO conditional.
//
// Note the opportunity for confusion -- the while loop's computation is nested
// within the root computation, but it's emitted using IrEmitterUnnested!  Don't
// think about it too hard.
//
// Examples of things that are not unnested computations:
//
//  - The body of a fusion node.  IrEmitterUnnested emits the relevant code
//    within a kernel function using FusedIrEmitter.  (FusedIrEmitter is not
//    really an IrEmitter, but is more an "IR generator generator".)
//
class IrEmitterUnnested : public IrEmitter {
 public:
  absl::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  using ValueVector3 = std::array<llvm::Value*, 3>;
  using ValueVector2 = std::array<llvm::Value*, 2>;

  using ConstantGenerator = std::function<llvm::Value*(int64_t)>;

  IrEmitterUnnested(const IrEmitterUnnested&) = delete;
  IrEmitterUnnested& operator=(const IrEmitterUnnested&) = delete;

  static std::unique_ptr<IrEmitterUnnested> Create(
      IrEmitterContext* ir_emitter_context);

  // Transfers the ownship of thunk_sequence_ out.
  std::unique_ptr<ThunkSequence> ConsumeThunkSequence() {
    return std::make_unique<ThunkSequence>(std::move(thunk_sequence_));
  }

  // Emits code for the given LMHLO region.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  Status EmitLmhloRegion(
      mlir::Region* region,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);

  // Emits code for the given HLO computation. Right now it is only used to emit
  // thunks for constructing command buffer. The plan is to replace
  // EmitLmhloRegion by this function altogether, after we support emitting
  // all instructions from HLO.
  Status EmitHloComputation(const HloComputation* computation);

  static void GetDependentDialects(mlir::DialectRegistry& registry);

 private:
  explicit IrEmitterUnnested(IrEmitterContext* ir_emitter_context);

  Status EmitUnreachable(mlir::Operation* op, std::string error_message);

  Status EmitCommandBufferThunk(const HloInstruction* instr);

  // IrEmitterUnnested handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  Status EmitConstant(mlir::Operation* op, const Literal& literal);

  Status EmitConditional(
      mlir::Operation* op,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);
  Status EmitConvolutionThunk(mlir::Operation* op);
  Status EmitGemmThunk(mlir::Operation* op);
#if GOOGLE_CUDA || TF_HIPBLASLT
  Status EmitCublasLtMatmulThunk(mlir::Operation* op);
#endif  // GOOGLE_CUDA || TF_HIPBLASLT
#if GOOGLE_CUDA
  Status EmitCublasLtMatmulThunkF8(mlir::Operation* op);
  Status EmitConvolutionReorderThunk(mlir::Operation* op);
  Status EmitNormThunk(mlir::Operation* op);
  StatusOr<FusionEmissionResult> EmitTritonFusion(
      const HloFusionAnalysis& hlo_fusion_analysis,
      const HloFusionInstruction* fusion, mlir::Operation* op);
  Status EmitFusedMHAThunk(mlir::Operation* op);
  Status EmitFusedMHABackwardThunk(mlir::Operation* op);
#endif  // GOOGLE_CUDA
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status EmitCubDeviceRadixSort(mlir::Operation* op);
  Status EmitCholeskyThunk(mlir::Operation* op);
  Status EmitCholeskyThunk(const HloInstruction* instr);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status EmitCustomCallThunk(mlir::Operation* op);
  Status EmitFftThunk(mlir::Operation* op);
  StatusOr<FusionEmissionResult> GetFusionEmissionResult(
      const HloFusionInstruction* instr, HloFusionAnalysis& fusion_analysis);
  Status EmitFusion(
      mlir::Operation* op,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);
  Status EmitFusion(const HloFusionInstruction* instr,
                    HloFusionAnalysis& fusion_analysis);
  Status EmitSelectAndScatter(
      mlir::Operation* op,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);
  Status EmitWhile(
      mlir::Operation* op,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);
  Status EmitInfeed(mlir::Operation* op);
  Status EmitOutfeed(mlir::Operation* op);
  Status EmitRngGetAndUpdateState(mlir::Operation* op);
  Status EmitSort(
      mlir::Operation* op,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status EmitTriangularSolveCustomCall(mlir::Operation* op);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  template <typename NcclThunkType, typename OpT>
  Status EmitNcclThunk(mlir::Operation* op);
  template <typename OpT>
  Status EmitNcclAsyncDone(Thunk::Kind kind, mlir::Operation* op);

  template <typename ThunkType, typename OpT>
  Status EmitReplicaOrPartitionId(mlir::Operation* op);

  template <typename NcclThunkType, typename OpT>
  Status EmitCollectivePermute(mlir::Operation* op);

  Status EmitOp(
      mlir::Operation* op,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);

  Status EmitHloInstruction(const HloInstruction* instr);

  static Thunk::ThunkInfo GetThunkInfo(mlir::Operation* op);

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Add a owning Thunk object to the thunk sequence.
  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
    thunk_sequence_.emplace_back(std::move(thunk));
  }

  // Load data from potentially unaligned address. If address is offset by
  // `alignment_bytes`, data is read in the unit of `alignment_bytes` to avoid
  // memory read misalignment in CUDA; otherwise, the entire data are loaded
  // from the given memory address.
  //
  //   address: the memory address to load data from.
  //   data_type: the type of data to load.
  //   alignment_bytes: the number of bytes required to align. The number of
  //     bytes of the data_type must be divisible by alignment_bytes.
  llvm::Value* CreateLoad(llvm::Value* address, llvm::Type* data_type,
                          int alignment_bytes);

  // Store data at a potentially unaligned address. If the address is offset by
  // `alignment_bytes`, data is stored in the unit of `alignment_bytes` to avoid
  // memory write misalignment in CUDA; otherwise, the entire data is stored at
  // the given memory address.
  //
  //   data: the data to be stored.
  //   address: the memory address to store data.
  //   alignment_bytes: the number of bytes required to align. The number of
  //     bytes of the data_type must be divisible by alignment_bytes.
  void CreateStore(llvm::Value* data, llvm::Value* address,
                   int alignment_bytes);

  // Input = {static array, dynamic_dim0, dynamic_dim1}
  // Output = {dynamic array(with dynamic dimension meta data at the end)}
  // For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
  // (`_` stands for padding)
  // Input = {{1,2,3,_,_,4,5,6_,_}, 2, 3}
  // Output = {{1,2,3,4,5,6,_,_,_,_,2,3}}

  // pseudo code for padToStatic on a 2d array
  //   ```
  // void padToStatic(int** input, int** output, int threads_per_block,
  //                  int meta_data_offset, int max_num_element,
  //                  int static_dim0_size, int static_dim1_size) {
  //   int* source_array = input[0];
  //   int* dest_array = output[0];

  //   // extract the dynamic dimension from the source array's metadata
  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);

  //   // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }

  //   int dyn_element_total = 1;
  //   dyn_element_total *= *dyn_dim0_size;
  //   dyn_element_total *= *dyn_dim1_size;
  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  //   return;
  // }
  //   ```
  Status EmitPadToStatic(mlir::Operation* op);

  // Input = {dynamic array(with dynamic dimension meta data at the end)}
  // Output = {static array, dynamic_dim0, dynamic_dim1}
  // For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
  // (`_` stands for padding)
  // Input = {{1,2,3,4,5,6,_,_,_,_,2,3}}
  // Output = {{1,2,3,_,_,4,5,6_,_}, 2, 3}

  // pseudo code for sliceToDynamic on a 2d array
  //   ```
  // void sliceToDynamic(int** input, int** output, int threads_per_block,
  //                  int meta_data_offset, int max_num_element,
  //                  int static_dim0_size, int static_dim1_size) {
  //   int* source_array = input[0];
  //   int* dest_array = output[0];

  //   // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset + sizeof(int);

  //   // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }

  //   int dyn_element_total = 1;
  //   dyn_element_total *= *dyn_dim0_size;
  //   dyn_element_total *= *dyn_dim1_size;
  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.dim1] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  //   return;
  // }
  //   ```
  Status EmitSliceToDynamic(mlir::Operation* op);

  StatusOr<BufferAllocation::Slice> GetAllocationSlice(mlir::Value v);
  StatusOr<std::vector<BufferAllocation::Slice>> GetAllocationSlices(
      mlir::OperandRange operands);

  int64_t ByteSizeOf(const Shape& shape) const {
    return llvm_ir::ByteSizeOf(
        shape, ir_emitter_context_->llvm_module()->getDataLayout());
  }

  // Structure describing a scatter operation for IR emission.
  // TODO(jurahul): Migrate element generators to use MLIR.
  //                Migrate update_computation to be an MLIR Region.
  struct ScatterDescriptor {
    std::string name;
    Shape operand_shape;
    Shape scatter_indices_shape;
    Shape updates_shape;
    ScatterDimensionNumbers dim_numbers;
    bool unique_indices;
    const HloComputation* update_computation;
    llvm_ir::IrArray output;
    llvm_ir::ElementGenerator scatter_indices_gen;
    llvm_ir::ElementGenerator updates_gen;
    std::function<llvm::Type*(int64_t)> get_index_type;
  };

  // Emits code for an in-place scatter using the provided scatter operation
  // description.
  Status EmitScatter(const ScatterDescriptor& desc,
                     const LaunchDimensions& launch_dimensions);

  StatusOr<FusionEmissionResult> EmitScatter(
      const HloFusionInstruction* fusion, mlir::lmhlo::FusionOp fusion_op,
      HloFusionAnalysis& fusion_analysis);

  // Emits kernel thunk for a custom fusion implemented with hand written custom
  // device kernels.
  StatusOr<FusionEmissionResult> EmitCustomFusion(
      const HloFusionInstruction* fusion, const CustomFusionConfig& config);

  // Builds a kernel thunk for a non-fusion operation, without reuse.
  //
  // All input and output tensors of `op` are passed to the kernel.
  //
  // TODO(tdanyluk): Consider also reusing non-fusion kernels.
  StatusOr<std::pair<std::vector<llvm_ir::IrArray> /*inputs*/,
                     std::vector<llvm_ir::IrArray> /*outputs*/>>
  BuildKernelThunkForNonFusionOp(mlir::Operation* op,
                                 const LaunchDimensions& launch_dimensions);

  // Builds a kernel thunk for a non-fusion operation, without reuse.
  //
  // Only the tensors specified in `needed_operands` are passed to the kernel.
  //
  // TODO(tdanyluk): Consider also reusing non-fusion kernels.
  StatusOr<std::pair<std::vector<llvm_ir::IrArray> /*inputs*/,
                     std::vector<llvm_ir::IrArray> /*outputs*/>>
  BuildKernelThunkForNonFusionOp(mlir::Operation* op,
                                 mlir::ValueRange needed_operands,
                                 const LaunchDimensions& launch_dimensions);

  Status BuildInitializerThunk(mlir::Operation* op, const HloInstruction* instr,
                               const HloInstruction* init_value,
                               mlir::Value init_value_mlir, mlir::Value dest);

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction 'hlo'.
  StatusOr<std::unique_ptr<Thunk>> BuildWhileThunk(
      mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);

  // Returns a ForThunk which executes 'loop_limit' invocations of a thunk
  // sequence from the 'body' sub-computation of the while instruction 'hlo'.
  StatusOr<std::unique_ptr<Thunk>> BuildForThunk(
      mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info,
      int64_t loop_limit,
      const absl::flat_hash_map<const mlir::Operation*, const HloInstruction*>&
          hlo_for_lmhlo);

  // Returns a ConditionalThunk which executes the thunk sequence for the
  // 'branch_computation' corresponding to the predicate/branch_index of the
  // given conditional instruction.
  StatusOr<std::unique_ptr<Thunk>> BuildConditionalThunk(
      const HloInstruction* conditional);

  Status AssertNonDeterminismIsOkay(const std::string& op_name);

  StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index) const;

  // The thunk sequence this IrEmitter generates for the input computation.
  ThunkSequence thunk_sequence_;

  // Maps async start ops to their executors so done can access the thunk.
  // Executor may be null if the start op is degenerate (so not emitted).
  absl::flat_hash_map<mlir::Operation*, NcclCollectiveThunk::AsyncExecutor*>
      async_executors_;

  // Begin optional members for XLA HLO -> LMHLO:
  absl::flat_hash_map<const mlir::Region*, std::unique_ptr<HloModule>>
      scratch_nested_computations_;
  // End optional members for XLA HLO -> LMHLO.

  // Returns the ShapedSlices for the given operands.
  StatusOr<std::vector<ShapedSlice>> GetShapedSlices(
      mlir::Operation::operand_range operands);

  GpuElementalIrEmitter elemental_emitter_;

  KernelReuseCache kernel_reuse_cache_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
