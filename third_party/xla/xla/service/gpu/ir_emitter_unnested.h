/* Copyright 2018 The OpenXLA Authors.

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
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "mlir/IR/Value.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/runtime/copy_thunk.h"
#include "xla/service/gpu/runtime/send_recv_thunk.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

#if TENSORFLOW_USE_ROCM
// for TF_HIPBLASLT
#include "rocm/rocm_config.h"
#endif

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
  std::unique_ptr<SequentialThunk> ConsumeThunkSequence() {
    return std::make_unique<SequentialThunk>(Thunk::ThunkInfo{},
                                             std::move(thunk_sequence_));
  }

  // Emits code for the given HLO computation.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  absl::Status EmitHloComputation(const HloComputation* computation);

 private:
  explicit IrEmitterUnnested(IrEmitterContext* ir_emitter_context);

  absl::Status EmitCommandBufferThunk(const HloInstruction* instr);

  // IrEmitterUnnested handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  absl::Status EmitConstant(const HloConstantInstruction* instr);

  absl::Status EmitConditional(const HloInstruction* instr);
  absl::Status EmitConvolutionThunk(const HloCustomCallInstruction* instr);
  absl::Status EmitGemmThunk(const HloCustomCallInstruction* instr);
#if GOOGLE_CUDA || TF_HIPBLASLT
  absl::Status EmitCublasLtMatmulThunk(const HloCustomCallInstruction* instr);
  absl::Status EmitCublasLtMatmulThunkF8(const HloCustomCallInstruction* instr);
#endif  // GOOGLE_CUDA || TF_HIPBLASLT
#if GOOGLE_CUDA
  absl::Status EmitConvolutionReorderThunk(
      const HloCustomCallInstruction* instr);
  absl::Status EmitNormThunk(const HloCustomCallInstruction* instr);
  absl::Status EmitFusedMHAThunk(const HloCustomCallInstruction* instr);
  absl::Status EmitFusedMHABackwardThunk(const HloCustomCallInstruction* instr);
#endif  // GOOGLE_CUDA
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  absl::Status EmitCubDeviceRadixSort(const HloCustomCallInstruction* instr);
  absl::Status EmitCholeskyThunk(const HloInstruction* instr);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  absl::Status EmitCustomCallThunk(const HloCustomCallInstruction* instr);
  absl::Status EmitFftThunk(const HloFftInstruction* instr);
  absl::Status EmitFusion(const HloFusionInstruction* instr);
  absl::Status EmitAsyncCustomCallStart(const HloInstruction* instr);
  absl::Status EmitSelectAndScatter(
      const HloSelectAndScatterInstruction* instr);
  absl::Status EmitWhile(const HloInstruction* instr);
  absl::Status EmitInfeed(const HloInfeedInstruction* instr);
  absl::Status EmitOutfeed(const HloOutfeedInstruction* instr);
  absl::Status EmitRngGetAndUpdateState(
      const HloRngGetAndUpdateStateInstruction* instr);

  absl::Status EmitSort(const HloSortInstruction* sort);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  absl::Status EmitTriangularSolveCustomCall(const HloInstruction* instr);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  absl::Status EmitTopKCustomCall(const HloCustomCallInstruction* instr);
  absl::Status EmitTritonCustomCall(const HloCustomCallInstruction* instr);

  absl::Status EmitSendThunk(const HloSendInstruction* instr);
  absl::Status EmitSendDoneThunk(const HloSendDoneInstruction* instr);

  absl::Status EmitRecvThunk(const HloRecvInstruction* instr);
  absl::Status EmitRecvDoneThunk(const HloRecvDoneInstruction* instr);

  template <typename NcclThunkType, typename HloInstType>
  absl::Status EmitNcclThunk(Thunk::Kind kind,
                             const HloInstruction* async_start,
                             const HloInstType* inst,
                             std::optional<bool> use_global_device_ids);

  absl::Status EmitNcclAsyncDone(Thunk::Kind kind, const HloInstruction* instr);

  template <typename ThunkType>
  absl::Status EmitReplicaOrPartitionId(const HloInstruction* instr);

  absl::Status EmitCollectiveBroadcast(
      const HloCollectiveBroadcastInstruction* instr);

  absl::Status EmitCollectivePermute(
      const HloCollectivePermuteInstruction* instr);

  absl::Status EmitCopyStartThunk(const HloCopyStartInstruction* instr);

  absl::Status EmitCopyDoneThunk(const HloInstruction* instr);

  absl::Status EmitHloInstruction(const HloInstruction* instr);

  absl::Status EmitTargetElementLoop(
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
  absl::Status EmitPadToStatic(const HloCustomCallInstruction* instr);

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
  absl::Status EmitSliceToDynamic(const HloCustomCallInstruction* instr);

  int64_t ByteSizeOf(const Shape& shape) const {
    return llvm_ir::ByteSizeOf(
        shape, ir_emitter_context_->llvm_module()->getDataLayout());
  }

  absl::StatusOr<std::pair<std::vector<llvm_ir::IrArray> /*inputs*/,
                           std::vector<llvm_ir::IrArray> /*outputs*/>>
  BuildKernelThunkForNonFusionOp(
      const HloInstruction* hlo,
      absl::Span<const HloInstruction* const> needed_operands,
      const LaunchDimensions& launch_dimensions);

  absl::Status BuildInitializerThunk(const HloInstruction* instr,
                                     const HloInstruction* init_value);

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction.
  absl::StatusOr<std::unique_ptr<Thunk>> BuildWhileThunk(
      const HloInstruction* instr, const Thunk::ThunkInfo& thunk_info,
      std::optional<int64_t> trip_count);

  // Returns a ConditionalThunk which executes the thunk sequence for the
  // 'branch_computation' corresponding to the predicate/branch_index of the
  // given conditional instruction.
  absl::StatusOr<std::unique_ptr<Thunk>> BuildConditionalThunk(
      const HloInstruction* conditional);

  absl::Status AssertNonDeterminismIsOkay(const std::string& op_name);

  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;

  CollectivesAsyncEvents& GetCollectivesAsyncEvents() {
    return ir_emitter_context_->collectives_async_events();
  }

  // The thunk sequence this IrEmitter generates for the input computation.
  ThunkSequence thunk_sequence_;

  // Container for async send/recv events shared by send/recv thunks.
  std::shared_ptr<SendRecvAsyncEvents> send_recv_events_;

  // Container for async copy-start/copy-done events.
  std::shared_ptr<CopyThunk::AsyncEvents> copy_events_;

  // Returns the ShapedSlices for the given operands.
  absl::StatusOr<std::vector<ShapedSlice>> GetShapedSlices(
      mlir::Operation::operand_range operands);

  GpuElementalIrEmitter elemental_emitter_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
