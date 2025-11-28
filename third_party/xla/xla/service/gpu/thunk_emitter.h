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

#ifndef XLA_SERVICE_GPU_THUNK_EMITTER_H_
#define XLA_SERVICE_GPU_THUNK_EMITTER_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

// Emits Thunks for the given HLO module.
class ThunkEmitter {
 public:
  absl::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  ThunkEmitter(const ThunkEmitter&) = delete;
  ThunkEmitter& operator=(const ThunkEmitter&) = delete;

  static std::unique_ptr<ThunkEmitter> Create(
      IrEmitterContext* ir_emitter_context);

  absl::StatusOr<ThunkSequence> EmitHloEntryComputation(
      const HloModule* module);

 private:
  explicit ThunkEmitter(IrEmitterContext* ir_emitter_context);

  // Emits code for the given HLO computation.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  absl::StatusOr<ThunkSequence> EmitHloComputation(
      const HloComputation* computation);

  absl::StatusOr<ThunkSequence> EmitCommandBufferThunk(
      const HloInstruction* instr);

  // ThunkEmitter handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  absl::Status EmitConstant(const HloConstantInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConditional(const HloInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitConvolutionThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitGemmThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunkF8(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitConvolutionReorderThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitNormThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitCuDnnThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitPtxCustomCall(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitCubDeviceRadixSort(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitCustomCallThunk(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitFftThunk(const HloFftInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitAsyncComputation(
      const HloInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitFusion(const HloFusionInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitCopy(const HloInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitAsyncCustomCallStart(
      const HloInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitWhile(const HloInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitInfeed(const HloInfeedInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitOutfeed(const HloOutfeedInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateState(
      const HloRngGetAndUpdateStateInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitSort(const HloSortInstruction* sort);
  absl::StatusOr<ThunkSequence> EmitTriangularSolveCustomCall(
      const HloInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitTopKCustomCall(
      const HloCustomCallInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitTritonCustomCall(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitSendThunk(const HloSendInstruction* instr,
                                              bool emit_group_thunks);
  absl::StatusOr<ThunkSequence> EmitSendDoneThunk(
      const HloSendDoneInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitRecvThunk(const HloRecvInstruction* instr,
                                              bool emit_group_thunks);
  absl::StatusOr<ThunkSequence> EmitRecvDoneThunk(
      const HloRecvDoneInstruction* instr);

  template <typename CollectiveThunkType, typename HloInstType>
  absl::StatusOr<ThunkSequence> EmitCollectiveThunk(
      Thunk::Kind kind, const HloInstruction* async_start,
      const HloInstType* inst, std::optional<bool> use_global_device_ids);

  absl::StatusOr<ThunkSequence> EmitCollectiveAsyncDone(
      Thunk::Kind kind, const HloInstruction* instr);

  template <typename NvshmemAllReduceThunkType,
            typename HloAllReduceInstruction>
  absl::StatusOr<ThunkSequence> EmitNvshmemThunk(
      Thunk::Kind kind, const HloInstruction* async_start,
      const HloAllReduceInstruction* inst,
      std::optional<bool> use_global_device_ids);

  absl::StatusOr<ThunkSequence> EmitNvshmemAsyncDone(
      Thunk::Kind kind, const HloInstruction* instr);

  template <typename HloInstType>
  absl::StatusOr<ThunkSequence> EmitDegeneratedCollectiveThunk(
      std::vector<CollectiveThunk::Buffer>& buffers,
      const HloInstruction* async_start, const HloInstType* inst);

  template <typename ThunkType>
  absl::StatusOr<ThunkSequence> EmitReplicaOrPartitionId(
      const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCollectiveMetadata(
      const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCollectivePermute(
      const HloCollectivePermuteInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCopyStartThunk(
      const HloCopyStartInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCopyDoneThunk(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitHloInstruction(
      const HloInstruction* instr, bool emit_group_thunks = false);

  absl::StatusOr<ThunkSequence> EmitCollectiveGroupStartThunk(
      const HloInstruction* instr);

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
  absl::StatusOr<ThunkSequence> EmitPadToStatic(
      const HloCustomCallInstruction* instr);

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
  absl::StatusOr<ThunkSequence> EmitSliceToDynamic(
      const HloCustomCallInstruction* instr);

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction.
  absl::StatusOr<std::unique_ptr<Thunk>> BuildWhileThunk(
      const HloInstruction* instr, const Thunk::ThunkInfo& thunk_info,
      std::optional<int64_t> trip_count);

  absl::Status AssertNonDeterminismIsOkay(const std::string& op_name);

  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;

  CollectivesAsyncEvents& GetCollectivesAsyncEvents() {
    return ir_emitter_context_->collectives_async_events();
  }

  InstructionToHostExecuteAsyncEvents&
  GetInstructionToHostExecuteAsyncEvents() {
    return ir_emitter_context_->instruction_to_host_execute_async_events();
  }
  IrEmitterContext* ir_emitter_context_;

  // Container for async host send/recv events shared by host send/recv thunks.
  std::shared_ptr<HostSendRecvAsyncEvents> send_recv_events_;

  // Container for async copy-start/copy-done events.
  std::shared_ptr<CopyThunk::AsyncEvents> copy_events_;

  // Shared buffer addresses registry for NVSHMEM put/get operations.
  std::shared_ptr<NvshmemBufferAddresses> nvshmem_buffer_addresses_;

  // Cache to store the call_graph.
  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_THUNK_EMITTER_H_
