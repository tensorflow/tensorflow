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
#ifndef XLA_HLO_TRANSFORMS_HOST_OFFLOADER_H_
#define XLA_HLO_TRANSFORMS_HOST_OFFLOADER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/host_offload_utils.h"

namespace xla {

class HloCostAnalysis;

// This pass does "host memory offloading". If a tensor is annotated to be moved
// to or from the host, this pass will remove the annotations and update each
// tensor's layout with host memory spaces and insert copies if necessary. This
// pass checks to make sure that no compute is done on the tensors annotated for
// host memory offload; if there is compute, it is considered a user error and
// an error will be returned.
// The pass will "walk down" the Hlo graph starting from either MoveToHost
// custom calls or from parameters with host memory space in their layout. All
// tensors along each path have their memory space set as host memory space. If
// a MoveToHost custom call is paired with a DynamicUpdateSlice, the
// DynamicUpdateSlice will write into host memory space. Otherwise, a copy from
// device to host will be inserted.
//
// If an output of a host offloaded computation is only used on host, the memory
// space of the usages are updated to reflect it and no copies to and from host
// are performed. Any MoveToHost instructions for outputs used only on host, are
// removed.
// TODO(b/347101407): A better approach could be to remove redundant copies in a
// generalized fashion. Should also be moved out of Host Offloader.
//
// All MoveToHost and MoveToDevice custom calls are removed by the end of this
// pass.
class HostOffloader : public HloModulePass {
 public:
  explicit HostOffloader(int64_t host_memory_space_color)
      : kHostMemorySpaceColor(host_memory_space_color) {}
  ~HostOffloader() override = default;

  absl::string_view name() const override { return "host-offloader"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Process the next "MoveToHost" instruction that resides at the beginning of
  // a host memory offload instruction chain. This ensures that redundant
  // "MoveToHost" (those already residing inside of a host memory offload
  // instruction chain) are ignored.
  absl::StatusOr<bool> ProcessNextMoveToHostInstr(HloComputation* computation);

  const int64_t kHostMemorySpaceColor;
  absl::flat_hash_set<HloInstruction*>
      already_visited_move_to_host_custom_calls_;
  absl::flat_hash_set<HloInstruction*> dynamic_update_slices_already_allocated_;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> copies_created_after_;
  absl::flat_hash_set<HloInstruction*> move_to_device_custom_calls_to_remove_;
  absl::flat_hash_set<host_offload_utils::InstructionAndShapeIndex>
      already_inserted_copy_before_;

  // Sometimes previous transformations turn a DynamicSlice into a Slice. Since
  // we're doing a DMA between the host and device, we need to turn the Slice
  // back into a DynamicSlice.
  absl::Status DynamifySlice(HloInstruction* slice);

  // Returns true if the instruction is allowed to be in the
  // middle of a path between a MoveToHost custom-call annotation and a
  // DynamicUpdateSlice. Ideally the custom-call should be immediately followed
  // by the DynamicUpdateSlice, but this is not always the case.
  bool InstructionIsAllowedBetweenMoveToHostAndDus(
      const HloInstruction* instruction) const;

  // Returns true if the instruction is allowed to be in the
  // middle of a path between a DynamicSlice and a MoveToDevice custom-call
  // annotation. Ideally the DynamicSlice should be immediately followed by the
  // custom-call, but this is not always the case.
  bool InstructionIsAllowedBetweenDsAndMoveToDevice(
      const HloInstruction* instruction) const;

  // Walks down the graph and does "host memory offloading" starting from every
  // host memory parameter in the entry computation.
  absl::StatusOr<bool> HandleInputStreaming(HloComputation* entry_computation);

  // Walks down the graph and does "host memory offloading" starting from every
  // MoveToHost custom call.
  absl::StatusOr<bool> HandleMoveToHostCustomCall(
      HloInstruction* custom_call_instruction);

  // Since we always walk the graph from the top down, this function only needs
  // to remove these lingering custom calls. This function should only be called
  // once all host memory offloading is done because multiple paths might lead
  // to the same MoveToDevice custom call. Removing it too early will confuse
  // subsequent walkings of the graph.
  absl::StatusOr<bool> HandleMoveToDeviceCustomCall(
      HloInstruction* custom_call_instruction);

  // DynamicUpdateSlices which write into host memory must have their
  // destination buffer allocated on the host. This function creates the
  // allocation and updates all positions to have host memory space.
  absl::Status CreateAllocateBufferForDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice);

  // One way to move data to the device is to use a Slice or DynamicSlice. This
  // function returns true if the slice is followed by a MoveToDevice custom
  // call.
  absl::StatusOr<bool> SliceLeadsToMoveToDeviceCustomCall(
      HloInstruction* slice);

  // Common function for doing the actual walking of the graph. Host memory
  // spaces are set and copies are inserted in here.
  absl::StatusOr<bool> WalkDownHostMemoryOffloadPaths(
      const host_offload_utils::InstructionAndShapeIndex&
          starting_instruction_and_index,
      bool insert_copy_before);

  // Given a custom call, this returns the first instruction and shape index to
  // start the host memory offload path from for each use of the custom call.
  absl::StatusOr<std::vector<host_offload_utils::InstructionAndShapeIndex>>
  GetStartingInstructions(HloInstruction* custom_call_instruction);

  // When a MoveToHost custom call is not paired with a DynamicUpdateSlice, a
  // copy from device to host must be inserted.
  absl::StatusOr<bool> InsertCopyBetween(
      const host_offload_utils::InstructionAndShapeIndex&
          before_instruction_and_index,
      const host_offload_utils::InstructionAndShapeIndex&
          after_instruction_and_index);

  // This is a fix for scheduling. Add copies to inputs of dynamic-update-slice
  // if the inserted value is directly a parameter of a computation. This is to
  // avoid cases in while loop where parameter/output aliasing can stop
  // scheduling because control-dependencies are added.
  absl::StatusOr<bool> ApplySchedulingFix(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Starting from the outputs of the host offloaded computation, track all
  // their usages. For the outputs that are ONLY used on host, remove redundant
  // copies to and from host, as well as update the memory space.
  absl::StatusOr<bool> HandleRedundantCopiesBackToHost(
      const HloModule* module, HloInstruction* instruction);
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_HOST_OFFLOADER_H_
