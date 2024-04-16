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
#ifndef XLA_SERVICE_HOST_OFFLOADER_H_
#define XLA_SERVICE_HOST_OFFLOADER_H_

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

class HloCostAnalysis;

// This pass does "host memory offloading". If a tensor is annotated to be moved
// to or from the host, this pass will remove the annotations and update each
// tensor's layout with host memory spaces and insert copies if necessary. This
// pass checks to make sure that no compute is done on the tensors annotated for
// host memory offload; if there is compute, it is considered a user error and
// an error will be returned.
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
  static absl::Span<const HloOpcode> GetAllowedPositionOpcodes() {
    return kAllowedPositionOpcodes;
  }

 private:
  const int64_t kHostMemorySpaceColor;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  absl::flat_hash_set<HloInstruction*> found_host_to_device_annotations_;
  absl::flat_hash_set<HloInstruction*> expected_host_to_device_annotations_;
  absl::flat_hash_set<HloInstruction*> custom_calls_to_remove_;
  absl::flat_hash_set<HloInstruction*> broadcasts_to_replace_;
  absl::flat_hash_set<HloPosition> positions_to_move_to_host_memory_;
  absl::flat_hash_set<HloInstruction*> annotations_for_copy_to_host_to_insert_;
  absl::flat_hash_set<HloInstruction*>
      annotations_for_copy_to_device_to_insert_;
  std::unique_ptr<CallGraph> call_graph_;

  // Positions of all HloValues of the given HloBuffer will be added to
  // positions_to_move_to_host_memory_.
  void AddAllPositionsToBeMovedToHostMemory(const HloBuffer& unique_buffer);

  absl::StatusOr<bool> TryParameterStreaming(HloInstruction* custom_call);
  absl::StatusOr<bool> TryOutputStreaming(HloInstruction* custom_call);
  Status HandleMoveToHostCustomCall(HloInstruction* custom_call);
  Status HandleMoveToDeviceCustomCall(HloInstruction* custom_call);

  // Handle memory-only offloading where the data is written to the host via a
  // dynamic-update-slice and is read back via a dynamic-slice.
  Status MemoryOnlyOffloadStartingWithDus(
      const HloInstruction* dynamic_update_slice);

  // Handle memory-only offloading where the data is written to the host via a
  // copy and is read back via a copy.
  Status MemoryOnlyOffloadStartingWithCopy(const HloInstruction* copy);

  // Handle memory-only offloading where there are no ops yet for data movement.
  // We will insert copies at the points where the annotations are.
  Status MemoryOnlyOffloadInsertCopies(HloInstruction* custom_call);

  Status DynamifySlice(HloInstruction* slice);

  static constexpr std::array kAllowedPositionOpcodes = {
      HloOpcode::kBitcast,
      HloOpcode::kGetTupleElement,
      HloOpcode::kOptimizationBarrier,
      HloOpcode::kParameter,
      HloOpcode::kTuple,
      HloOpcode::kWhile};
};

}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOADER_H_
