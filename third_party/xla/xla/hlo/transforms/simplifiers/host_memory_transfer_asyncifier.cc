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

#include "xla/hlo/transforms/simplifiers/host_memory_transfer_asyncifier.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

class HostMemoryTransferAsyncifierVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit HostMemoryTransferAsyncifierVisitor(int64_t host_memory_space_color)
      : kHostMemorySpaceColor(host_memory_space_color) {}
  bool Changed() const { return changed_; }

  absl::Status DefaultAction(HloInstruction* hlo_instruction) override {
    return absl::OkStatus();
  }

  // Replace all dynamic-slice ops which slice from host memory to device memory
  // with an asynchronous dynamic-slice.
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override {
    // Check that the dynamic_slice and its first operand have layouts. This
    // pass must only be run after LayoutAssignment.
    HloInstruction* dynamic_slice_operand = dynamic_slice->mutable_operand(0);
    if (!dynamic_slice->shape().has_layout()) {
      return InternalStrCat(dynamic_slice->name(), " does not have a layout.");
    }
    if (!dynamic_slice_operand->shape().has_layout()) {
      return InternalStrCat(dynamic_slice->name(), "'s operand, ",
                            dynamic_slice_operand->name(),
                            ", does not have a layout.");
    }

    VLOG(3) << absl::StreamFormat(
        "\"%s\" from S(%d) to S(%d)", dynamic_slice->name(),
        dynamic_slice_operand->shape().layout().memory_space(),
        dynamic_slice->shape().layout().memory_space());
    // Check that this is a dynamic-slice slicing from host memory to device
    // memory.
    if (dynamic_slice_operand->shape().layout().memory_space() !=
        kHostMemorySpaceColor) {
      // Only care about dynamic-slice from host memory.
      return absl::OkStatus();
    }
    if (dynamic_slice->shape().layout().memory_space() !=
        xla::Layout::kDefaultMemorySpace) {
      // Only care about dynamic-slice to device memory.
      return absl::OkStatus();
    }

    // Everything is as expected. Replace this dynamic-slice with the async
    // equivalent.
    const Shape context_shape = ShapeUtil::MakeScalarShape(U32);
    TF_ASSIGN_OR_RETURN(HloInstruction * async_done,
                        dynamic_slice->parent()->CreateAsyncInstructions(
                            dynamic_slice, {context_shape}));
    VLOG(1) << "DynamicSlice \"" << dynamic_slice->ToString()
            << "\" is slicing from host memory. Converting to async "
            << async_done->ToString();
    MarkAsChanged();
    return absl::OkStatus();
  }

  // Replace all dynamic-update-slice ops which update into host memory from
  // device memory with an asynchronous dynamic-update-slice.
  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override {
    // Check that the dynamic-update-slice and its first two operands have
    // layouts. This pass must only be run after LayoutAssignment.
    HloInstruction* dynamic_update_slice_operand =
        dynamic_update_slice->mutable_operand(0);
    HloInstruction* dynamic_update_slice_update =
        dynamic_update_slice->mutable_operand(1);
    if (!dynamic_update_slice->shape().has_layout()) {
      return InternalStrCat(dynamic_update_slice->name(),
                            " does not have a layout.");
    }
    if (!dynamic_update_slice_operand->shape().has_layout()) {
      return InternalStrCat(dynamic_update_slice->name(), "'s operand, ",
                            dynamic_update_slice_operand->name(),
                            ", does not have a layout.");
    }
    if (!dynamic_update_slice_update->shape().has_layout()) {
      return InternalStrCat(dynamic_update_slice->name(), "'s update, ",
                            dynamic_update_slice_update->name(),
                            ", does not have a layout.");
    }

    // Check that this is a dynamic-update-slice updating from device memory
    // into host memory.
    if (dynamic_update_slice_update->shape().layout().memory_space() !=
        xla::Layout::kDefaultMemorySpace) {
      // Only care about dynamic-update-slice from device memory.
      return absl::OkStatus();
    }
    if (dynamic_update_slice->shape().layout().memory_space() !=
        kHostMemorySpaceColor) {
      // Only care about dynamic-update-slice to host memory.
      return absl::OkStatus();
    }
    if (dynamic_update_slice_operand->shape().layout().memory_space() !=
        dynamic_update_slice->shape().layout().memory_space()) {
      return InternalStrCat(
          "Unexpected that ", dynamic_update_slice_operand->name(),
          "'s memory space is not the same as the dynamic-update-slice.");
    }

    // Everything is as expected. Replace this dynamic-update-slice with the
    // async equivalent.
    const Shape context_shape = ShapeUtil::MakeScalarShape(U32);
    TF_ASSIGN_OR_RETURN(HloInstruction * async_done,
                        dynamic_update_slice->parent()->CreateAsyncInstructions(
                            dynamic_update_slice, {context_shape}));
    VLOG(1) << "DynamicUpdateSlice \"" << dynamic_update_slice->ToString()
            << "\" is slicing into host memory space. Converting to async "
            << async_done->ToString();
    MarkAsChanged();
    return absl::OkStatus();
  }

  // Replace all copy ops which copy from host memory to device memory or from
  // device memory to host memory with an asynchronous copy.
  absl::Status HandleCopy(HloInstruction* copy) override {
    HloInstruction* operand = copy->mutable_operand(0);
    if (!operand->shape().has_layout()) {
      return InternalStrCat(operand->name(), " does not have a layout.");
    }
    if (!copy->shape().has_layout()) {
      return InternalStrCat(copy->name(), " does not have a layout.");
    }

    const auto copy_src_memory_space = operand->shape().layout().memory_space();
    const auto copy_dst_memory_space = copy->shape().layout().memory_space();
    if (!((copy_src_memory_space == kHostMemorySpaceColor &&
           copy_dst_memory_space == xla::Layout::kDefaultMemorySpace) ||
          (copy_src_memory_space == xla::Layout::kDefaultMemorySpace &&
           copy_dst_memory_space == kHostMemorySpaceColor))) {
      VLOG(2)
          << "Skipping copy because it is not a copy between device memory and "
             "host memory: "
          << copy->ToString();
      // Only care about copies between device memory and host memory.
      return absl::OkStatus();
    }

    // Everything is as expected. Replace this copy with the async equivalent.
    const Shape context_shape = ShapeUtil::MakeScalarShape(U32);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * async_done,
        copy->parent()->CreateAsyncInstructions(copy, {context_shape}));
    VLOG(1)
        << "Copy \"" << copy->name()
        << "\" is between device and host memory space. Converting to async "
        << async_done->ToString();
    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  const int64_t kHostMemorySpaceColor;
  bool changed_ = false;

  void MarkAsChanged() { changed_ = true; }
};

}  // namespace

absl::StatusOr<bool> HostMemoryTransferAsyncifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HostMemoryTransferAsyncifierVisitor visitor(kHostMemorySpaceColor);
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }
  return visitor.Changed();
}

}  // namespace xla
