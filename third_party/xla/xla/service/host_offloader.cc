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

#include "xla/service/host_offloader.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

using ::xla::host_memory_offload_annotations::kMoveToDeviceCustomCallTarget;
using ::xla::host_memory_offload_annotations::kMoveToHostCustomCallTarget;

void SetMemorySpace(Shape* shape, int64_t memory_space_color) {
  CHECK(shape->has_layout());
  shape->mutable_layout()->set_memory_space(memory_space_color);
}

// Checks if all of the HloPositions of this HloValue, apart from the defining
// position, are allowed when doing memory-only offload.
bool AllPositionsAreAllowed(const HloValue* value) {
  // Given an HloValue, validate that none of its positions are doing any
  // compute.
  for (const HloPosition& position : value->positions()) {
    if (position == value->defining_position()) {
      // Skip defining positions.
      continue;
    }
    // Check if this position is of an allowed type.
    if (!absl::c_linear_search(HostOffloader::GetAllowedPositionOpcodes(),
                               position.instruction->opcode())) {
      VLOG(1) << "Position " << position.instruction->ToString()
              << " is not supported.";
      return false;
    }
  }

  // Did not find any invalid ops.
  return true;
}

bool DefiningPositionIsAllowed(const HloInstruction* instruction) {
  static constexpr std::array kAllowedOpcodes = {HloOpcode::kWhile,
                                                 HloOpcode::kParameter};
  return absl::c_linear_search(kAllowedOpcodes, instruction->opcode());
}

template <typename MatcherType>
absl::StatusOr<HloInstruction*> BufferHasPositionWithUser(
    const HloBuffer& buffer, MatcherType matcher) {
  HloInstruction* result = nullptr;
  for (const HloValue* value : buffer.values()) {
    for (const HloPosition& position : value->positions()) {
      for (HloInstruction* user : position.instruction->users()) {
        if (Match(user, matcher)) {
          if (result != nullptr && result != user) {
            return Internal("Found multiple matching users! At least %s and %s",
                            result->name(), user->name());
          }
          result = user;
        }
      }
    }
  }
  return result;
}

template <typename MatcherType>
absl::StatusOr<std::vector<HloInstruction*>> GetBufferPositionsWithUser(
    const HloBuffer& buffer, MatcherType matcher) {
  std::vector<HloInstruction*> result;
  for (const HloValue* value : buffer.values()) {
    for (const HloPosition& position : value->positions()) {
      for (HloInstruction* user : position.instruction->users()) {
        if (Match(user, matcher)) {
          result.emplace_back(user);
        }
      }
    }
  }
  return result;
}

template <typename MatcherType>
absl::StatusOr<std::vector<HloInstruction*>> GetBufferUsersOfType(
    const HloBuffer& buffer, MatcherType matcher) {
  std::vector<HloInstruction*> result;
  for (const HloValue* value : buffer.values()) {
    VLOG(3) << "Buffer defined at " << value->defining_instruction()->name()
            << " has positions ["
            << absl::StrJoin(value->positions(), ", ",
                             [](std::string* out, const HloPosition& position) {
                               out->append(position.instruction->name());
                             })
            << "]";
    for (const HloPosition& position : value->positions()) {
      VLOG(4) << "  Position " << position.instruction->name() << " has users ["
              << absl::StrJoin(
                     position.instruction->users(), ", ",
                     [](std::string* out, const HloInstruction* user) {
                       out->append(user->name());
                     })
              << "]";
      for (HloInstruction* user : position.instruction->users()) {
        if (Match(user, matcher)) {
          result.emplace_back(user);
        }
      }
    }
  }
  return result;
}

// Returns true if the instruction passed in preserves the underlying buffer,
// And the buffer is passed through the first operand.
// This is used to trace the graph between an annotation and its relevant slice.
bool CanTraverseOpBetweenAnnotation(HloInstruction* hlo) {
  if (hlo->opcode() == HloOpcode::kBitcast ||
      hlo->opcode() == HloOpcode::kCopy) {
    return true;
  } else if (hlo->opcode() == HloOpcode::kReshape) {
    return ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(), hlo->shape());
  } else if (hlo->opcode() == HloOpcode::kReduce) {
    // TODO(b/333902007): Remove this once trivial reduces no longer appear.
    return ShapeUtil::TrueRank(hlo->operand(0)->shape()) ==
           ShapeUtil::TrueRank(hlo->shape());
  }
  return false;
}

// Starting from a slice or dynamic-slice, trace the graph down through reshapes
// and bitcasts to find the annotation that signals that the data is being moved
// to the device from the host. If no custom call is found, returns an empty
// optional.
std::optional<HloInstruction*> FindAnnotationFromDS(HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kDynamicSlice ||
        hlo->opcode() == HloOpcode::kSlice)
      << "Expected a dynamic-slice or slice as input.";
  if (hlo->user_count() != 1) {
    return std::nullopt;
  }
  hlo = hlo->users()[0];
  while (!hlo->IsCustomCall(kMoveToDeviceCustomCallTarget)) {
    if (!CanTraverseOpBetweenAnnotation(hlo) || hlo->user_count() != 1) {
      break;
    }
    hlo = hlo->users()[0];
  }
  if (hlo->IsCustomCall(kMoveToDeviceCustomCallTarget)) {
    return hlo;
  }
  return std::nullopt;
}

// Starting from a MoveToHost custom call, trace the graph down through reshapes
// and bitcasts to return the dynamic-update-slice that moves the data from the
// host to the device. If no DUS is found, returns an empty optional.
std::optional<HloInstruction*> FindDUSFromAnnotation(HloInstruction* hlo) {
  CHECK(hlo->IsCustomCall(kMoveToHostCustomCallTarget))
      << "Expected a MoveToHost custom call as input.";
  if (hlo->user_count() != 1) {
    return std::nullopt;
  }
  hlo = hlo->users()[0];
  while (hlo->opcode() != HloOpcode::kDynamicUpdateSlice) {
    if (!CanTraverseOpBetweenAnnotation(hlo) || hlo->user_count() != 1) {
      break;
    }
    hlo = hlo->users()[0];
  }
  if (hlo->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return hlo;
  }
  return std::nullopt;
}

// Starting from a dynamic-update-slice, trace the graph up reshapes,
// bitcasts and reduces to return the MoveToHost custom call that feeds into the
// DUS, if it exists. If no MoveToHost call is found, returns an empty optional.
std::optional<HloInstruction*> FindAnnotationFromDUS(HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kDynamicUpdateSlice)
      << "Expected a dynamic-update-slice as input.";
  // We expect the custom call to come from the written slice, i.e. operand 1.
  hlo = hlo->mutable_operand(1);
  while (!hlo->IsCustomCall(kMoveToHostCustomCallTarget)) {
    if (!CanTraverseOpBetweenAnnotation(hlo)) {
      break;
    }
    hlo = hlo->mutable_operand(0);
  }
  if (hlo->IsCustomCall(kMoveToHostCustomCallTarget)) {
    return hlo;
  }
  return std::nullopt;
}

}  // namespace

absl::Status HostOffloader::HandleMoveToHostCustomCall(
    HloInstruction* custom_call) {
  VLOG(2) << "Found a custom call annotating start-of-host-offload: "
          << custom_call->ToString();
  // Save a pointer to this custom call for when we want to remove it later.
  custom_calls_to_remove_.emplace(custom_call);

  // We expect that either the custom call is the root or the DUS is the only
  // user of this custom call.
  if (!custom_call->IsRoot() && custom_call->user_count() != 1) {
    return FailedPrecondition(
        "Expecting custom call %s to either be the root or only have 1 user; "
        "it is not the root and has %d users: [%s]",
        custom_call->name(), custom_call->user_count(),
        absl::StrJoin(custom_call->users(), ", ",
                      [](std::string* out, const HloInstruction* user) {
                        out->append(user->name());
                      }));
  }

  HloInstruction* consumer = nullptr;
  if (!custom_call->IsRoot()) {
    consumer = custom_call->users().at(0);
    // Skip past any bitcasts.
    while (consumer != nullptr && consumer->opcode() == HloOpcode::kBitcast) {
      VLOG(1) << "Skipping bitcast " << consumer->ToString();
      consumer = consumer->users().at(0);
    }
  }

  if (consumer != nullptr &&
      consumer->opcode() == HloOpcode::kDynamicUpdateSlice) {
    TF_RETURN_IF_ERROR(MemoryOnlyOffloadStartingWithDus(consumer));
  } else if (consumer != nullptr && consumer->opcode() == HloOpcode::kCopy) {
    TF_RETURN_IF_ERROR(MemoryOnlyOffloadStartingWithCopy(consumer));
  } else {
    TF_RETURN_IF_ERROR(MemoryOnlyOffloadInsertCopies(custom_call));
  }
  return OkStatus();
}

absl::Status HostOffloader::MemoryOnlyOffloadStartingWithDus(
    const HloInstruction* dynamic_update_slice) {
  // The user wants to offload the data defined by this dynamic-update-slice.
  VLOG(2) << "Host memory offload starts with a dynamic-update-slice: "
          << dynamic_update_slice->name();
  // Get the buffer for this DUS.
  const HloBuffer& unique_buffer =
      alias_analysis_->GetUniqueBufferAt(dynamic_update_slice);

  // We must find at least two HloValues:
  //  1. Defined by a broadcast.
  //    a. For now, we only offload if the original destination of DUS is
  //    created by a broadcast.
  //  2. Defined by a dynamic-update-slice.
  const HloValue* dus_value = nullptr;
  const HloValue* broadcast_value = nullptr;
  for (const HloValue* value : unique_buffer.values()) {
    HloInstruction* defining_instruction =
        value->defining_position().instruction;
    if (defining_instruction->opcode() == HloOpcode::kBroadcast) {
      if (broadcast_value != nullptr) {
        LOG(WARNING) << "Already found one broadcast ("
                     << broadcast_value->defining_position().instruction->name()
                     << ") value for this buffer. This one is "
                     << defining_instruction->name();
      }
      broadcast_value = value;
    } else if (defining_instruction->opcode() ==
               HloOpcode::kDynamicUpdateSlice) {
      if (dus_value != nullptr) {
        LOG(WARNING) << "Already found one dynamic-update-slice ("
                     << dus_value->defining_position().instruction->name()
                     << ") value for this buffer. This one is "
                     << defining_instruction->name();
      }
      dus_value = value;
    } else {
      // For all values other than the two we were looking for, ensure that the
      // defining position is non-compute as well as all other positions.
      if (!DefiningPositionIsAllowed(value->defining_position().instruction)) {
        return Internal(
            "HloValue is defined by an unsupported op: %s. HloValue: %s",
            defining_instruction->name(), value->ToString());
      }
      if (!AllPositionsAreAllowed(value)) {
        return Internal(
            "HloValue defined by %s has an invalid position. HloValue: %s",
            defining_instruction->name(), value->ToString());
      }
    }
  }

  // For the two found HloValues, ensure that all other positions are
  // non-compute.
  if (dus_value == nullptr) {
    return Internal(
        "DynamicUpdateSlice's buffer does not have a value which is defined by "
        "a dynamic update slice. HloBuffer: %s",
        unique_buffer.ToString());
  }
  if (!AllPositionsAreAllowed(dus_value)) {
    return Internal(
        "HloValue defined by %s has an invalid position. HloValue: %s",
        dus_value->defining_position().instruction->name(),
        dus_value->ToString());
  }
  if (broadcast_value == nullptr) {
    return Internal(
        "DynamicUpdateSlice's buffer does not have a value which is defined by "
        "a broadcast. HloBuffer: %s",
        unique_buffer.ToString());
  }
  if (!AllPositionsAreAllowed(broadcast_value)) {
    return Internal(
        "HloValue defined by %s has an invalid position. HloValue: %s",
        broadcast_value->defining_position().instruction->name(),
        broadcast_value->ToString());
  }

  // TODO(b/319681297): Further analyze the HloValue defined by the broadcast.
  // Make sure that nothing is expecting the result of the broadcast, as we'll
  // be replacing it.

  // Check that this buffer is finally an input to at least one slice or
  // dynamic-slice.
  TF_ASSIGN_OR_RETURN(
      std::vector<HloInstruction*> consuming_slices,
      GetBufferUsersOfType(
          unique_buffer,
          match::AnyOf<HloInstruction>(match::Slice(), match::DynamicSlice())));
  VLOG(2) << dynamic_update_slice->name()
          << " is consumed by [dynamic-]slices: ["
          << absl::StrJoin(consuming_slices, ", ",
                           [](std::string* out, const HloInstruction* inst) {
                             out->append(inst->name());
                           })
          << ']';
  if (!dus_for_streamed_buffer_.contains(dynamic_update_slice) &&
      consuming_slices.empty()) {
    return Internal(
        "The dynamic-update-slice (%s) never feeds into a slice nor "
        "dynamic-slice.",
        dynamic_update_slice->name());
  }

  // Each dynamic_slice and slice should feed into another annotation.
  for (HloInstruction* consuming_slice : consuming_slices) {
    VLOG(1) << "Host data produced by " << dynamic_update_slice->name()
            << " is consumed by " << consuming_slice->name();
    if (consuming_slice->user_count() != 1) {
      return Internal(
          "Slice/Dynamic-slice %s should only have one user. It should be an "
          "annotation "
          "to load the data back on the device. Instead, it has users [%s]",
          consuming_slice->name(),
          absl::StrJoin(consuming_slice->users(), ", ",
                        [](std::string* out, const HloInstruction* inst) {
                          out->append(inst->name());
                        }));
    }
    std::optional<HloInstruction*> slice_user =
        FindAnnotationFromDS(consuming_slice);
    if (!slice_user.has_value()) {
      return Internal(
          "Slice/Dynamic-slice %s does not have a matching annotation.",
          consuming_slice->name());
    }

    HloInstruction* consuming_slice_user = slice_user.value();
    if (consuming_slice_user->custom_call_target() !=
        host_memory_offload_annotations::kMoveToDeviceCustomCallTarget) {
      return Internal(
          "Found custom-call (%s) is not the expected matching host offload "
          "annotation",
          consuming_slice_user->name());
    }
    expected_host_to_device_annotations_.emplace(consuming_slice_user);
  }

  // Save the broadcast to later be replaced with a
  // custom-call("AllocateBuffer")
  broadcasts_to_replace_.emplace(
      broadcast_value->defining_position().instruction);
  AddAllPositionsToBeMovedToHostMemory(unique_buffer);
  return OkStatus();
}

void HostOffloader::AddAllPositionsToBeMovedToHostMemory(
    const HloBuffer& unique_buffer) {
  for (const HloValue* value : unique_buffer.values()) {
    for (const HloPosition& position : value->positions()) {
      positions_to_move_to_host_memory_.emplace(position);
    }
  }
}

absl::Status HostOffloader::MemoryOnlyOffloadStartingWithCopy(
    const HloInstruction* copy) {
  // The user wants to offload the data defined by this copy.
  VLOG(2) << "Host memory offload starts with a copy: " << copy->name();

  // Get the buffer for this copy.
  const HloBuffer& unique_buffer = alias_analysis_->GetUniqueBufferAt(copy);

  // Look for a value defined by a copy.
  const HloValue* copy_value = nullptr;
  for (const HloValue* value : unique_buffer.values()) {
    HloInstruction* defining_instruction =
        value->defining_position().instruction;
    if (defining_instruction->opcode() == HloOpcode::kCopy) {
      if (copy_value != nullptr) {
        LOG(WARNING)
            << "Already found one dynamic-update-slice value for this buffer";
      }
      copy_value = value;
    } else {
      // For all other values (that aren't defined by a copy), ensure that the
      // defining position is non-compute as well as all other positions.
      if (!DefiningPositionIsAllowed(value->defining_position().instruction)) {
        return Internal(
            "HloValue is defined by an unsupported op: %s. HloValue: %s",
            defining_instruction->name(), value->ToString());
      }
      if (!AllPositionsAreAllowed(value)) {
        return Internal(
            "HloValue defined by %s has an invalid position. HloValue: %s",
            defining_instruction->name(), value->ToString());
      }
    }
  }

  if (copy_value == nullptr) {
    return Internal(
        "Copy's buffer does not have a value which is defined by a copy. "
        "HloBuffer: %s",
        unique_buffer.ToString());
  }
  // For the copy, ensure that all other positions are non-compute.
  if (!AllPositionsAreAllowed(copy_value)) {
    return Internal(
        "HloValue defined by %s has an invalid position. HloValue: %s",
        copy_value->defining_position().instruction->name(),
        copy_value->ToString());
  }

  // Check that this buffer is finally an input to another copy.
  TF_ASSIGN_OR_RETURN(HloInstruction * consuming_copy,
                      BufferHasPositionWithUser(unique_buffer, match::Copy()));
  if (consuming_copy == nullptr) {
    return Internal("The copy (%s) never feeds into another copy.",
                    copy->name());
  }

  // The copy should feed into another annotation.
  if (consuming_copy->user_count() != 1) {
    return Internal(
        "Copy should only have one user. It should be an annotation to load "
        "the data back on the device. Instead, it has users [%s]",
        absl::StrJoin(consuming_copy->users(), ", ",
                      [](std::string* out, const HloInstruction* inst) {
                        out->append(inst->name());
                      }));
  }
  HloInstruction* consuming_copy_user = consuming_copy->users()[0];
  if (consuming_copy_user->opcode() != HloOpcode::kCustomCall) {
    return Internal("Copy does not have a matching annotation.");
  }
  if (consuming_copy_user->custom_call_target() !=
      host_memory_offload_annotations::kMoveToDeviceCustomCallTarget) {
    return Internal(
        "Found custom-call is not the expected matching host offload "
        "annotation");
  }
  expected_host_to_device_annotations_.emplace(consuming_copy_user);

  AddAllPositionsToBeMovedToHostMemory(unique_buffer);
  return OkStatus();
}

absl::Status HostOffloader::MemoryOnlyOffloadInsertCopies(
    HloInstruction* custom_call) {
  VLOG(3) << "Found an offload annotation (" << custom_call->name()
          << "). Expecting that we'll need to insert copies";
  const HloBuffer& unique_buffer =
      alias_analysis_->GetUniqueBufferAt(custom_call);
  for (const HloValue* value : unique_buffer.values()) {
    HloInstruction* defining_instruction =
        value->defining_position().instruction;
    if (!AllPositionsAreAllowed(value)) {
      return Internal(
          "HloValue defined by %s has an invalid position. HloValue: %s",
          defining_instruction->name(), value->ToString());
    }
  }

  // Check that this buffer is finally an input to a load-from-host custom-call.
  TF_ASSIGN_OR_RETURN(
      std::vector<HloInstruction*> matching_annotations,
      GetBufferPositionsWithUser(
          unique_buffer,
          match::CustomCall({host_memory_offload_annotations::
                                 kMoveToDeviceCustomCallTarget})));

  // This fits the pattern that we're looking for. Save these annotations to
  // later insert copies around.
  annotations_for_copy_to_host_to_insert_.emplace(custom_call);
  for (HloInstruction* matching_annotation : matching_annotations) {
    annotations_for_copy_to_device_to_insert_.emplace(matching_annotation);

    // Save the matching annotation to later be removed.
    expected_host_to_device_annotations_.emplace(matching_annotation);
  }

  AddAllPositionsToBeMovedToHostMemory(unique_buffer);
  return OkStatus();
}

absl::Status HostOffloader::DynamifySlice(HloInstruction* slice) {
  VLOG(3) << "Dynamifying slice " << slice->ToString();
  std::vector<HloInstruction*> start_constants;
  for (int64_t start : slice->slice_starts()) {
    HloInstruction* constant = slice->parent()->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(start)));
    start_constants.push_back(constant);
  }
  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(slice->slice_limits().size());
  for (int i = 0; i < slice->slice_limits().size(); ++i) {
    slice_sizes.push_back(slice->slice_limits()[i] - slice->slice_starts()[i]);
  }
  HloInstruction* new_ds =
      slice->parent()->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice->shape(), slice->mutable_operand(0), start_constants,
          slice_sizes));
  VLOG(3) << "Newly created dynamic slice: " << new_ds->name();
  TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(new_ds));
  TF_RETURN_IF_ERROR(slice->parent()->RemoveInstruction(slice));
  return OkStatus();
}

// Taking an instruction representing a move-to-device custom call, creates a
// copy to device for that operand and replaces all uses of the operand of the
// load annotation with the copy.
absl::Status HostOffloader::CreateCopyForInputStreaming(
    HloInstruction* custom_call) {
  HloInstruction* operand_of_load_annotation = custom_call->mutable_operand(0);
  Shape copy_shape = operand_of_load_annotation->shape();
  SetMemorySpace(&copy_shape, Layout::kDefaultMemorySpace);
  HloInstruction* copy_to_device =
      custom_call->parent()->AddInstruction(HloInstruction::CreateUnary(
          copy_shape, HloOpcode::kCopy, operand_of_load_annotation));

  auto users = operand_of_load_annotation->users();
  for (HloInstruction* use : users) {
    if (use == copy_to_device) {
      continue;
    }
    auto callers = call_graph_->GetComputationCallers(copy_to_device->parent());
    if (callers.size() > 1) {
      return absl::InvalidArgumentError(
          "Expected to be called only by one caller");
    }
    if (callers.size() == 1 && callers[0]->opcode() == HloOpcode::kWhile &&
        use->opcode() == HloOpcode::kTuple && use->IsRoot()) {
      // Need some special filtering for while body's root instruction.
      for (int i = 0; i < use->operands().size(); i++) {
        if (use->operands()[i] == operand_of_load_annotation) {
          if (operand_of_load_annotation->opcode() ==
                  HloOpcode::kGetTupleElement &&
              operand_of_load_annotation->operand(0)->opcode() ==
                  HloOpcode::kParameter &&
              operand_of_load_annotation->tuple_index() == i) {
            // A special case where move-to-device is put into the result
            // tuple element at the same index as where the move-to-device
            // gets the data from. In this case, while loop's result tuple
            // should not use move-to-device since at loop entry it's still
            // on host.
            continue;
          }
          TF_RETURN_IF_ERROR(operand_of_load_annotation->ReplaceUseWith(
              use, i, copy_to_device));
        }
      }
    } else {
      TF_RETURN_IF_ERROR(
          operand_of_load_annotation->ReplaceUseWith(use, copy_to_device));
    }
  }
  return OkStatus();
}

// From a unique buffer on host memory, finds move-to-device custom calls
// for this buffer and inserts the appropriate copies.
absl::Status HostOffloader::HandleStreamedBuffer(
    const HloBuffer& unique_buffer) {
  // Find all move-to-device custom calls that are using this buffer.
  for (const HloValue* value : unique_buffer.values()) {
    // First, handle the defining instruction of this buffer, as a potential
    // move-to-host custom call.
    if (value->defining_instruction()->IsCustomCall(
            kMoveToHostCustomCallTarget)) {
      annotations_for_copy_to_host_to_insert_.emplace(
          value->defining_instruction());
      AddAllPositionsToBeMovedToHostMemory(unique_buffer);
    } else if (value->defining_instruction()->opcode() ==
               HloOpcode::kDynamicUpdateSlice) {
      std::optional<HloInstruction*> dus =
          FindAnnotationFromDUS(value->defining_instruction());
      if (dus.has_value()) {
        dus_for_streamed_buffer_.emplace(value->defining_instruction());
        AddAllPositionsToBeMovedToHostMemory(unique_buffer);
      }
    }
    // Next, handle uses of this buffer as potential move-to-device custom
    // calls.
    for (const HloUse& use : value->GetUses()) {
      if (use.instruction->IsCustomCall(
              host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
        HloInstruction* move_to_device_custom_call = use.instruction;

        // Create a copy to device for the move-to-device custom call. Mark
        // the move-to-device custom call as expected.
        TF_RETURN_IF_ERROR(
            CreateCopyForInputStreaming(move_to_device_custom_call));
        expected_host_to_device_annotations_.emplace(
            move_to_device_custom_call);
      } else if (use.instruction->opcode() == HloOpcode::kDynamicSlice ||
                 use.instruction->opcode() == HloOpcode::kSlice) {
        std::optional<HloInstruction*> move_to_device_custom_call =
            FindAnnotationFromDS(use.instruction);
        if (move_to_device_custom_call.has_value()) {
          TF_RETURN_IF_ERROR(
              CreateCopyForInputStreaming(move_to_device_custom_call.value()));
          expected_host_to_device_annotations_.emplace(
              move_to_device_custom_call.value());
        }
      }
    }
  }
  AddAllPositionsToBeMovedToHostMemory(unique_buffer);
  return OkStatus();
}

// Finds parameters of the entry computation that are in host memory space and
// corresponding move-to-device custom calls for these parameters. Once found,
// adds these move-to-device custom calls to the expected host-to-device
// annotations, and creates the necessary copies for input streaming.
absl::Status HostOffloader::HandleInputStreaming(HloComputation* computation) {
  const ComputationLayout& entry_computation_layout =
      computation->parent()->entry_computation_layout();

  for (int i = 0; i < entry_computation_layout.parameter_count(); ++i) {
    if (entry_computation_layout.parameter_shape(i).IsToken()) {
      LOG(WARNING) << "Token parameters are not supported for streaming.";
      continue;
    }
    ShapeUtil::ForEachSubshape(
        entry_computation_layout.parameter_shape(i),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.has_layout() &&
              subshape.layout().memory_space() == kHostMemorySpaceColor) {
            VLOG(4) << "Handling streamed element in input with shape: "
                    << subshape.ToString(true);
            const HloBuffer& unique_buffer = alias_analysis_->GetUniqueBufferAt(
                computation->parameter_instruction(i), {index});
            TF_CHECK_OK(HandleStreamedBuffer(unique_buffer));
          }
        });
  }
  return OkStatus();
}

// Starts from the result of the entry computation and looks for a case of
// output streaming. This function will not change any hlo, it will only mark
// instructions to be converted to host memory space.
absl::Status HostOffloader::HandleOutputStreaming(HloComputation* computation) {
  const ComputationLayout& entry_computation_layout =
      computation->parent()->entry_computation_layout();

  ShapeUtil::ForEachSubshape(
      entry_computation_layout.result_shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.has_layout() &&
            subshape.layout().memory_space() == kHostMemorySpaceColor) {
          VLOG(4) << "Handling streamed element in result with shape: "
                  << subshape.ToString(true);
          const HloBuffer& unique_buffer = alias_analysis_->GetUniqueBufferAt(
              computation->root_instruction(), {index});
          TF_CHECK_OK(HandleStreamedBuffer(unique_buffer));
        }
      });
  return OkStatus();
}

absl::StatusOr<bool> HostOffloader::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  call_graph_ = CallGraph::Build(module);

  // Run HloAliasAnalysis on module.
  TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));

  TF_RETURN_IF_ERROR(HandleInputStreaming(module->entry_computation()));
  TF_RETURN_IF_ERROR(HandleOutputStreaming(module->entry_computation()));

  // Iterate over all instructions and look for XLA host offload annotations.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      if (instruction->custom_call_target() ==
          host_memory_offload_annotations::kMoveToHostCustomCallTarget) {
        TF_RETURN_IF_ERROR(HandleMoveToHostCustomCall(instruction));
      } else if (instruction->custom_call_target() ==
                 host_memory_offload_annotations::
                     kMoveToDeviceCustomCallTarget) {
        found_host_to_device_annotations_.emplace(instruction);
      }
    }
  }

  // Insert copies to the host for the saved annotations.
  for (HloInstruction* to_host_annotation :
       annotations_for_copy_to_host_to_insert_) {
    HloInstruction* data_to_host = to_host_annotation->mutable_operand(0);
    // Create a copy (to host) of the first and only operand to the given custom
    // call.
    HloInstruction* copy_to_host =
        data_to_host->parent()->AddInstruction(HloInstruction::CreateUnary(
            data_to_host->shape(), HloOpcode::kCopy, data_to_host));
    // Replace all uses of the to-host annotation with the first copy.
    TF_RETURN_IF_ERROR(to_host_annotation->ReplaceAllUsesWith(copy_to_host));
    // Also save the position of the newly created copy-to-host to later have
    // its memory space updated.
    positions_to_move_to_host_memory_.emplace(HloPosition{copy_to_host});
  }

  // Insert copies to the device for the saved annotations.
  for (HloInstruction* to_device_annotation :
       annotations_for_copy_to_device_to_insert_) {
    HloInstruction* data_to_device = to_device_annotation->mutable_operand(0);
    // Create another copy (back to device) of that copy.
    HloInstruction* copy_to_device =
        data_to_device->parent()->AddInstruction(HloInstruction::CreateUnary(
            data_to_device->shape(), HloOpcode::kCopy, data_to_device));
    // Replace all uses of the to-device annotation with the second copy.
    TF_RETURN_IF_ERROR(
        to_device_annotation->ReplaceAllUsesWith(copy_to_device));
  }

  // Check that we found all the annotations that we expected.
  if (found_host_to_device_annotations_ !=
      expected_host_to_device_annotations_) {
    return Internal(
        "There is a mismatch between the expected host-to-device annotations "
        "(%s) and the found host-to-device annotations (%s)",
        absl::StrJoin(expected_host_to_device_annotations_, ", ",
                      [](std::string* str, HloInstruction* instr) {
                        str->append(instr->name());
                      }),
        absl::StrJoin(found_host_to_device_annotations_, ", ",
                      [](std::string* str, HloInstruction* instr) {
                        str->append(instr->name());
                      }));
  }

  // Remove these host-to-device annotations.
  for (HloInstruction* instr : found_host_to_device_annotations_) {
    custom_calls_to_remove_.emplace(instr);
  }

  absl::flat_hash_set<HloInstruction*> slices_to_dynamify;
  // Change the memory space of these positions to the host memory space.
  for (const HloPosition& position : positions_to_move_to_host_memory_) {
    // If a user of this position is a slice, change it to be a dynamic-slice.
    for (HloInstruction* user : position.instruction->users()) {
      if (user->opcode() == HloOpcode::kSlice) {
        slices_to_dynamify.emplace(user);
      }
    }
    Shape* shape_to_change = ShapeUtil::GetMutableSubshape(
        position.instruction->mutable_shape(), position.index);
    VLOG(2) << "Setting instruction to have host memory space: "
            << position.instruction->name();
    SetMemorySpace(shape_to_change, kHostMemorySpaceColor);
    changed = true;
  }

  for (HloInstruction* user : slices_to_dynamify) {
    TF_RETURN_IF_ERROR(DynamifySlice(user));
  }

  // Replace these broadcasts with AllocateBuffer instructions for host memory.
  for (HloInstruction* broadcast : broadcasts_to_replace_) {
    HloInstruction* allocate_buffer =
        broadcast->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            broadcast->shape(), {}, "AllocateBuffer"));
    VLOG(2) << "Replacing broadcast " << broadcast->name()
            << " with AllocateBuffer " << allocate_buffer->ToString();
    SetMemorySpace(allocate_buffer->mutable_shape(), kHostMemorySpaceColor);
    CHECK_OK(broadcast->ReplaceAllUsesWith(allocate_buffer));
    TF_RETURN_IF_ERROR(broadcast->parent()->RemoveInstruction(broadcast));
    changed = true;
  }

  // Recompute alias analysis after changes.
  TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));
  auto uses_parameter_buffer = [this](HloInstruction* hlo) {
    for (const HloBuffer* buffer : alias_analysis_->ComputeBuffersAt(hlo)) {
      for (const HloValue* value : buffer->values()) {
        for (const HloPosition& pos : value->positions()) {
          if (absl::c_linear_search(hlo->parent()->parameter_instructions(),
                                    pos.instruction)) {
            return true;
          }
        }
      }
    }
    return false;
  };
  // Remove these custom-calls that were previously used for annotation.
  for (HloInstruction* custom_call : custom_calls_to_remove_) {
    CHECK_EQ(custom_call->operand_count(), 1);
    HloInstruction* operand = custom_call->operands()[0];
    if (custom_call->parent() !=
            custom_call->GetModule()->entry_computation() &&
        custom_call->IsCustomCall(
            host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
      // Replace custom call with a copy for dynamic-update-slice in case it
      // used parameter buffer directly because in case of aliasing with loop
      // parameters control dependencies can mess with scheduling.
      if (uses_parameter_buffer(operand)) {
        VLOG(10) << "Adding copy for custom call " << custom_call->name();
        operand =
            custom_call->parent()->AddInstruction(HloInstruction::CreateUnary(
                operand->shape(), HloOpcode::kCopy, operand));
      } else {
        VLOG(10) << "NOT Adding copy for custom call " << custom_call->name();
      }
    }
    CHECK_OK(custom_call->ReplaceAllUsesWith(operand));
    TF_RETURN_IF_ERROR(custom_call->parent()->RemoveInstruction(custom_call));
    changed = true;
  }

  return changed;
}

}  // namespace xla
