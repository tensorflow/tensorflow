/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/constant_allocation.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

se::DeviceMemoryBase ConstantAllocation::AsDeviceMemoryBase() const {
  if (auto* _ = std::get_if<std::monostate>(&data)) {
    return se::DeviceMemoryBase();
  }

  if (auto* owned = std::get_if<std::unique_ptr<Literal>>(&data)) {
    return se::DeviceMemoryBase((*owned)->untyped_data(),
                                (*owned)->size_bytes());
  }

  auto* view = std::get_if<absl::Span<const uint8_t>>(&data);
  return se::DeviceMemoryBase(
      const_cast<void*>(reinterpret_cast<const void*>(view->data())),
      view->size());
}

static absl::StatusOr<ConstantAllocation> LiteralToConstantAllocation(
    BufferAllocation::Index index, const Literal& literal) {
  // TODO(ezhulenev): This code is almost identical to code in XLA:GPU, we
  // should standardize it. See `xla/service/gpu/ir_emission_utils.cc`.
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError(
        "Only array literals can be converted to constant allocations");
  }

  int64_t size_bytes = literal.size_bytes();
  const void* untyped_data = literal.untyped_data();

  // Pack sub-byte types into an XLA storage format.
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    int bit_width = primitive_util::BitWidth(element_type);
    int packed_size_bytes = CeilOfRatio<int64_t>(size_bytes, 8 / bit_width);

    // Use Literal as a storage for packed data as it allocates underlying
    // buffer with correct alignment. Keep it allocated on heap to avoid
    // capturing stack address that will be invalidated by a move below.
    auto packed = std::make_unique<Literal>(
        ShapeUtil::MakeShape(U8, {packed_size_bytes}));

    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(untyped_data), size_bytes),
        absl::MakeSpan(reinterpret_cast<char*>(packed->untyped_data()),
                       packed->size_bytes()));

    return ConstantAllocation{index, std::move(packed)};
  }

  // Create a constant allocation from the literal's untyped data.
  return ConstantAllocation{
      index, absl::Span<const uint8_t>(
                 reinterpret_cast<const uint8_t*>(untyped_data), size_bytes)};
}

// Creates a vector of constant allocations from the given buffer assignment.
absl::StatusOr<std::vector<ConstantAllocation>> CreateConstantAllocations(
    const BufferAssignment& assignment) {
  std::vector<ConstantAllocation> constants;

  for (const BufferAllocation& allocation : assignment.Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    // Find the constant instruction defining the value for allocation.
    HloInstruction* const_instr = nullptr;
    for (const auto& [value, _] : allocation.assigned_buffers()) {
      // Multiple aliasing instructions can share the allocation, we need to
      // find the original constant instruction that defines the value.
      if (value->instruction()->opcode() == HloOpcode::kConstant) {
        if (const_instr != nullptr) {
          return absl::InternalError(
              absl::StrCat("Multiple constant instructions define buffer ",
                           allocation.ToString()));
        }
        const_instr = value->instruction();
      }
    }
    if (const_instr == nullptr) {
      return absl::InternalError(
          absl::StrCat("Could not find constant instruction defining buffer ",
                       allocation.ToString()));
    }

    VLOG(3) << "Create constant allocation for index " << allocation.index()
            << " from constant literal " << const_instr->name()
            << "; shape=" << const_instr->literal().shape();
    TF_ASSIGN_OR_RETURN(constants.emplace_back(),
                        LiteralToConstantAllocation(allocation.index(),
                                                    const_instr->literal()));
  }

  return constants;
}

}  // namespace xla::cpu
