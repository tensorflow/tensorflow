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
#include "xla/service/gpu/fusions/thunk_util.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/runtime/memset_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::optional<std::unique_ptr<Thunk>>>
BuildConstantInitializerThunk(IrEmitterContext& ir_emitter_context,
                              const HloInstruction* instr,
                              const HloInstruction* init_value,
                              BufferAllocation::Slice dest_slice) {
  if (const HloConstantInstruction* constant =
          DynCast<HloConstantInstruction>(init_value)) {
    const Literal& literal = constant->literal();
    absl::Span<const uint8_t> literal_bytes(
        static_cast<const uint8_t*>(literal.untyped_data()),
        literal.size_bytes());
    int64_t num_bytes = literal_bytes.size();

    const Shape dest_shape = instr->shape();

    Thunk::ThunkInfo thunk_info =
        Thunk::ThunkInfo::WithProfileAnnotation(instr);
    if (absl::c_all_of(literal_bytes, [](uint8_t byte) { return byte == 0; })) {
      return {{std::make_unique<MemzeroThunk>(thunk_info, dest_slice)}};
    }

    // If the literal is 8 or 16 bits wide, we can emit a 32-bit memset by
    // repeating the literal 4 or 2 times, so long as the destination buffer is
    // an even multiple of 32 bits long.
    if ((num_bytes == 1 || num_bytes == 2) &&
        ShapeUtil::ByteSizeOf(dest_shape) % 4 == 0) {
      uint16_t pattern16;
      if (num_bytes == 1) {
        uint8_t b = literal_bytes.front();
        pattern16 = uint16_t{b} | (uint16_t{b} << 8);
      } else {
        memcpy(&pattern16, literal_bytes.data(), sizeof(pattern16));
      }
      uint32_t pattern32 = uint32_t{pattern16} | (uint32_t{pattern16} << 16);
      return {{std::make_unique<Memset32BitValueThunk>(thunk_info, pattern32,
                                                       dest_slice)}};
    }

    // If the literal is an even multiple of 32 bits wide, we can emit a 32-bit
    // memset so long as all 32-bit words of the scalar are equal to each other.
    if (num_bytes >= 4 && num_bytes % 4 == 0 &&
        memcmp(literal_bytes.data(), literal_bytes.data() + 4,
               literal_bytes.size() - 4) == 0) {
      uint32_t word;
      memcpy(&word, literal_bytes.data(), sizeof(word));
      return {{std::make_unique<Memset32BitValueThunk>(thunk_info, word,
                                                       dest_slice)}};
    }
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
