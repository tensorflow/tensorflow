/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/runtime3/memset_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {
namespace {

// TODO(b/291536641): Clean this up. What's the difference between this and the
// caller?
std::optional<std::unique_ptr<Thunk>> BuildConstantInitializerThunk(
    mlir::Operation* op, absl::Span<const uint8_t> init_value, mlir::Value dest,
    const BufferAllocation::Slice& dest_slice, const Shape& output_shape) {
  int64_t num_bytes = init_value.size();
  if (absl::c_all_of(init_value, [](uint8_t byte) { return byte == 0; })) {
    return {{std::make_unique<MemzeroThunk>(Thunk::ThunkInfo(op), dest_slice,
                                            dest)}};
  }

  // If the literal is 8 or 16 bits wide, we can emit a 32-bit memset by
  // repeating the literal 4 or 2 times, so long as the destination buffer is
  // an even multiple of 32 bits long.
  if ((num_bytes == 1 || num_bytes == 2) &&
      ShapeUtil::ByteSizeOf(output_shape) % 4 == 0) {
    uint16_t pattern16;
    if (num_bytes == 1) {
      uint8_t b = init_value.front();
      pattern16 = uint16_t{b} | (uint16_t{b} << 8);
    } else {
      memcpy(&pattern16, init_value.data(), sizeof(pattern16));
    }
    uint32_t pattern32 = uint32_t{pattern16} | (uint32_t{pattern16} << 16);
    return {{std::make_unique<Memset32BitValueThunk>(
        Thunk::ThunkInfo(op), pattern32, dest_slice, dest)}};
  }

  // If the literal is an even multiple of 32 bits wide, we can emit a 32-bit
  // memset so long as all 32-bit words of the scalar are equal to each other.
  if (num_bytes >= 4 && num_bytes % 4 == 0 &&
      memcmp(init_value.data(), init_value.data() + 4, init_value.size() - 4) ==
          0) {
    uint32_t word;
    memcpy(&word, init_value.data(), sizeof(word));
    return {{std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(op), word,
                                                     dest_slice, dest)}};
  }

  return std::nullopt;
}

}  // namespace

StatusOr<std::optional<std::unique_ptr<Thunk>>> BuildConstantInitializerThunk(
    IrEmitterContext& ir_emitter_context, mlir::Operation* op,
    const HloInstruction* instr, const HloInstruction* init_value,
    mlir::Value dest, BufferAllocation::Slice dest_slice) {
  if (const HloConstantInstruction* constant =
          DynCast<HloConstantInstruction>(init_value)) {
    const Literal& literal = constant->literal();
    absl::Span<const uint8_t> literal_bytes(
        static_cast<const uint8_t*>(literal.untyped_data()),
        literal.size_bytes());

    const Shape dest_shape = instr->shape();
    return BuildConstantInitializerThunk(op, literal_bytes, dest, dest_slice,
                                         dest_shape);
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
