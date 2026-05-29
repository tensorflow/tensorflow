/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_ANALYSIS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// DynamicSliceDescriptor
//===-----------------------------------------------------------------------===/

// Fully resolved description of a dynamic-slice or dynamic-update-slice whose
// offset is either a linear function of a parent while loop's induction
// variable, or fully static (all-constant offsets).
//
// Loop-dependent case (byte_stride != 0):
//   buffer address at iteration i = base + byte_offset + byte_stride * i
//   while_loop and loop_index are set.
//
// Fully static case (byte_stride == 0):
//   buffer address = base + byte_offset
//   while_loop and loop_index are nullopt.
struct DynamicSliceDescriptor {
  // The while loop whose induction variable drives the slice offset.
  // Nullopt when the offset is fully static (all-constant offsets).
  std::optional<const HloInstruction*> while_loop;

  // Index into the while loop nest (0 = innermost, 1 = one level up, etc.).
  // Nullopt when the offset is fully static.
  //
  // Currently always 0 (loop analysis resolves offsets only from the
  // immediately enclosing while loop), but the runtime supports arbitrary loop
  // nesting depths.
  std::optional<int64_t> loop_index;

  // Byte offset into the buffer at iteration 0 (or the static byte offset).
  int64_t byte_offset;

  // Byte stride per loop iteration. Zero when the offset is fully static.
  int64_t byte_stride;
};

// Analyzes a dynamic-slice or dynamic-update-slice instruction and resolves its
// runtime offset into a DynamicSliceDescriptor. Handles two cases:
//
//  1. Loop-dependent offsets: at least one offset operand depends on a parent
//     while loop's induction variable. Evaluates the offset expression for all
//     loop iterations using HloEvaluator, verifies linearity (constant stride),
//     and returns byte_offset, byte_stride, while_loop, and loop_index.
//
//  2. Fully static offsets: all offset operands are compile-time constants.
//     Computes the byte offset directly and returns byte_stride=0 with
//     while_loop and loop_index unset.
//
// Returns nullopt when the instruction is not a DS/DUS, the slice is not
// contiguous, an offset depends on runtime data that is not an induction
// variable, or the offset pattern is not linear.
absl::StatusOr<std::optional<DynamicSliceDescriptor>> AnalyzeDynamicSlice(
    const HloInstruction* instr);

// Returns the index of the first offset operand for a DS or DUS instruction.
int32_t GetFirstOffsetOperandIndex(const HloInstruction* slice);

// Returns the slice size in the given dimension for a DS or DUS instruction.
int64_t GetSliceSize(const HloInstruction* slice, int32_t dim);

//===-----------------------------------------------------------------------===/
// DynamicSliceChain
//===-----------------------------------------------------------------------===/

// All dynamic-slice (DS) and dynamic-update-slice (DUS) operations that access
// the same underlying buffer in a while loop body. The buffer enters as a
// parameter tuple element and exits through the ROOT tuple. DS reads from the
// buffer and DUS writes to it (aliasing parameter with result).
//
// DUS operations can form chains: each DUS writes to the result of the
// previous, building up the final buffer value:
//   buf = GTE(param, 1)
//   updated1 = DUS(buf, v1, off1)       <-- first write
//   updated2 = DUS(updated1, v2, off2)  <-- second write, chains from first
//   ROOT tuple(..., updated2, ...)
//
// DS operations read from the input value (before any DUS):
//   slice = DS(buf, ivar, 0)            <-- reads from original buffer
//
// Here {slice} and {updated1, updated2} form a DynamicSliceChain.
struct DynamicSliceChain {
  // Buffer source: GTE(parameter) for tuple parameters, or parameter directly.
  const HloInstruction* buffer = nullptr;

  // The last DUS in the chain whose result feeds into the ROOT tuple, or
  // nullptr if there are no DUS updates. For the example above, this is
  // updated2.
  const HloInstruction* result = nullptr;

  std::vector<const HloDynamicSliceInstruction*> slices;
  std::vector<const HloDynamicUpdateSliceInstruction*> updates;
};

// Finds the set of all DS/DUS operations in a while loop body that access the
// same underlying buffer as `instr`. The instruction must be a dynamic-slice or
// dynamic-update-slice. Traces the buffer operand back through bitcasts and DUS
// chains to find the source parameter element, then collects all DS/DUS in the
// computation that share the same source. Also identifies the last DUS in the
// chain (the one whose result feeds into the ROOT tuple).
absl::StatusOr<DynamicSliceChain> FindDynamicSliceChain(
    const HloInstruction* instr);

// Returns true if all DUS operations in the chain write to non-overlapping byte
// ranges at every loop iteration. DS reads are not checked — it is valid for a
// DS and DUS to access the same slice (read before write within an iteration).
// Returns nullopt if any DUS cannot be analyzed (e.g. non-linear offsets or
// missing loop metadata).
std::optional<bool> IsNonOverlapping(const DynamicSliceChain& chain);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_ANALYSIS_H_
