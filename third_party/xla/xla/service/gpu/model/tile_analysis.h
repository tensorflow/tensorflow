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

#ifndef XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// Contains an affine map with N dimension expressions and M symbols:
//   (d0, ..., d_{N - 1})[s_0, ..., s_{M - 1}] -> f(d_i, s_j)
// Dimensions d_i correspond to the iteration space of the output tensor. Some
// or all of the dimensions of the input operands can be expressed as a function
// of dimensions of output. For example, for broadcasts and cwise ops all
// dimensions of the inputs are covered by the output dimensions.
// Symbols s_j correspond to the dimensions that are present ONLY in inputs.
// `sizes` is an array that holds the upper bounds for the iteration sizes for
//  every output/input dimension
// i.e. elements size_0, ..., size_{N - 1} correspond to the output dimensions
//  d0, ..., d_{N - 1} and elements size_N, ..., size_{N + M - 1} correspond
// to the input-only dimensions s_0, ..., s_{M - 1}.
// Note, that the sizes have upper bounds only and the lower bounds are always
// 0, since we can encode the offsets in the affine map.
//
// Example:
//
// 1. Indexing map for the input of the following reduction
// ```
//   p0 = f32[150, 20, 10, 50] parameter(0)
//   reduce = f32[150, 10] reduce(p0, p0_init), dimensions={3, 1}
// ```
// can be written as `(d0, d1)[s0, s1] -> (d0, s0, d1, s1)`  with the sizes
// `[/*d0 size=*/150, /*d1 size=*/10, /*s0 size=*/20, /*s1 size=*/50]`.
//
// 2. Indexing map for the input of the reverse op
// ```
//  %p0 = f32[1, 17, 9, 9] parameter(0)
//  reverse = f32[1, 17, 9, 9] reverse(%p0), dimensions={1, 2}
// ```
// can be written as `(d0, d1, d2, d3) -> (d0, -d1 + 17, -d2 + 9, d3)` with the
// sizes `[/*d0 size=*/1, /*d1 size=*/17, /*d2 size=*/9, /*d3 size=*/9]`.
struct IndexingMap {
  mlir::AffineMap affine_map;
  std::vector<int64_t> sizes;

  std::string ToString() const;
};
std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map);

// Contains 1 or more indexing maps for the `operand_id`. There are cases, when
// the same input operand is read multiple times in various ways. Especially, it
// happens a lot in fusion ops.
struct HloOperandIndexing {
  std::vector<IndexingMap> indexing_maps;
  int64_t operand_id;

  std::string ToString() const;
};
std::ostream& operator<<(std::ostream& out,
                         const HloOperandIndexing& operand_indexing);

// Contains indexing maps for all N-dimensional tensor input operands that
// correspond to a particular output.
struct HloInstructionIndexing {
  std::vector<HloOperandIndexing> operand_indexing_maps;

  std::string ToString() const;
};
std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing);

std::string ToString(const mlir::AffineMap& affine_map);

// Computes indexing maps for all input operands necessary to compute an element
// of the `output_id` instruction output.
StatusOr<HloInstructionIndexing> ComputeInstructionIndexing(
    const HloInstruction* instr, int output_id,
    mlir::MLIRContext* mlir_context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_
