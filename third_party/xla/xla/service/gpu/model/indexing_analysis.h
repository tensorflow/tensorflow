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

#ifndef XLA_SERVICE_GPU_MODEL_INDEXING_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_INDEXING_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// Range represents a semi-closed interval [lower_bound, upper_bound).
struct Range {
  std::string ToString() const;

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
};
std::ostream& operator<<(std::ostream& out, const Range& range);

template <typename H>
H AbslHashValue(H h, const Range& range) {
  return H::combine(std::move(h), range.lower_bound, range.upper_bound);
}

// Domain contains ranges for symbols and dimensions of an affine map.
struct Domain {
  std::string ToString() const;

  static Domain FromUpperBounds(
      absl::Span<const int64_t> dimension_upper_bounds,
      absl::Span<const int64_t> symbol_upper_bounds);

  std::vector<Range> dimension_ranges;
  std::vector<Range> symbol_ranges;
};
std::ostream& operator<<(std::ostream& out, const Domain& domain);

template <typename H>
H AbslHashValue(H h, const Domain& domain) {
  return H::combine(std::move(h), domain.dimension_ranges,
                    domain.symbol_ranges);
}

// Contains an affine map with N dimension expressions and M symbols:
//   (d0, ..., d_{N - 1})[s_0, ..., s_{M - 1}] -> f(d_i, s_j)
// Dimensions d_i correspond to the iteration space of the output tensor. Some
// or all of the dimensions of the input operands can be expressed as a function
// of dimensions of output. For example, for broadcasts and cwise ops all
// dimensions of the inputs are covered by the output dimensions.
// Domain specifies for what ranges of values the indexing map is specified.
//
// Example:
//
// 1. Indexing map for the input of the following reduction
// ```
//   p0 = f32[150, 20, 10, 50] parameter(0)
//   reduce = f32[150, 10] reduce(p0, p0_init), dimensions={3, 1}
// ```
// can be written as `(d0, d1)[s0, s1] -> (d0, s0, d1, s1)`  with
// d0 in [0, 150), d1 in [0, 10), s0 in [0, 20) and s1 in [0, 50).
//
// 2. Indexing map for the input of the reverse op
// ```
//  %p0 = f32[1, 17, 9, 9] parameter(0)
//  reverse = f32[1, 17, 9, 9] reverse(%p0), dimensions={1, 2}
// ```
// can be written as `(d0, d1, d2, d3) -> (d0, -d1 + 16, -d2 + 8, d3)` with
// d0 in [0, 1), d1 in [0, 17), d2 in [0, 9) and d3 in [0, 9).
struct IndexingMap {
  std::string ToString() const;

  // Returns true if the map was simplified.
  bool Simplify();

  mlir::AffineMap affine_map;
  Domain domain;
};
std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map);
bool operator==(const IndexingMap& lhs, const IndexingMap& rhs);

template <typename H>
H AbslHashValue(H h, const IndexingMap& indexing_map) {
  llvm::hash_code affine_map_hash = llvm::hash_combine(indexing_map.affine_map);
  return H::combine(std::move(h), static_cast<size_t>(affine_map_hash),
                    indexing_map.domain);
}

// Contains indexing maps for all N-dimensional tensor input operands that
// correspond to a particular output.
struct HloInstructionIndexing {
  std::string ToString() const;

  // Returns true if the indexing was simplified.
  bool Simplify();

  // Creates a HloInstructionIndexing from a list of indexing maps for all
  // operands and sorted w.r.t. operand index, i.e. indexing_maps[i] corresponds
  // to operand[i] of the instruction.
  static HloInstructionIndexing FromIndexingMaps(
      absl::Span<const IndexingMap> indexing_maps);

  // Maps input operand index to the indexing map for one particular output.
  absl::flat_hash_map<int64_t, absl::flat_hash_set<IndexingMap>> indexing_maps;
};
std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing);

std::string ToString(const mlir::AffineMap& affine_map);

// Computes indexing maps for all input operands necessary to compute an element
// of the `output_id` instruction output.
StatusOr<HloInstructionIndexing> ComputeOutputToInputIndexing(
    const HloInstruction* instr, int output_id, mlir::MLIRContext* ctx);

// Computes indexing maps for all output operands that the element of the
// `input_id` instruction input will participate in.
StatusOr<HloInstructionIndexing> ComputeInputToOutputIndexing(
    const HloInstruction* instr, int input_id, mlir::MLIRContext* ctx);

// Groups indexing maps by instructions.
using IndexingMapSet = absl::flat_hash_set<IndexingMap>;
using GroupedByOpIndexingMap =
    absl::flat_hash_map<const HloInstruction*, IndexingMapSet>;
StatusOr<GroupedByOpIndexingMap> ComputeGroupedOutputToInputIndexing(
    const HloFusionAdaptor& fusion_adaptor, int output_id,
    mlir::MLIRContext* ctx);

// Computes a transpose indexing map.
mlir::AffineMap ComputeTransposeIndexingMap(
    absl::Span<const int64_t> permutation, mlir::MLIRContext* mlir_context);

template <typename T>
std::string ToStringImpl(const T& value) {
  std::string s;
  std::stringstream ss(s);
  ss << value;
  return ss.str();
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_ANALYSIS_H_
