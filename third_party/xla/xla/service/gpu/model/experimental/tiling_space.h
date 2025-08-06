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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_TILING_SPACE_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_TILING_SPACE_H_

#include <cstdint>
#include <deque>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla::gpu {

// TilingSpace contains information about all parallel and sequential dimensions
// and runtime variables in a fusion.
// The parallel dimensions correspond to the dimensions of the outputs of the
// fusion.
// The sequential dimensions correspond to the contraction/reduction dimensions
// of the dots/reduces in the fusion.
// The runtime variables correspond to the offsets of the dynamic slices in the
// fusion.
class TilingSpace {
 public:
  // Unique ID for the dimension or runtime variable.
  using ID = int64_t;

  enum class DimensionSemantics { kParallel, kSequential };
  struct DimensionInfo {
    // Unique ID for the dimension.
    ID id;
    // Size of the dimension.
    int64_t dimension_size;
    // Type of the dimension.
    DimensionSemantics type;
    // HLO instruction that defines the dimension.
    const HloInstruction* hlo;
    // Index into the ordered list of dimensions of the HLO instruction.
    // All dimensions in the HLO instruction are described as
    // [all parallel dims of the output, all reduction/contraction dims].
    //
    // Example:
    // [output_dims] dot(lhs, rhs, lhs_contracting_dims, rhs_contracting_dims)
    // The dimensions are ordered as [output_dims, LHS[lhs_contracting_dims]].
    int64_t dim_position;
  };

  struct RTVarInfo {
    // Unique ID for the runtime variable.
    ID id;
    // Feasible bounds of the runtime variable. The values outside of the bounds
    // will be clamped.
    Interval bounds;
    // HLO instruction that defines the runtime variable.
    const HloInstruction* hlo;
  };

  static TilingSpace Create(const HloFusionAdaptor& fusion);

  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TilingSpace& space) {
    sink.Append(space.ToString());
  }

  const DimensionInfo& GetDimensionInfo(const HloInstruction& hlo,
                                        int64_t dim_position) const;

  const RTVarInfo& GetRTVarInfo(const HloInstruction& hlo,
                                int64_t operand_id) const;

 private:
  void AppendDimension(const HloInstruction& hlo, int64_t dim_position,
                       int64_t dim_size, DimensionSemantics dim_type);
  void AppendRTVar(const HloInstruction& hlo, int64_t operand_id,
                   const HloInstruction& rt_var, int64_t upper_bound);

  void ProcessDot(const HloInstruction& hlo);
  void ProcessReduce(const HloInstruction& hlo);
  void ProcessDynamicSlice(const HloInstruction& hlo);

  // Maps from (hlo, dim_position) to the dimension info.
  absl::flat_hash_map<std::pair<const HloInstruction*, int64_t>,
                      const DimensionInfo*>
      hlo_to_dimension_;
  // The deque is used to guarantee the pointer stability.
  std::deque<DimensionInfo> dimensions_;

  // Maps from (hlo, operand_id) to the runtime variable info.
  absl::flat_hash_map<std::pair<const HloInstruction*, int64_t>,
                      const RTVarInfo*>
      hlo_to_rt_var_;
  // The deque is used to guarantee the pointer stability.
  std::deque<RTVarInfo> rt_vars_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_TILING_SPACE_H_
