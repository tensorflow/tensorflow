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

#ifndef XLA_SERVICE_GPU_REDUCTION_UTILS_H_
#define XLA_SERVICE_GPU_REDUCTION_UTILS_H_

#include <cstdint>
#include <ostream>

#include "absl/container/inlined_vector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Need at least 1024 threads/block for reasonable tree reduction
// performance (assuming all data fits).
inline constexpr int64_t MinThreadsXRowReduction() { return 1024; }

// When doing batched row reduction, how big the batch dimension could be.
inline constexpr int64_t BatchedReductionRaceFreeBound() { return 8; }

struct ReductionDimensions {
  // The reduction dimension indices used below.
  constexpr static int kRowMajorReducedDimension = 0;
  constexpr static int kRowKeptDimension = 1;
  constexpr static int kRowMinorReducedDimension = 2;

  constexpr static int kColMajorKeptDimension = 0;
  constexpr static int kColReducedDimension = 1;
  constexpr static int kColMinorKeptDimension = 2;

  // Indicates whether the reduction is a row reduction or a column reduction.
  bool is_row_reduction;

  // We collapse contiguous reduced or kept dimensions into a single dimension
  // for the reduction. However, for historical reasons, this is not done at the
  // HLO level. We only support reductions where either all the reduced or all
  // the kept dimensions are contiguous, so we end up with two types:
  //
  //   row reductions:    [a, b, c] -> [b]    (a and c are reduced).
  //   column reductions: [a, b, c] -> [a, c] (b is reduced).
  //
  // If the input has less than three dimensions, a (and b if it's a 1d
  // reduction) are set to 1.
  Vector3 dimensions;

  absl::InlinedVector<int64_t, 2> GetOutputShape() const {
    if (is_row_reduction) {
      return {dimensions[kRowKeptDimension]};
    }
    return {dimensions[kColMajorKeptDimension],
            dimensions[kColMinorKeptDimension]};
  }

  bool operator==(const ReductionDimensions& other) const {
    return is_row_reduction == other.is_row_reduction &&
           dimensions == other.dimensions;
  }
};

std::ostream& operator<<(std::ostream& os,
                         const ReductionDimensions& reduction_dimensions);

// Returns true if using the reduction emitter is estimated to be faster than
// using the elemental emitter.
bool IsUnnestedReductionFasterThanElemental(
    const ReductionDimensions& reduction_dimensions,
    const se::DeviceDescription& device_description);

// Returns true if either the dimensions being reduced or the dimensions being
// kept are contiguous in the input of the reduce instruction.
bool IsReductionFromOrToContiguousDimensions(
    const HloInstruction& reduce,
    const se::DeviceDescription& device_description);

// Given the input shape and dimensions to reduce for a reduction, returns
// ReductionDimensions.
//
// Prerequisite: the reduction instruction passes the check
// IsReductionFromOrToContiguousDimensions, which guarantees either the
// dimensions to reduce or the dimensions to keep are consecutive.
ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce);

// Get tiling per thread for the given reduction in dimensions [D, H, W].
Vector3 GetReductionTiling(const ReductionDimensions& reduction_dimensions);

// How big the reduction dimension can be to be race free.
int64_t ReductionDimensionRaceFreeBound(
    const ReductionDimensions& reduction_dimensions,
    const se::DeviceDescription& device_description);

// Returns whether the given reduction can be safely generated without atomics :
// that is, at most one block will write to every output element.
bool ReductionIsRaceFree(const ReductionDimensions& reduction_dimensions,
                         const se::DeviceDescription& device_description);

// Whether the instruction is a reduction hero for the given root.
bool IsRealReductionHero(const HloInstruction& root, const HloInstruction& hero,
                         const se::DeviceDescription& device_description);

// Whether `reduction_hero` is compatible with `first_reduce`.
bool AreReductionsMultiOutputFusionCompatible(
    const HloInstruction* reduce_hero, const HloInstruction* first_reduce);

}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_REDUCTION_UTILS_H_
