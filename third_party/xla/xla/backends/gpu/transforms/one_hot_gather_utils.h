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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_ONE_HOT_GATHER_UTILS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_ONE_HOT_GATHER_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns true when `indices_type` is an integral type of at least 8 bits (so
// not `PRED` or a sub-byte type) that can represent `num_classes - 1`
// (`num_classes > 1`) without overflow when constructing the upper-bound
// comparison `indices <= num_classes - 1`.
bool IsValidOneHotIndexType(PrimitiveType indices_type, int64_t num_classes);

// Maps a batch dimension of `gather_source` (collapsed with slice size 1) to
// the corresponding dimension in `indices`.
struct BatchDimMapping {
  int64_t source_dim;
  int64_t indices_dim;
};

struct OneHotGatherSpec {
  // Integer class indices with shape [B_0, ..., B_{p-1}, M_0, ..., M_{q-1}],
  // where `num_batch_dims == p`.
  HloInstruction* indices = nullptr;
  int64_t num_batch_dims = 0;

  // Source tensor to gather from; `source_contract_dim` has size `num_classes`.
  HloInstruction* gather_source = nullptr;
  int64_t source_contract_dim = 0;

  // Batch dimensions of `gather_source` indexed via `Iota` coordinates in
  // `start_indices` and collapsed with slice size 1. Every dimension of
  // `gather_source` other than `source_contract_dim` and these batch dimensions
  // is gathered in full as an offset dimension (`N_0, ..., N_{m-1}`).
  std::vector<BatchDimMapping> source_batch_to_indices_dim;

  // Controls where `gather_source`'s offset dimensions `N_0..N_{m-1}` appear:
  // - true:  [B_0..B_{p-1}, M_0..M_{q-1}, N_0..N_{m-1}]
  // - false: [B_0..B_{p-1}, N_0..N_{m-1}, M_0..M_{q-1}]
  bool lhs_is_one_hot = true;

  PrimitiveType out_type = PRIMITIVE_TYPE_INVALID;

  // Optional scalar zero constant of `out_type` to reuse; if null, a new
  // constant 0 of `out_type` is created.
  HloInstruction* zero_scalar = nullptr;

  // If set, applies `reduce-precision` (8 exponent bits, this many mantissa
  // bits) to the gathered values before converting to `out_type`.
  std::optional<int> weights_round_mantissa_bits;
};

// Emits a 1-slice `Gather` (with no `operand_batching_dims`, so it satisfies
// `BatchedGatherScatterNormalizer`'s post-normalization invariant) guarded by
// an in-bounds `Select` (`0 <= indices <= num_classes - 1`) that masks
// out-of-bounds indices to zero.
absl::StatusOr<HloInstruction*> BuildOneHotGatherAndSelect(
    HloComputation* computation, const OneHotGatherSpec& spec,
    const OpMetadata* metadata = nullptr);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_ONE_HOT_GATHER_UTILS_H_
