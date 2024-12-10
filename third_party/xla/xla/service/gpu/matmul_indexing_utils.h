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

#ifndef XLA_SERVICE_GPU_MATMUL_INDEXING_UTILS_H_
#define XLA_SERVICE_GPU_MATMUL_INDEXING_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Ordered non-contracting dimensions for a dot instruction operand.
absl::StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims);

// Batch dimensions of an operand of a dot instruction.
// Just an unified accessor to lhs_batch_dimensions and rhs_batch_dimensions.
const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, int operand_number);

// Index of the only contracting dimension of dot instruction operand.
absl::StatusOr<int64_t> ContractingDimensionIndex(const HloInstruction& dot,
                                                  int operand_number);

// Index of the only non-contracting dimension of dot instruction operand.
absl::StatusOr<int64_t> NonContractingDimensionIndex(const HloInstruction& dot,
                                                     int operand_number);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MATMUL_INDEXING_UTILS_H_
