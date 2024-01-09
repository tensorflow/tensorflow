/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_LLVM_IR_SORT_UTIL_H_
#define XLA_SERVICE_LLVM_IR_SORT_UTIL_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Value.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "tsl/platform/status.h"

namespace xla {
namespace llvm_ir {
using EmitCallToNestedComputationCallback =
    std::function<Status(absl::Span<llvm::Value* const>, llvm::Value*)>;
// Emits llvm IR to do pairwise comparisons/swaps in the 'dimension_to_sort'
// dimension of each array in 'values_arrays'. All other dimensions are kept
// as-is. This implements the inner loop of BitonicSort. It is assumed that
// 'xor_masks' contains only powers of 2, or values 2^k - 1 (k > 0).
Status EmitSortInPlace(
    int64_t dimension_to_sort, const std::vector<IrArray>& values_arrays,
    absl::string_view name, absl::Span<const int64_t> xor_masks,
    llvm::IRBuilder<>* b, const gpu::LaunchDimensions& launch_dimensions,
    int64_t num_iterations_in_sort_dim, int64_t tile_size,
    const EmitCallToNestedComputationCallback& emit_compare_callback);
}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_SORT_UTIL_H_
