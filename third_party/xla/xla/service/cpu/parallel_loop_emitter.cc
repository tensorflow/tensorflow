/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/parallel_loop_emitter.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    const llvm_ir::IrArray& target_array,
    const DynamicLoopBounds* dynamic_loop_bounds, llvm::IRBuilderBase* b)
    : LoopEmitter(target_element_generator, target_array, b),
      dynamic_loop_bounds_(dynamic_loop_bounds) {}

std::vector<llvm_ir::IrArray::Index>
ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock(absl::string_view loop_name,
                                                   llvm::Type* index_type,
                                                   llvm::Value* base_index) {
  CHECK_NE(index_type, nullptr);

  CHECK_EQ(base_index, nullptr)
      << "XLA CPU implementation of"
      << " ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock doesn't support"
      << " base_index, but it was requested.";

  CHECK(!shape_.IsTuple());
  CHECK(!ShapeUtil::IsScalar(shape_));

  llvm_ir::ForLoopNest loop_nest(loop_name, b_);
  const int64_t num_dims = shape_.rank();
  std::vector<llvm::Value*> array_multi_index(num_dims);

  // Add loops from outer-most to inner-most dimensions.
  for (int i = LayoutUtil::MinorToMajor(shape_).size() - 1; i >= 0; --i) {
    const int64_t dimension = LayoutUtil::Minor(shape_.layout(), i);
    const int bounds_index = num_dims - 1 - i;
    // Only unroll the most minor dimension, this seems to give us good runtime
    // performance with a large improvement in compile time.
    auto unroll_mode = (i == 0) ? llvm_ir::UnrollMode::kDefaultUnroll
                                : llvm_ir::UnrollMode::kNoUnroll;
    if (bounds_index < dynamic_loop_bounds_->size()) {
      // Emit dynamic loop bounds for this dimension. Dynamic loop bounds
      // are read from ir function dynamic loop bounds argument.
      llvm::Value* start_index = (*dynamic_loop_bounds_)[bounds_index].first;
      llvm::Value* end_index = (*dynamic_loop_bounds_)[bounds_index].second;

      std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
          /*suffix=*/absl::StrFormat("dim.%d", dimension), start_index,
          end_index, unroll_mode);
      array_multi_index[dimension] = loop->GetIndVarValue();
    } else {
      // Emit static loop bounds for this dimension.
      std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
          /*start_index=*/0,
          /*end_index=*/shape_.dimensions(dimension),
          /*suffix=*/absl::StrFormat("dim.%d", dimension), unroll_mode);
      array_multi_index[dimension] = loop->GetIndVarValue();
    }
  }
  // Point IR builder at inner loop BB.
  llvm_ir::SetToFirstInsertPoint(loop_nest.GetInnerLoopBodyBasicBlock(), b_);

  // Set exit_bb_ to the exit block of the loop nest.
  exit_bb_ = loop_nest.GetOuterLoopExitBasicBlock();
  CHECK(exit_bb_ != nullptr);

  llvm_ir::IrArray::Index array_index(array_multi_index, shape_, index_type);
  return {array_index};
}

}  // namespace cpu
}  // namespace xla
