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

#include "tensorflow/compiler/xla/service/llvm_ir/sort_util.h"

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

namespace {
// Adds the inner comparison loop where we compare elements pointed to by
// 'keys_index' and 'compare_keys_index'.
void EmitCompareLoop(int64 dimension_to_sort,
                     const llvm_ir::IrArray::Index& keys_index,
                     const llvm_ir::IrArray::Index& compare_keys_index,
                     const llvm_ir::IrArray& keys_array,
                     llvm::IRBuilder<>* ir_builder) {
  // TODO(b/26783907): parallelize this loop.

  // if (is_smaller_index &&
  //     compare_keys[dimension_to_sort] < dimension_to_sort_bound)
  llvm::Value* is_smaller_index = ir_builder->CreateICmpSLT(
      keys_index[dimension_to_sort], compare_keys_index[dimension_to_sort]);
  int64 dimension_to_sort_bound =
      keys_array.GetShape().dimensions(dimension_to_sort);
  auto if_data = llvm_ir::EmitIfThenElse(
      ir_builder->CreateAnd(
          is_smaller_index,
          ir_builder->CreateICmpSLT(
              compare_keys_index[dimension_to_sort],
              keys_index.GetConstantWithIndexType(dimension_to_sort_bound))),
      "smaller_comparison_index", ir_builder, /*emit_else=*/false);
  SetToFirstInsertPoint(if_data.true_block, ir_builder);
  auto key1 = keys_array.EmitReadArrayElement(keys_index, ir_builder);
  auto key2 = keys_array.EmitReadArrayElement(compare_keys_index, ir_builder);
  auto key_type = keys_array.GetShape().element_type();
  auto comparison =
      primitive_util::IsFloatingPointType(key_type)
          // TODO(b/26783907): Figure out how to handle NaNs.
          ? ir_builder->CreateFCmp(llvm::FCmpInst::FCMP_ULT, key1, key2)
          : ir_builder->CreateICmp(
                primitive_util::IsSignedIntegralType(key_type)
                    ? llvm::ICmpInst::ICMP_SLT
                    : llvm::ICmpInst::ICMP_ULT,
                key1, key2);
  auto min_key = ir_builder->CreateSelect(comparison, key1, key2);
  auto max_key = ir_builder->CreateSelect(comparison, key2, key1);
  keys_array.EmitWriteArrayElement(keys_index, min_key, ir_builder);
  keys_array.EmitWriteArrayElement(compare_keys_index, max_key, ir_builder);
}
}  // namespace

Status EmitSortInPlace(int64 dimension_to_sort, const IrArray& keys_array,
                       tensorflow::StringPiece name,
                       llvm::IRBuilder<>* ir_builder) {
  const Shape& keys_shape = keys_array.GetShape();

  // TODO(b/26783907): This case can probably be avoided with the Algebraic
  // Simplifier.
  if (ShapeUtil::IsScalar(keys_shape)) {
    return Status::OK();
  }

  // Create loop nests which loop through the operand dimensions. The sort
  // dimension is handled in three separate innermost loops which perform the
  // sorting.
  ForLoopNest loop_nest(name, ir_builder);
  IrArray::Index keys_index =
      loop_nest.EmitOperandArrayLoopNest(keys_array, dimension_to_sort, "keys");

  // 'compare_keys_index' is the index of the element that 'keys_index' should
  // be compared to.
  IrArray::Index compare_keys_index(keys_index.GetType());
  for (size_t dimension = 0; dimension < keys_index.size(); ++dimension) {
    if (dimension != dimension_to_sort) {
      compare_keys_index.push_back(keys_index[dimension]);
    } else {
      compare_keys_index.push_back(nullptr);
    }
  }

  // Create the sorting loops which do the sorting.
  int64 dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  std::unique_ptr<ForLoop> stages_loop = loop_nest.AddLoop(
      /*start_index=*/0,
      /*end_index=*/
      tensorflow::Log2Ceiling64(dimension_to_sort_bound),
      /*suffix=*/"sort_stages");
  std::unique_ptr<ForLoop> mask_loop = loop_nest.AddLoop(
      /*suffix=*/"mask",
      /*start_index=*/keys_index.GetConstantWithIndexType(0),
      /*end_index=*/stages_loop->GetIndVarValue());
  std::unique_ptr<ForLoop> compare_loop = loop_nest.AddLoop(
      /*start_index=*/0,
      /*end_index=*/dimension_to_sort_bound,
      /*suffix=*/"compare");

  // Naive C++ code for the inner loops (without parallelization):
  //
  // for (int64 stage = 0; stage < Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64 first_xor_mask = (1LL << (stage + 1)) - 1;
  //   for (int64 i = 0; i < dimension_to_sort_bound; ++i) {
  //     int64 j = i ^ first_xor_mask;
  //     if (i < j && j < dimension_to_sort_bound) {
  //       int64 min_key = std::min(keys[i], keys[j]);
  //       keys[j] = std::max(keys[i], keys[j]);
  //       keys[i] = min_key;
  //     }
  //   }
  //   for (int64 mask = 0; mask < stage; ++mask) {
  //     int64 later_xor_mask = (1LL << (stage - (mask + 1));
  //     for (int64 i = 0; i < dimension_to_sort_bound; ++i) {
  //       int64 j = i ^ later_xor_mask;
  //       if (i < j && j < dimension_to_sort_bound) {
  //         int64 min_key = std::min(keys[i], keys[j]);
  //         keys[j] = std::max(keys[i], keys[j]);
  //         keys[i] = min_key;
  //       }
  //     }
  //   }
  // }
  //
  // This follows the algorithm described on Wikipedia:
  // https://en.wikipedia.org/wiki/Bitonic_sorter

  SetToFirstInsertPoint(stages_loop->GetBodyBasicBlock(), ir_builder);
  // The first xor mask of a stage is 2^(stage + 1) - 1.
  auto first_xor_mask = ir_builder->CreateSub(
      ir_builder->CreateShl(
          keys_index.GetConstantWithIndexType(1),
          ir_builder->CreateAdd(stages_loop->GetIndVarValue(),
                                keys_index.GetConstantWithIndexType(1))),
      keys_index.GetConstantWithIndexType(1));
  std::unique_ptr<ForLoop> first_compare_loop = ForLoop::EmitForLoop(
      /*prefix=*/"first_compare",
      /*start_index=*/keys_index.GetConstantWithIndexType(0),
      /*end_index=*/
      keys_index.GetConstantWithIndexType(dimension_to_sort_bound),
      /*step=*/keys_index.GetConstantWithIndexType(1),
      /*ir_builder=*/ir_builder);

  SetToFirstInsertPoint(first_compare_loop->GetBodyBasicBlock(), ir_builder);
  // 'first_compare_loop' iterates through the 'dimension_to_sort'.
  keys_index[dimension_to_sort] = first_compare_loop->GetIndVarValue();
  compare_keys_index[dimension_to_sort] = ir_builder->CreateXor(
      first_compare_loop->GetIndVarValue(), first_xor_mask);
  EmitCompareLoop(dimension_to_sort, keys_index, compare_keys_index, keys_array,
                  ir_builder);

  SetToFirstInsertPoint(compare_loop->GetPreheaderBasicBlock(), ir_builder);
  // The later masks of a stage are 2^(stage - (mask_loop_ind_var + 1)).
  auto later_xor_mask = ir_builder->CreateShl(
      keys_index.GetConstantWithIndexType(1),
      ir_builder->CreateSub(
          stages_loop->GetIndVarValue(),
          ir_builder->CreateAdd(mask_loop->GetIndVarValue(),
                                keys_index.GetConstantWithIndexType(1))));

  SetToFirstInsertPoint(compare_loop->GetBodyBasicBlock(), ir_builder);
  // 'compare_loop' iterates through the 'dimension_to_sort'.
  keys_index[dimension_to_sort] = compare_loop->GetIndVarValue();
  compare_keys_index[dimension_to_sort] =
      ir_builder->CreateXor(compare_loop->GetIndVarValue(), later_xor_mask);
  EmitCompareLoop(dimension_to_sort, keys_index, compare_keys_index, keys_array,
                  ir_builder);

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop. This ensures later instructions are inserted after this loop nest.
  ir_builder->SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());

  return Status::OK();
}

}  // namespace llvm_ir
}  // namespace xla
