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
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

namespace {
// Adds the inner comparison loop where we compare elements pointed to by
// 'keys_index' and 'compare_keys_index'.
void EmitCompareLoop(int64 dimension_to_sort, const IrArray::Index& keys_index,
                     const IrArray::Index& compare_keys_index,
                     const IrArray& keys_array,
                     const tensorflow::gtl::optional<IrArray>& values_array,
                     llvm::IRBuilder<>* b) {
  // if (is_smaller_index &&
  //     compare_keys[dimension_to_sort] < dimension_to_sort_bound)
  llvm::Value* is_smaller_index = b->CreateICmpSLT(
      keys_index[dimension_to_sort], compare_keys_index[dimension_to_sort]);
  int64 dimension_to_sort_bound =
      keys_array.GetShape().dimensions(dimension_to_sort);
  auto if_data = EmitIfThenElse(
      b->CreateAnd(is_smaller_index,
                   b->CreateICmpSLT(compare_keys_index[dimension_to_sort],
                                    keys_index.GetConstantWithIndexType(
                                        dimension_to_sort_bound))),
      "smaller_comparison_index", b, /*emit_else=*/false);
  SetToFirstInsertPoint(if_data.true_block, b);
  auto key1 = keys_array.EmitReadArrayElement(keys_index, b);
  auto key2 = keys_array.EmitReadArrayElement(compare_keys_index, b);
  auto key_type = keys_array.GetShape().element_type();
  auto comparison =
      primitive_util::IsFloatingPointType(key_type)
          // TODO(b/26783907): Figure out how to handle NaNs.
          ? b->CreateFCmp(llvm::FCmpInst::FCMP_ULT, key2, key1)
          : b->CreateICmp(primitive_util::IsSignedIntegralType(key_type)
                              ? llvm::ICmpInst::ICMP_SLT
                              : llvm::ICmpInst::ICMP_ULT,
                          key2, key1);
  // If key2 < key1
  auto if_smaller_data =
      EmitIfThenElse(comparison, "is_smaller_than", b, /*emit_else=*/false);
  SetToFirstInsertPoint(if_smaller_data.true_block, b);
  // Swap key1 with key2.
  keys_array.EmitWriteArrayElement(keys_index, key2, b);
  keys_array.EmitWriteArrayElement(compare_keys_index, key1, b);
  if (values_array.has_value()) {
    // Also swap the values.
    auto value1 = values_array.value().EmitReadArrayElement(keys_index, b);
    auto value2 =
        values_array.value().EmitReadArrayElement(compare_keys_index, b);
    values_array.value().EmitWriteArrayElement(keys_index, value2, b);
    values_array.value().EmitWriteArrayElement(compare_keys_index, value1, b);
  }
}
}  // namespace

Status EmitSortInPlace(int64 dimension_to_sort, const IrArray& keys_array,
                       const tensorflow::gtl::optional<IrArray>& values_array,
                       tensorflow::StringPiece name, llvm::Value* xor_mask,
                       llvm::IRBuilder<>* b,
                       const gpu::LaunchDimensions* launch_dimensions) {
  const Shape& keys_shape = keys_array.GetShape();

  // Create loop nests which loop through the operand dimensions. The sort
  // dimension is handled in the innermost loop which performs the sorting.
  ForLoopNest loop_nest(name, b);
  IrArray::Index keys_index =
      loop_nest.EmitOperandArrayLoopNest(keys_array, dimension_to_sort, "keys");
  if (loop_nest.GetInnerLoopBodyBasicBlock() != nullptr) {
    SetToFirstInsertPoint(loop_nest.GetInnerLoopBodyBasicBlock(), b);
  }

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

  // Naive C++ code for the inner compare loop:
  //
  // for (int64 i = 0; i < dimension_to_sort_bound; ++i) {
  //   int64 j = i ^ xor_mask;
  //   if (i < j && j < dimension_to_sort_bound) {
  //     int64 min_key = std::min(keys[i], keys[j]);
  //     keys[j] = std::max(keys[i], keys[j]);
  //     keys[i] = min_key;
  //   }
  // }
  //
  // This follows the algorithm described on Wikipedia:
  // https://en.wikipedia.org/wiki/Bitonic_sorter

  int64 dimension_to_sort_bound =
      keys_array.GetShape().dimensions(dimension_to_sort);
  Shape compare_shape = ShapeUtil::MakeShape(keys_shape.element_type(),
                                             {dimension_to_sort_bound});
  auto compare_loop_body_emitter =
      [&](const IrArray::Index& compare_index) -> Status {
    keys_index[dimension_to_sort] = compare_index[0];
    compare_keys_index[dimension_to_sort] =
        b->CreateXor(compare_index[0], xor_mask);
    EmitCompareLoop(dimension_to_sort, keys_index, compare_keys_index,
                    keys_array, values_array, b);
    return Status::OK();
  };
  if (launch_dimensions != nullptr) {
    TF_RETURN_IF_ERROR(gpu::ParallelLoopEmitter(compare_loop_body_emitter,
                                                compare_shape,
                                                *launch_dimensions, b)
                           .EmitLoop(name));
  } else {
    TF_RETURN_IF_ERROR(LoopEmitter(compare_loop_body_emitter, compare_shape, b)
                           .EmitLoop(name));
  }

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop. This ensures later instructions are inserted after this loop nest.
  b->SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());

  return Status::OK();
}

}  // namespace llvm_ir
}  // namespace xla
