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

#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
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
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

namespace {

// Adds the inner comparison loop where we compare elements pointed to by
// 'keys_index' and 'compare_keys_index'.
void EmitCompareLoopBody(
    llvm::Value* dimension_to_sort_bound, PrimitiveType key_type,
    int64 num_values, llvm::Value* keys_index, llvm::Value* compare_keys_index,
    std::function<llvm::Value*(int64 operand, llvm::Value* index)> read_element,
    std::function<void(int64 operand, llvm::Value* index, llvm::Value* value)>
        write_element,
    llvm::IRBuilder<>* b) {
  // if (is_smaller_index && compare_keys < dimension_to_sort_bound)
  llvm::Value* is_smaller_index =
      b->CreateICmpSLT(keys_index, compare_keys_index);
  auto if_data = EmitIfThenElse(
      b->CreateAnd(is_smaller_index, b->CreateICmpSLT(compare_keys_index,
                                                      dimension_to_sort_bound)),
      "smaller_comparison_index", b, /*emit_else=*/false);
  SetToFirstInsertPoint(if_data.true_block, b);
  auto key1 = read_element(0, keys_index);
  auto key2 = read_element(0, compare_keys_index);
  auto compare_key1 = key1;
  auto compare_key2 = key2;
  bool is_signed_comparison = true;
  if (primitive_util::IsFloatingPointType(key_type)) {
    // We would like a total order of floating point numbers so that the sort
    // has a predictable behavior in the presence of NaNs. Rather than using
    // floating point comparison, we use the following trick:
    // If f is a float, and
    // x = bit_cast<int32>(f);
    // y = x < 0 ? 0x7FFFFFFF - x : x;
    // then y is ordered as an int32 such that finite values have the obvious
    // order, -0 is ordered before 0, and -NaN and NaN appear at the beginning
    // and end of the ordering.
    auto k = b->getInt(llvm::APInt::getSignedMaxValue(
        key1->getType()->getPrimitiveSizeInBits()));
    auto comparison_type = k->getType();
    auto zero = llvm::ConstantInt::get(comparison_type, 0);
    auto maybe_flip = [&](llvm::Value* v) {
      return b->CreateSelect(b->CreateICmp(llvm::ICmpInst::ICMP_SLT, v, zero),
                             b->CreateSub(k, v), v);
    };
    compare_key1 = b->CreateBitCast(key1, comparison_type);
    compare_key2 = b->CreateBitCast(key2, comparison_type);
    compare_key1 = maybe_flip(compare_key1);
    compare_key2 = maybe_flip(compare_key2);
  } else if (!primitive_util::IsSignedIntegralType(key_type)) {
    is_signed_comparison = false;
  }
  auto comparison =
      b->CreateICmp(is_signed_comparison ? llvm::ICmpInst::ICMP_SLT
                                         : llvm::ICmpInst::ICMP_ULT,
                    compare_key2, compare_key1);
  // If key2 < key1
  auto if_smaller_data =
      EmitIfThenElse(comparison, "is_smaller_than", b, /*emit_else=*/false);
  SetToFirstInsertPoint(if_smaller_data.true_block, b);
  // Swap key1 with key2.
  write_element(0, keys_index, key2);
  write_element(0, compare_keys_index, key1);
  for (int64 i = 1; i <= num_values; ++i) {
    // Also swap the values.
    auto value1 = read_element(i, keys_index);
    auto value2 = read_element(i, compare_keys_index);
    write_element(i, keys_index, value2);
    write_element(i, compare_keys_index, value1);
  }
}

void EmitTiledCompareLoop(const IrArray::Index& tiled_keys_index,
                          int64 dimension_to_sort,
                          int64 dimension_to_sort_bound,
                          PrimitiveType keys_type,
                          absl::Span<const int64> xor_masks,
                          const std::vector<IrArray>& params, int64 tile_size,
                          llvm::IRBuilder<>* b) {
  IrArray::Index keys_index = tiled_keys_index;
  auto read_element = [&](int64 operand, llvm::Value* index) {
    keys_index[dimension_to_sort] = index;
    return params[operand].EmitReadArrayElement(keys_index, b);
  };
  auto write_element = [&](int64 operand, llvm::Value* index,
                           llvm::Value* value) {
    keys_index[dimension_to_sort] = index;
    params[operand].EmitWriteArrayElement(keys_index, value, b);
  };

  for (int64 xor_mask : xor_masks) {
    std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
        llvm_ir::ForLoop::EmitForLoop(
            "element_id_in_tile", tiled_keys_index.GetConstantWithIndexType(0),
            tiled_keys_index.GetConstantWithIndexType(tile_size),
            tiled_keys_index.GetConstantWithIndexType(1), b);
    llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(), b);
    auto current_keys_index = b->CreateAdd(
        b->CreateMul(tiled_keys_index[dimension_to_sort],
                     tiled_keys_index.GetConstantWithIndexType(tile_size)),
        tile_element_loop->GetIndVarValue());
    auto compare_keys_index =
        b->CreateXor(current_keys_index,
                     tiled_keys_index.GetConstantWithIndexType(xor_mask));
    EmitCompareLoopBody(
        tiled_keys_index.GetConstantWithIndexType(dimension_to_sort_bound),
        keys_type, params.size() - 1, current_keys_index, compare_keys_index,
        read_element, write_element, b);
    SetToFirstInsertPoint(tile_element_loop->GetExitBasicBlock(), b);
  }
}
}  // namespace

Status EmitSortInPlace(int64 dimension_to_sort, const IrArray& keys_array,
                       const std::vector<IrArray>& values_arrays,
                       absl::string_view name,
                       absl::Span<const int64> xor_masks, llvm::IRBuilder<>* b,
                       const gpu::LaunchDimensions& launch_dimensions,
                       const int64 tile_size) {
  // Iterate through the keys shape in physical order, but skip the dimension to
  // sort and make it the innermost loop which is the loop where the comparisons
  // happen. In the dimension to sort, if we use tiling, we iterate through it
  // in tiles of 64 elements each, so we use another loop that happens within
  // one thread to process this tile worth of data (thereby combining several
  // comparison stages of the bitonic sort algorithm because they all happen
  // within those 64 elements and are therefore independent of the other
  // comparisons).

  const Shape& keys_shape = keys_array.GetShape();
  int64 rank = ShapeUtil::Rank(keys_shape);
  int64 dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64 num_tiles = CeilOfRatio(dimension_to_sort_bound, tile_size);
  std::vector<int64> dimensions_in_iteration_order(rank);
  std::vector<int64> iteration_order_to_logical_order(rank);
  int64 dim = 0;
  for (int64 dimension : LayoutUtil::MinorToMajor(keys_shape)) {
    if (dimension != dimension_to_sort) {
      dimensions_in_iteration_order[dim] = keys_shape.dimensions(dimension);
      iteration_order_to_logical_order[dim++] = dimension;
    }
  }
  dimensions_in_iteration_order[dim] = dimension_to_sort_bound;
  iteration_order_to_logical_order[dim] = dimension_to_sort;

  Shape iteration_shape = ShapeUtil::MakeShape(keys_shape.element_type(),
                                               dimensions_in_iteration_order);
  Shape tiled_iteration_shape = iteration_shape;
  tiled_iteration_shape.set_dimensions(dim, num_tiles);
  std::vector<IrArray> params(1, keys_array);
  params.insert(params.end(), values_arrays.begin(), values_arrays.end());

  auto compare_loop_body_emitter =
      [&](const IrArray::Index& tiles_index) -> Status {
    // Naive C++ code for the inner compare loop:
    //
    // for (int64 i = 0; i < dimension_to_sort_bound; ++i) {
    //   int64 j = i ^ xor_mask;
    //   /* emitted in EmitCompareLoopBody() */
    //   if (i < j && j < dimension_to_sort_bound) {
    //     int64 min_key = std::min(keys[i], keys[j]);
    //     keys[j] = std::max(keys[i], keys[j]);
    //     keys[i] = min_key;
    //   }
    // }
    //
    // This follows the algorithm described on Wikipedia:
    // https://en.wikipedia.org/wiki/Bitonic_sorter
    IrArray::Index keys_index(tiles_index.GetType(), rank);
    for (int64 i = 0; i < rank; ++i) {
      keys_index[iteration_order_to_logical_order[i]] = tiles_index[i];
    }
    if (xor_masks.size() > 1) {
      EmitTiledCompareLoop(keys_index, dimension_to_sort,
                           dimension_to_sort_bound, keys_shape.element_type(),
                           xor_masks, params, tile_size, b);
    } else {
      auto read_element = [&](int64 operand, llvm::Value* index) {
        keys_index[dimension_to_sort] = index;
        return params[operand].EmitReadArrayElement(keys_index, b);
      };
      auto write_element = [&](int64 operand, llvm::Value* index,
                               llvm::Value* value) {
        keys_index[dimension_to_sort] = index;
        params[operand].EmitWriteArrayElement(keys_index, value, b);
      };
      auto current_keys_index = tiles_index[rank - 1];
      auto compare_keys_index =
          b->CreateXor(current_keys_index,
                       tiles_index.GetConstantWithIndexType(xor_masks[0]));
      EmitCompareLoopBody(
          tiles_index.GetConstantWithIndexType(dimension_to_sort_bound),
          keys_shape.element_type(), values_arrays.size(), current_keys_index,
          compare_keys_index, read_element, write_element, b);
    }
    return Status::OK();
  };
  return gpu::ParallelLoopEmitter(
             compare_loop_body_emitter,
             xor_masks.size() > 1 ? tiled_iteration_shape : iteration_shape,
             launch_dimensions, b)
      .EmitLoop(name);
}

}  // namespace llvm_ir
}  // namespace xla
