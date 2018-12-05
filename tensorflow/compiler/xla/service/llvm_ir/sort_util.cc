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
#include "absl/strings/str_cat.h"
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
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
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

// Adds the inner comparison loop body where we compare elements.
void EmitCompareLoopBody(
    int64 iteration_bound, PrimitiveType key_type, int64 num_values,
    int64 iota_values_parameter_index, llvm::Value* element_pair_index,
    int64 xor_mask, llvm::Type* index_type,
    std::function<llvm::Value*(int64 operand, llvm::Value* index)> read_element,
    std::function<void(int64 operand, llvm::Value* index, llvm::Value* value)>
        write_element,
    llvm::IRBuilder<>* b, bool needs_bounds_checks = true) {
  auto index_typed_constant = [&](int64 value) {
    return llvm::ConstantInt::get(index_type, value);
  };
  // The 'xor_mask' determines which elements are compared against each other.
  // Index 'current_keys_index' will be compared with 'current_keys_index' xor
  // 'xor_mask'. This means that we will always compare a block of consecutive
  // elements against elements from the adjacent block of the same size. When
  // 'xor_mask' is a power of 2, it immediately identifies the size of such a
  // block. We can also have 'xor_mask' being 2^k - 1 (for some value of k). In
  // that case, we essentially flip the last 'k' - 1 bits when computing the
  // position of the element to compare to, so the block size is 2^(k - 1).
  int64 block_size = xor_mask;
  // Check if it is a value 2^k - 1.
  if (xor_mask > 1 && (xor_mask & (xor_mask + 1)) == 0) {
    block_size = (xor_mask + 1) / 2;
  }
  auto current_keys_index = element_pair_index;
  if (block_size == 1) {
    // If the block size is 1, we take every second element and compare it to
    // the next one.
    current_keys_index =
        b->CreateMul(current_keys_index, index_typed_constant(2));
  } else if (block_size * 2 < iteration_bound) {
    // current_keys_index iterates through the 'left' elements of the element
    // pairs to be compared. We first need to compute the comparison block to
    // which the element belongs. The block id of that block is index /
    // block_size.
    auto block_id =
        b->CreateUDiv(current_keys_index, index_typed_constant(block_size));
    // The index of the 'left' element within its block is simply the remainder
    // when dividing by 'block_size'.
    auto index_within_block =
        b->CreateURem(current_keys_index, index_typed_constant(block_size));
    // The first element of the 'left' block of elements that is compared
    // against elements from the adjacent 'right' block of elements is
    // 'block_id' * (2 * 'block_size').
    auto first_element_in_block =
        b->CreateMul(block_id, index_typed_constant(2 * block_size));
    current_keys_index =
        b->CreateAdd(first_element_in_block, index_within_block);
  }
  auto compare_keys_index =
      b->CreateXor(current_keys_index, index_typed_constant(xor_mask));
  // current_keys_index < compare_keys_index
  llvm::Value* is_smaller_index =
      b->CreateICmpSLT(current_keys_index, compare_keys_index);
  // compare_keys_index < iteration_bound
  llvm::Value* index_is_inbounds = b->CreateICmpSLT(
      compare_keys_index, index_typed_constant(iteration_bound));
  llvm::Value* do_comparison =
      needs_bounds_checks ? b->CreateAnd(is_smaller_index, index_is_inbounds)
                          : b->getInt1(true);

  // if (is_smaller_index && index_is_inbounds)
  KernelSupportLibrary ksl(b);
  ksl.IfReturnVoid("smaller_comparison_index", do_comparison, [&]() {
    auto key1 = read_element(0, current_keys_index);
    auto key2 = read_element(0, compare_keys_index);
    auto compare_key1 = key1;
    auto compare_key2 = key2;
    bool is_signed_comparison = true;
    if (primitive_util::IsFloatingPointType(key_type)) {
      // We would like a total order of floating point numbers so that the
      // sort has a predictable behavior in the presence of NaNs. Rather
      // than using floating point comparison, we use the following trick:
      // If f is a float, and
      // x = bit_cast<int32>(f);
      // y = x < 0 ? 0x7FFFFFFF - x : x;
      // then y is ordered as an int32 such that finite values have the
      // obvious order, -0 is ordered before 0, and -NaN and NaN appear at
      // the beginning and end of the ordering.
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
    // If key2 < key1
    auto is_smaller_than =
        b->CreateICmp(is_signed_comparison ? llvm::ICmpInst::ICMP_SLT
                                           : llvm::ICmpInst::ICMP_ULT,
                      compare_key2, compare_key1);
    if (iota_values_parameter_index >= 0) {
      auto keys_equal = b->CreateICmpEQ(compare_key1, compare_key2);
      auto key_index1 =
          read_element(iota_values_parameter_index, current_keys_index);
      auto key_index2 =
          read_element(iota_values_parameter_index, compare_keys_index);
      auto index_is_smaller_than =
          b->CreateICmp(llvm::ICmpInst::ICMP_ULT, key_index2, key_index1);
      is_smaller_than = b->CreateOr(
          is_smaller_than, b->CreateAnd(keys_equal, index_is_smaller_than));
    }
    ksl.IfReturnVoid("is_smaller_than", is_smaller_than, [&]() {
      // Swap key1 with key2.
      write_element(0, current_keys_index, key2);
      write_element(0, compare_keys_index, key1);
      for (int64 i = 1; i <= num_values; ++i) {
        // Also swap the values.
        auto value1 = read_element(i, current_keys_index);
        auto value2 = read_element(i, compare_keys_index);
        write_element(i, current_keys_index, value2);
        write_element(i, compare_keys_index, value1);
      }
    });
  });
}

void EmitTiledCompareLoop(
    const IrArray::Index& tiled_keys_index, int64 dimension_to_sort,
    int64 dimension_to_sort_bound, PrimitiveType keys_type,
    absl::Span<const int64> xor_masks, const std::vector<IrArray>& params,
    const std::vector<llvm::Value*>& param_shmem_buffers,
    int64 iota_values_parameter_index, int64 tile_size, llvm::IRBuilder<>* b) {
  KernelSupportLibrary ksl(b);
  llvm::Value* thread_id = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {}, b);
  llvm_ir::AddRangeMetadata(0, tile_size / 2,
                            llvm::cast<llvm::Instruction>(thread_id));
  thread_id = b->CreateIntCast(thread_id, tiled_keys_index.GetType(),
                               /*isSigned=*/true, "thread.id.x");

  auto copy_loop_body =
      [&](std::function<void(llvm::Value * cache_index, llvm::Value * index)>
              read_or_write) {
        auto value_one = tiled_keys_index.GetConstantWithIndexType(1);
        auto current_keys_index =
            b->CreateShl(tiled_keys_index[dimension_to_sort], value_one);
        // We want to copy two adjacent elements. We first check whether the
        // first index position is within bounds.
        ksl.IfReturnVoid(
            "smaller_keys_index",
            b->CreateICmpSLT(current_keys_index,
                             tiled_keys_index.GetConstantWithIndexType(
                                 dimension_to_sort_bound)),
            [&]() {
              auto cache_index = b->CreateShl(thread_id, value_one);
              read_or_write(cache_index, current_keys_index);
              // Increment to go the next index position.
              current_keys_index = b->CreateAdd(current_keys_index, value_one);
              // Here we check whether the next index position is within bounds.
              ksl.IfReturnVoid(
                  "inner_smaller_keys_index",
                  b->CreateICmpSLT(current_keys_index,
                                   tiled_keys_index.GetConstantWithIndexType(
                                       dimension_to_sort_bound)),
                  [&]() {
                    cache_index = b->CreateAdd(cache_index, value_one);
                    read_or_write(cache_index, current_keys_index);
                  });
            });
      };

  // Copy operand tiles from the operand buffers to shared memory.
  IrArray::Index keys_index = tiled_keys_index;
  for (int64 i = 0; i < params.size(); ++i) {
    copy_loop_body([&](llvm::Value* cache_index, llvm::Value* index) {
      keys_index[dimension_to_sort] = index;
      auto value = params[i].EmitReadArrayElement(keys_index, b);
      b->CreateStore(value,
                     b->CreateGEP(param_shmem_buffers[i],
                                  {tiled_keys_index.GetConstantWithIndexType(0),
                                   cache_index}));
    });
  }
  // Wait until all reads have happened.
  llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {}, b);

  // Now emit the bodies of the comparison loops.
  auto read_element = [&](int64 operand, llvm::Value* index) {
    return b->CreateLoad(
        b->CreateGEP(param_shmem_buffers[operand],
                     {tiled_keys_index.GetConstantWithIndexType(0), index}));
  };
  auto write_element = [&](int64 operand, llvm::Value* index,
                           llvm::Value* value) {
    b->CreateStore(
        value,
        b->CreateGEP(param_shmem_buffers[operand],
                     {tiled_keys_index.GetConstantWithIndexType(0), index}));
  };
  for (int64 xor_mask : xor_masks) {
    // The index of the element pair to be compared within the tile stored in
    // shared memory. We order the element pairs by the element with the smaller
    // index.
    auto element_pair_index = thread_id;
    // If 'dimension_to_sort_bound' is evenly divisible by 'tile_size', we don't
    // need any bounds checks.
    if (dimension_to_sort_bound % tile_size) {
      // Otherwise we need a bounds check for the last tile. The last tile has
      // size 'dimension_to_sort_bound' % 'tile_size'.
      ksl.IfReturnVoid(
          "is_last_tile",
          b->CreateICmpUGE(
              b->CreateMul(tiled_keys_index[dimension_to_sort],
                           tiled_keys_index.GetConstantWithIndexType(2)),
              tiled_keys_index.GetConstantWithIndexType(
                  RoundDownToNearest(dimension_to_sort_bound, tile_size))),
          [&]() {
            EmitCompareLoopBody(dimension_to_sort_bound % tile_size, keys_type,
                                params.size() - 1, iota_values_parameter_index,
                                element_pair_index, xor_mask,
                                tiled_keys_index.GetType(), read_element,
                                write_element, b);
          },
          [&]() {
            EmitCompareLoopBody(tile_size, keys_type, params.size() - 1,
                                iota_values_parameter_index, element_pair_index,
                                xor_mask, tiled_keys_index.GetType(),
                                read_element, write_element, b,
                                /*needs_bounds_checks=*/false);
          });
    } else {
      EmitCompareLoopBody(tile_size, keys_type, params.size() - 1,
                          iota_values_parameter_index, element_pair_index,
                          xor_mask, tiled_keys_index.GetType(), read_element,
                          write_element, b, /*needs_bounds_checks=*/false);
    }
    // Wait until all comparisons have happened.
    llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {}, b);
  }

  // Copy the operand tiles back from shared memory to the operand buffers.
  for (int64 i = 0; i < params.size(); ++i) {
    copy_loop_body([&](llvm::Value* cache_index, llvm::Value* index) {
      keys_index[dimension_to_sort] = index;
      auto value = b->CreateLoad(b->CreateGEP(
          param_shmem_buffers[i],
          {tiled_keys_index.GetConstantWithIndexType(0), cache_index}));
      params[i].EmitWriteArrayElement(keys_index, value, b);
    });
  }
  // We should normally synchronize here to make sure all writes have happened.
  // However the very next thing each thread does is reading 2 elements from the
  // operand buffer and writing it into the same location in shared memory from
  // which it previously copied it to the operand buffer, and we synchronize
  // after this has happened. We can be sure that a thread always writes to the
  // same location in shared memory because we have exactly tile_size / 2 many
  // threads, and the linear index calculated by ParallelLoopEmitter uses
  // linear_index = blockIdx.x * blockDim.x + threadIdx.x;
}
}  // namespace

Status EmitSortInPlace(int64 dimension_to_sort, const IrArray& keys_array,
                       const std::vector<IrArray>& values_arrays,
                       int64 iota_values_parameter_index,
                       absl::string_view name,
                       absl::Span<const int64> xor_masks, llvm::IRBuilder<>* b,
                       const gpu::LaunchDimensions& launch_dimensions,
                       int64 num_iterations_in_sort_dim,
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
  std::vector<int64> dimensions_in_iteration_order(rank);
  std::vector<int64> iteration_order_to_logical_order(rank);
  int64 dim = 0;
  for (int64 dimension : LayoutUtil::MinorToMajor(keys_shape)) {
    if (dimension != dimension_to_sort) {
      dimensions_in_iteration_order[dim] = keys_shape.dimensions(dimension);
      iteration_order_to_logical_order[dim++] = dimension;
    }
  }
  dimensions_in_iteration_order[dim] = num_iterations_in_sort_dim;
  iteration_order_to_logical_order[dim] = dimension_to_sort;

  Shape iteration_shape = ShapeUtil::MakeShape(keys_shape.element_type(),
                                               dimensions_in_iteration_order);
  std::vector<IrArray> params(1, keys_array);
  params.insert(params.end(), values_arrays.begin(), values_arrays.end());

  // Allocate shared memory for the tiled compare loop.
  std::vector<llvm::Value*> param_shmem_buffers(params.size(), nullptr);
  if (xor_masks.size() > 1) {
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    for (int64 i = 0; i < params.size(); ++i) {
      llvm::Type* tile_type =
          llvm::ArrayType::get(llvm_ir::PrimitiveTypeToIrType(
                                   params[i].GetShape().element_type(), module),
                               tile_size);
      param_shmem_buffers[i] = llvm_ir::AllocateSharedMemoryTile(
          module, tile_type, absl::StrCat(name, "_tile_param_", i));
    }
  }

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
                           xor_masks, params, param_shmem_buffers,
                           iota_values_parameter_index, tile_size, b);
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
      EmitCompareLoopBody(dimension_to_sort_bound, keys_shape.element_type(),
                          values_arrays.size(), iota_values_parameter_index,
                          tiles_index[rank - 1], xor_masks[0],
                          tiles_index.GetType(), read_element, write_element,
                          b);
    }
    return Status::OK();
  };
  return gpu::ParallelLoopEmitter(compare_loop_body_emitter, iteration_shape,
                                  launch_dimensions, b)
      .EmitLoop(name);
}

}  // namespace llvm_ir
}  // namespace xla
