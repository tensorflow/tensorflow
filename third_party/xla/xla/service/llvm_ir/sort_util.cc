/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/sort_util.h"

#include <algorithm>
#include <functional>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/status.h"

namespace xla {
namespace llvm_ir {

namespace {

// Adds the inner comparison loop body where we compare elements.
absl::Status EmitCompareLoopBody(
    int64_t iteration_bound, int64_t num_values,
    llvm::Value* element_pair_index, int64_t xor_mask, llvm::Type* index_type,
    std::function<llvm::Value*(int64_t operand, llvm::Value* index)>
        element_address,
    std::function<llvm::Type*(int64_t operand, llvm::Value* index)>
        element_address_pointee_type,
    std::function<void(int64_t operand, llvm::Value* index, llvm::Value* value)>
        write_element,
    const EmitCallToNestedComputationCallback& emit_compare_callback,
    llvm::IRBuilder<>* b, bool needs_bounds_checks = true) {
  auto index_typed_constant = [&](int64_t value) {
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
  int64_t block_size = xor_mask;
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
  return ksl.IfWithStatus("smaller_comparison_index", do_comparison, [&]() {
    std::vector<llvm::Value*> values_to_compare;
    std::vector<llvm::Type*> values_to_compare_types;
    for (int i = 0; i < num_values; ++i) {
      values_to_compare.push_back(element_address(i, compare_keys_index));
      values_to_compare_types.push_back(
          element_address_pointee_type(i, compare_keys_index));

      values_to_compare.push_back(element_address(i, current_keys_index));
      values_to_compare_types.push_back(
          element_address_pointee_type(i, current_keys_index));
    }
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    llvm::Type* pred_type = llvm_ir::PrimitiveTypeToIrType(PRED, module);
    llvm::Value* compare_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        pred_type, "compare_return_buffer", b);
    TF_RETURN_IF_ERROR(
        emit_compare_callback(values_to_compare, compare_return_buffer));
    llvm::Value* result = b->CreateLoad(pred_type, compare_return_buffer);

    // Check if the 'compare' function returns true.
    llvm::Value* is_smaller_than =
        b->CreateICmpNE(result, llvm::ConstantInt::get(result->getType(), 0),
                        "boolean_predicate");
    ksl.If("is_smaller_than", is_smaller_than, [&]() {
      for (int64_t i = 0; i < num_values; ++i) {
        // Swap the values.
        auto value1 = b->CreateLoad(values_to_compare_types[i * 2],
                                    values_to_compare[i * 2]);
        auto value2 = b->CreateLoad(values_to_compare_types[i * 2 + 1],
                                    values_to_compare[i * 2 + 1]);
        write_element(i, current_keys_index, value1);
        write_element(i, compare_keys_index, value2);
      }
    });
    return absl::OkStatus();
  });
}

absl::Status EmitTiledCompareLoop(
    const IrArray::Index& tiled_keys_index, int64_t dimension_to_sort,
    int64_t dimension_to_sort_bound, absl::Span<const int64_t> xor_masks,
    const std::vector<IrArray>& params,
    const std::vector<llvm::GlobalVariable*>& param_shmem_buffers,
    int64_t tile_size,
    const EmitCallToNestedComputationCallback& emit_compare_callback,
    llvm::IRBuilder<>* b) {
  KernelSupportLibrary ksl(b);
  llvm::Value* thread_id = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kThreadIdx, {}, {}, b);
  llvm_ir::AddRangeMetadata(0, tile_size / 2,
                            llvm::cast<llvm::Instruction>(thread_id),
                            b->GetInsertBlock()->getModule());
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
        ksl.If(
            "smaller_keys_index",
            b->CreateICmpSLT(current_keys_index,
                             tiled_keys_index.GetConstantWithIndexType(
                                 dimension_to_sort_bound)),
            [&]() {
              auto cache_index = b->CreateShl(thread_id, value_one);
              read_or_write(cache_index, current_keys_index);
              // Increment to go to the next index position.
              current_keys_index = b->CreateAdd(current_keys_index, value_one);
              // Here we check whether the next index position is within bounds.
              ksl.If("inner_smaller_keys_index",
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
  std::vector<llvm::Value*> keys_multi_index = tiled_keys_index.multidim();
  for (int64_t i = 0; i < params.size(); ++i) {
    copy_loop_body([&](llvm::Value* cache_index, llvm::Value* index) {
      keys_multi_index[dimension_to_sort] = index;
      IrArray::Index keys_index(keys_multi_index, params[i].GetShape(),
                                tiled_keys_index.GetType());
      auto value = params[i].EmitReadArrayElement(keys_index, b);
      b->CreateStore(
          value,
          b->CreateGEP(
              param_shmem_buffers[i]->getValueType(), param_shmem_buffers[i],
              {tiled_keys_index.GetConstantWithIndexType(0), cache_index}));
    });
  }
  // Wait until all reads have happened.
  gpu::EmitCallToTargetIntrinsic(gpu::TargetIntrinsicID::kBarrierId, {}, {}, b);

  // Now emit the bodies of the comparison loops.
  auto element_address = [&](int64_t operand, llvm::Value* index) {
    auto shared_memory_address =
        b->CreateGEP(param_shmem_buffers[operand]->getValueType(),
                     param_shmem_buffers[operand],
                     {tiled_keys_index.GetConstantWithIndexType(0), index});
    auto ptr_type = shared_memory_address->getType();
    // We need a generic pointer with address space 0 instead of a pointer to
    // shared memory (address space 3) so that we can pass it to the comparison
    // computation.
    return b->CreateAddrSpaceCast(
        shared_memory_address,
        llvm::PointerType::get(
            llvm::cast<llvm::PointerType>(ptr_type)->getContext(),
            /*AddressSpace=*/0));
  };
  auto element_address_pointee_type = [&](int64_t operand, llvm::Value* index) {
    return llvm::GetElementPtrInst::getIndexedType(
        param_shmem_buffers[operand]->getValueType(),
        {tiled_keys_index.GetConstantWithIndexType(0), index});
  };
  auto write_element = [&](int64_t operand, llvm::Value* index,
                           llvm::Value* value) {
    b->CreateStore(
        value,
        b->CreateGEP(param_shmem_buffers[operand]->getValueType(),
                     param_shmem_buffers[operand],
                     {tiled_keys_index.GetConstantWithIndexType(0), index}));
  };
  for (int64_t xor_mask : xor_masks) {
    // The index of the element pair to be compared within the tile stored in
    // shared memory. We order the element pairs by the element with the smaller
    // index.
    auto element_pair_index = thread_id;
    // If 'dimension_to_sort_bound' is evenly divisible by 'tile_size', we don't
    // need any bounds checks.
    if (dimension_to_sort_bound % tile_size) {
      // Otherwise we need a bounds check for the last tile. The last tile has
      // size 'dimension_to_sort_bound' % 'tile_size'.
      TF_RETURN_IF_ERROR(ksl.IfWithStatus(
          "is_last_tile",
          b->CreateICmpUGE(
              b->CreateMul(tiled_keys_index[dimension_to_sort],
                           tiled_keys_index.GetConstantWithIndexType(2)),
              tiled_keys_index.GetConstantWithIndexType(
                  RoundDownTo(dimension_to_sort_bound, tile_size))),
          [&]() {
            return EmitCompareLoopBody(
                dimension_to_sort_bound % tile_size, params.size(),
                element_pair_index, xor_mask, tiled_keys_index.GetType(),
                element_address, element_address_pointee_type, write_element,
                emit_compare_callback, b);
          },
          [&]() {
            return EmitCompareLoopBody(
                tile_size, params.size(), element_pair_index, xor_mask,
                tiled_keys_index.GetType(), element_address,
                element_address_pointee_type, write_element,
                emit_compare_callback, b,
                /*needs_bounds_checks=*/false);
          }));
    } else {
      TF_RETURN_IF_ERROR(EmitCompareLoopBody(
          tile_size, params.size(), element_pair_index, xor_mask,
          tiled_keys_index.GetType(), element_address,
          element_address_pointee_type, write_element, emit_compare_callback, b,
          /*needs_bounds_checks=*/false));
    }
    // Wait until all comparisons have happened.
    gpu::EmitCallToTargetIntrinsic(gpu::TargetIntrinsicID::kBarrierId, {}, {},
                                   b);
  }

  // Copy the operand tiles back from shared memory to the operand buffers.
  for (int64_t i = 0; i < params.size(); ++i) {
    copy_loop_body([&](llvm::Value* cache_index, llvm::Value* index) {
      keys_multi_index[dimension_to_sort] = index;
      IrArray::Index keys_index(keys_multi_index, params[i].GetShape(),
                                tiled_keys_index.GetType());
      auto gep = b->CreateGEP(
          param_shmem_buffers[i]->getValueType(), param_shmem_buffers[i],
          {tiled_keys_index.GetConstantWithIndexType(0), cache_index});
      auto gep_type = llvm::GetElementPtrInst::getIndexedType(
          param_shmem_buffers[i]->getValueType(),
          {tiled_keys_index.GetConstantWithIndexType(0), cache_index});
      auto value = b->CreateLoad(gep_type, gep);
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
  return absl::OkStatus();
}
}  // namespace

absl::Status EmitSortInPlace(
    int64_t dimension_to_sort, const std::vector<IrArray>& values_arrays,
    absl::string_view name, absl::Span<const int64_t> xor_masks,
    llvm::IRBuilder<>* b, const gpu::LaunchDimensions& launch_dimensions,
    int64_t num_iterations_in_sort_dim, const int64_t tile_size,
    const EmitCallToNestedComputationCallback& emit_compare_callback) {
  // Iterate through the keys shape in physical order, but skip the dimension to
  // sort and make it the innermost loop which is the loop where the comparisons
  // happen. In the dimension to sort, if we use tiling, we iterate through it
  // in tiles of 64 elements each, so we use another loop that happens within
  // one thread to process this tile worth of data (thereby combining several
  // comparison stages of the bitonic sort algorithm because they all happen
  // within those 64 elements and are therefore independent of the other
  // comparisons).

  const Shape& keys_shape = values_arrays[0].GetShape();
  int64_t rank = keys_shape.rank();
  int64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  std::vector<int64_t> dimensions_in_iteration_order(rank);
  std::vector<int64_t> iteration_order_to_logical_order(rank);
  int64_t dim = 0;
  for (int64_t dimension : LayoutUtil::MinorToMajor(keys_shape)) {
    if (dimension != dimension_to_sort) {
      dimensions_in_iteration_order[dim] = keys_shape.dimensions(dimension);
      iteration_order_to_logical_order[dim++] = dimension;
    }
  }
  dimensions_in_iteration_order[dim] = num_iterations_in_sort_dim;
  iteration_order_to_logical_order[dim] = dimension_to_sort;

  Shape iteration_shape = ShapeUtil::MakeShape(keys_shape.element_type(),
                                               dimensions_in_iteration_order);

  // Allocate shared memory for the tiled compare loop.
  // We process 64 elements at a time, so the buffer cannot be less than that.
  std::vector<llvm::GlobalVariable*> param_shmem_buffers(values_arrays.size(),
                                                         nullptr);
  if (xor_masks.size() > 1) {
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    for (int64_t i = 0; i < values_arrays.size(); ++i) {
      llvm::Type* tile_type = llvm::ArrayType::get(
          llvm_ir::PrimitiveTypeToIrType(
              values_arrays[i].GetShape().element_type(), module),
          std::max(tile_size, static_cast<int64_t>(64)));
      param_shmem_buffers[i] = llvm_ir::AllocateSharedMemoryTile(
          module, tile_type, absl::StrCat(name, "_tile_param_", i));
    }
  }

  auto compare_loop_body_emitter =
      [&](const IrArray::Index& tiles_index) -> absl::Status {
    // Naive C++ code for the inner compare loop:
    //
    // for (int64_t i = 0; i < dimension_to_sort_bound; ++i) {
    //   int64_t j = i ^ xor_mask;
    //   /* emitted in EmitCompareLoopBody() */
    //   if (i < j && j < dimension_to_sort_bound) {
    //     int64_t min_key = std::min(keys[i], keys[j]);
    //     keys[j] = std::max(keys[i], keys[j]);
    //     keys[i] = min_key;
    //   }
    // }
    //
    // This follows the algorithm described on Wikipedia:
    // https://en.wikipedia.org/wiki/Bitonic_sorter
    std::vector<llvm::Value*> keys_multi_index(rank);
    for (int64_t i = 0; i < rank; ++i) {
      keys_multi_index[iteration_order_to_logical_order[i]] = tiles_index[i];
    }
    if (xor_masks.size() > 1) {
      IrArray::Index keys_index(keys_multi_index, values_arrays[0].GetShape(),
                                tiles_index.GetType());
      TF_RETURN_IF_ERROR(EmitTiledCompareLoop(
          keys_index, dimension_to_sort, dimension_to_sort_bound, xor_masks,
          values_arrays, param_shmem_buffers, tile_size, emit_compare_callback,
          b));
    } else {
      auto element_address = [&](int64_t operand, llvm::Value* index) {
        keys_multi_index[dimension_to_sort] = index;
        IrArray::Index keys_index(keys_multi_index,
                                  values_arrays[operand].GetShape(),
                                  tiles_index.GetType());
        return values_arrays[operand].EmitArrayElementAddress(keys_index, b);
      };
      auto element_address_pointee_type = [&](int64_t operand, llvm::Value*) {
        return values_arrays[operand].GetElementLlvmType();
      };
      auto write_element = [&](int64_t operand, llvm::Value* index,
                               llvm::Value* value) {
        keys_multi_index[dimension_to_sort] = index;
        IrArray::Index keys_index(keys_multi_index,
                                  values_arrays[operand].GetShape(),
                                  tiles_index.GetType());
        values_arrays[operand].EmitWriteArrayElement(keys_index, value, b);
      };
      TF_RETURN_IF_ERROR(EmitCompareLoopBody(
          dimension_to_sort_bound, values_arrays.size(), tiles_index[rank - 1],
          xor_masks[0], tiles_index.GetType(), element_address,
          element_address_pointee_type, write_element, emit_compare_callback,
          b));
    }
    return absl::OkStatus();
  };
  return gpu::ParallelLoopEmitter(compare_loop_body_emitter, iteration_shape,
                                  launch_dimensions, b)
      .EmitLoop(name);
}

}  // namespace llvm_ir
}  // namespace xla
