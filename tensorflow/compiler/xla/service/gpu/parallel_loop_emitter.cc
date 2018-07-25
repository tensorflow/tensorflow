/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"

#include <memory>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

ParallelLoopEmitter::ParallelLoopEmitter(
    BodyEmitter body_emitter, const Shape& shape,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,
    int unroll_factor)
    : LoopEmitter(body_emitter, shape, b),
      launch_dimensions_(launch_dimensions),
      unroll_factor_(unroll_factor) {}

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    tensorflow::gtl::ArraySlice<llvm_ir::IrArray> target_arrays,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,
    int unroll_factor)
    : LoopEmitter(target_element_generator, target_arrays, b),
      launch_dimensions_(launch_dimensions),
      unroll_factor_(unroll_factor) {}

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    const llvm_ir::IrArray& target_array,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,
    int unroll_factor)
    : LoopEmitter(target_element_generator, target_array, b),
      launch_dimensions_(launch_dimensions),
      unroll_factor_(unroll_factor) {}

std::vector<llvm_ir::IrArray::Index>
ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock(
    tensorflow::StringPiece loop_name, llvm::Type* index_type) {
  // Emit the following code in LLVM IR:
  //   linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (linear_index < num_elements) {
  //     array_index = LinearIndexToMultidimensionalIndex(shape_, linear_index);
  //     ...
  //   }

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %ctaid.x <  %nctaid.x"
  //
  // %nctaid.x is currently specified as 2147483647.
  VLOG(3) << "EmitIndexAndSetExitBasicBlock unroll_factor " << unroll_factor_;
  CHECK_NE(index_type, nullptr);
  std::vector<llvm_ir::IrArray::Index> array_indices;
  llvm::Value* block_id = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.block_count(),
                            static_cast<llvm::Instruction*>(block_id));
  block_id = b_->CreateZExtOrTrunc(block_id, index_type, "block_id");

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %tid.x <  %ntid.x"
  //
  // %ntid.x is currently specified as 1024.
  llvm::Value* thread_id = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.threads_per_block(),
                            static_cast<llvm::Instruction*>(thread_id));
  thread_id = b_->CreateZExtOrTrunc(thread_id, index_type, "thread_id");

  llvm::Value* linear_index_base = b_->CreateAdd(
      b_->CreateMul(block_id,
                    llvm::ConstantInt::get(
                        index_type, launch_dimensions_.threads_per_block()),
                    "",
                    /*HasNUW=*/true, /*HasNSW=*/true),
      thread_id, "linear_index", /*HasNUW=*/true, /*HasNSW=*/true);

  // Add an @llvm.assume(linear_index < threads_per_block * num_blocks).
  //
  // This might seem obvious from the computation above, but LLVM does not
  // currently determine the range of linear_index precisely.  InstCombine uses
  // known-bits, which, when applied to the task of determining a value's range,
  // is imprecise for everything other than powers of 2.  And
  // CorrelatedValuePropagation is, as a cost-saving measure, disabled for
  // conditions in the same basic block as their operands.
  llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::assume,
      {b_->CreateICmpULT(
          linear_index_base,
          llvm::ConstantInt::get(index_type,
                                 launch_dimensions_.threads_per_block() *
                                     launch_dimensions_.block_count()),
          "linear_index_in_range")},
      {}, b_);

  if (unroll_factor_ > 1) {
    linear_index_base = b_->CreateMul(
        linear_index_base, llvm::ConstantInt::get(index_type, unroll_factor_),
        "linear_index_base", /*HasNUW=*/true, /*HasNSW=*/true);
  }

  array_indices.emplace_back(linear_index_base, shape_, b_);
  for (int i = 1; i < unroll_factor_; ++i) {
    llvm::Value* linear_index =
        b_->CreateAdd(linear_index_base, llvm::ConstantInt::get(index_type, i),
                      "linear_index",
                      /*HasNUW=*/true, /*HasNSW=*/true);
    array_indices.emplace_back(linear_index, shape_, b_);
  }

  auto if_in_bounds = llvm_ir::EmitIfThenElse(
      b_->CreateICmpULT(
          linear_index_base,
          llvm::ConstantInt::get(index_type, ShapeUtil::ElementsIn(shape_))),
      llvm_ir::IrName(loop_name, "in_bounds"), b_, false);

  // Set exit_bb_ to the exit block of the if structure.
  exit_bb_ = if_in_bounds.after_block;
  CHECK_NE(nullptr, exit_bb_);

  // Set IR builder insertion point to the body of the if structure.
  llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, b_);

  return array_indices;
}

}  // namespace gpu
}  // namespace xla
