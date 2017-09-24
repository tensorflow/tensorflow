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
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* ir_builder)
    : LoopEmitter(body_emitter, shape, ir_builder),
      launch_dimensions_(launch_dimensions) {}

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    tensorflow::gtl::ArraySlice<llvm_ir::IrArray> target_arrays,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* ir_builder)
    : LoopEmitter(target_element_generator, target_arrays, ir_builder),
      launch_dimensions_(launch_dimensions) {}

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    const llvm_ir::IrArray& target_array,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* ir_builder)
    : LoopEmitter(target_element_generator, target_array, ir_builder),
      launch_dimensions_(launch_dimensions) {}

llvm_ir::IrArray::Index ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock(
    tensorflow::StringPiece loop_name) {
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
  llvm::Value* block_id = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, {}, ir_builder_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.block_count(),
                            static_cast<llvm::Instruction*>(block_id));
  block_id =
      ir_builder_->CreateZExt(block_id, ir_builder_->getInt64Ty(), "block_id");

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %tid.x <  %ntid.x"
  //
  // %ntid.x is currently specified as 1024.
  llvm::Value* thread_id = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {}, ir_builder_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.threads_per_block(),
                            static_cast<llvm::Instruction*>(thread_id));
  thread_id = ir_builder_->CreateZExt(thread_id, ir_builder_->getInt64Ty(),
                                      "thread_id");

  llvm::Value* linear_index = ir_builder_->CreateAdd(
      ir_builder_->CreateMul(
          block_id,
          ir_builder_->getInt64(launch_dimensions_.threads_per_block()), "",
          /*HasNUW=*/true, /*HasNSW=*/true),
      thread_id, "linear_index", /*HasNUW=*/true, /*HasNSW=*/true);

  auto if_in_bounds = llvm_ir::EmitIfThenElse(
      ir_builder_->CreateICmpULT(
          linear_index, ir_builder_->getInt64(ShapeUtil::ElementsIn(shape_))),
      llvm_ir::IrName(loop_name, "in_bounds"), ir_builder_, false);

  // Set exit_bb_ to the exit block of the if structure.
  exit_bb_ = if_in_bounds.after_block;
  CHECK_NE(nullptr, exit_bb_);

  // Set IR builder insertion point to the body of the if structure.
  llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, ir_builder_);
  return llvm_ir::IrArray::Index(linear_index, shape_, ir_builder_);
}

}  // namespace gpu
}  // namespace xla
