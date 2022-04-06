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

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

ParallelLoopEmitter::ParallelLoopEmitter(
    llvm_ir::BodyEmitter body_emitter, const Shape& shape,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,
    LaunchDimensionsConfig launch_config)
    : launch_dimensions_(launch_dimensions),
      launch_config_(launch_config),
      body_emitter_(body_emitter),
      shape_(shape),
      b_(b) {}

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    absl::Span<const llvm_ir::IrArray> target_arrays,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,

    LaunchDimensionsConfig launch_config)
    : launch_dimensions_(launch_dimensions),
      launch_config_(launch_config),
      body_emitter_(
          llvm_ir::MakeBodyEmitter(target_element_generator, target_arrays, b,
                                   /*is_tuple=*/target_arrays.size() > 1)),
      shape_(target_arrays[0].GetShape()),
      b_(b) {}

ParallelLoopEmitter::LinearBaseAndThreadIdx
ParallelLoopEmitter::EmitLinearBaseAndThreadIdx(llvm::Type* index_type,
                                                llvm::Value* base_index) {
  llvm::Value* block_id =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.block_counts().x,
                            static_cast<llvm::Instruction*>(block_id));
  block_id = b_->CreateZExtOrTrunc(block_id, index_type, "block_id");

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %tid.x <  %ntid.x"
  llvm::Value* thread_id_x =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.thread_counts_per_block().x,
                            static_cast<llvm::Instruction*>(thread_id_x));
  thread_id_x = b_->CreateZExtOrTrunc(thread_id_x, index_type, "thread_id_x");

  const int unroll_factor =
      launch_config_.unroll_factor > 1 ? launch_config_.unroll_factor : 1;

  // Linear base is different for logical order vs physical order stores.
  // For logical,  linear_base = block_id*num_threads*unroll + thread_idx
  // For physical, linear_base = (block_id*num_threads + thread_idx)*unroll
  int block_id_multiplier =
      launch_config_.logical_order
          ? launch_dimensions_.total_nb_threads() * unroll_factor
          : launch_dimensions_.total_nb_threads();

  llvm::Value* linear_index_base = b_->CreateMul(
      block_id, llvm::ConstantInt::get(index_type, block_id_multiplier), "",
      /*HasNUW=*/true, /*HasNSW=*/true);

  linear_index_base =
      b_->CreateAdd(linear_index_base, thread_id_x, "linear_index",
                    /*HasNUW=*/true, /*HasNSW=*/true);

  if (launch_dimensions_.thread_counts_per_block().y > 1) {
    CHECK(!launch_config_.logical_order);
    llvm::Value* thread_id_y =
        EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdy, {}, {}, b_);
    llvm_ir::AddRangeMetadata(0, launch_dimensions_.thread_counts_per_block().y,
                              static_cast<llvm::Instruction*>(thread_id_y));
    thread_id_y = b_->CreateZExtOrTrunc(thread_id_y, index_type, "thread_id_y");
    linear_index_base = b_->CreateAdd(
        linear_index_base,
        b_->CreateMul(
            thread_id_y,
            llvm::ConstantInt::get(
                index_type, launch_dimensions_.thread_counts_per_block().x),
            "",
            /*HasNUW=*/true, /*HasNSW=*/true),
        "",
        /*HasNUW=*/true, /*HasNSW=*/true);
  }

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
          llvm::ConstantInt::get(
              index_type,
              block_id_multiplier * launch_dimensions_.block_counts().x),
          "linear_index_in_range")},
      {}, b_);

  if (!launch_config_.logical_order && launch_config_.unroll_factor > 1) {
    linear_index_base = b_->CreateMul(
        linear_index_base,
        llvm::ConstantInt::get(index_type, launch_config_.unroll_factor),
        "linear_index_base", /*HasNUW=*/true, /*HasNSW=*/true);
  }

  if (base_index != nullptr) {
    linear_index_base =
        b_->CreateAdd(linear_index_base, base_index, "linear_index_plus_base",
                      /*HasNUW=*/true, /*HasNSW=*/true);
  }
  return {linear_index_base, thread_id_x};
}

std::vector<llvm_ir::IrArray::Index>
ParallelLoopEmitter::EmitLogicalIndexAndSetExitBasicBlock(
    absl::string_view loop_name, llvm::Type* index_type,
    llvm::Value* base_index) {
  std::vector<llvm_ir::IrArray::Index> array_indices;

  LinearBaseAndThreadIdx base_and_threadidx =
      EmitLinearBaseAndThreadIdx(index_type, base_index);
  llvm::Value* linear_index_base = base_and_threadidx.linear_base;
  const int unroll_factor = launch_config_.unroll_factor;

  llvm::Value* linear_base = linear_index_base;

  for (int i = 0; i < unroll_factor; ++i) {
    std::vector<llvm::Value*> multidim(shape_.rank(), nullptr);
    if (i > 0) {
      llvm::Value* addend = llvm::ConstantInt::get(
          index_type, launch_dimensions_.total_nb_threads());
      linear_base =
          b_->CreateAdd(linear_base, addend, absl::StrCat("linear_index", i),
                        /*HasNUW=*/true, /*HasNSW=*/true);
    }
    auto dims = shape_.dimensions();
    llvm::Value* last_val = linear_base;
    for (int j = dims.size() - 1; j >= 0; j--) {
      multidim[j] =
          b_->CreateURem(last_val, llvm::ConstantInt::get(index_type, dims[j]));
      last_val =
          b_->CreateUDiv(last_val, llvm::ConstantInt::get(index_type, dims[j]));
    }
    array_indices.emplace_back(multidim, shape_, index_type);
  }

  // We don't need to do bounds checking because this method is only
  // triggered for cases where we have already verified the bounds.
  llvm::BasicBlock* current_block = b_->GetInsertBlock();
  llvm::BasicBlock* body_block =
      llvm_ir::CreateBasicBlock(nullptr, "fusion-body", b_);
  if (current_block->getTerminator() == nullptr) {
    exit_bb_ = llvm_ir::CreateBasicBlock(nullptr, "after-fusion-body", b_);
  } else {
    exit_bb_ = current_block->splitBasicBlock(b_->GetInsertPoint(),
                                              "after-fusion-body");
    current_block->getTerminator()->eraseFromParent();
  }
  b_->SetInsertPoint(current_block);
  b_->CreateBr(body_block);
  b_->SetInsertPoint(body_block);
  b_->CreateBr(exit_bb_);

  // Set IR builder insertion point to the body of the if structure.
  llvm_ir::SetToFirstInsertPoint(body_block, b_);
  return array_indices;
}

std::vector<llvm_ir::IrArray::Index>
ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock(absl::string_view loop_name,
                                                   llvm::Type* index_type,
                                                   llvm::Value* base_index) {
  // Emit the following code in LLVM IR:
  //   linear_index = blockIdx.x * blockDim.x * blockDim.y [+ threadIdx.y *
  //   blockDim.x] + threadIdx.x; if (linear_index < num_elements) {
  //     array_index = LinearIndexToMultidimensionalIndex(shape_, linear_index);
  //     ...
  //   }
  // The part between [] are added only if blockDim.y > 1.
  // blockIdx.y and gridDim.y are always 1.

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %ctaid.x <  %nctaid.x"
  //
  // %nctaid.x is currently specified as 2147483647.
  if (launch_dimensions_.thread_counts_per_block().y > 1) {
    // When blockDim.y > 1, then we are in the small row case. Each
    // blockDim.x do exatly to one row and blockDim.y map to some
    // consecutive row. This prevents too small block size that isn't
    // efficient.
    CHECK(launch_config_.row_vectorized);
    CHECK_EQ(shape_.dimensions().back(),
             launch_dimensions_.thread_counts_per_block().x *
                 launch_config_.unroll_factor);
  }
  CHECK_EQ(launch_dimensions_.thread_counts_per_block().z, 1);
  CHECK_EQ(launch_dimensions_.block_counts().y, 1);
  CHECK_EQ(launch_dimensions_.block_counts().z, 1);
  VLOG(3) << "EmitIndexAndSetExitBasicBlock unroll_factor "
          << launch_config_.unroll_factor;
  CHECK_NE(index_type, nullptr);

  if (launch_config_.logical_order) {
    return EmitLogicalIndexAndSetExitBasicBlock(loop_name, index_type,
                                                base_index);
  }

  std::vector<llvm_ir::IrArray::Index> array_indices;
  LinearBaseAndThreadIdx linear_base_and_thread_idx =
      EmitLinearBaseAndThreadIdx(index_type, base_index);

  llvm::Value* linear_index_base = linear_base_and_thread_idx.linear_base;
  llvm::Value* thread_id_x = linear_base_and_thread_idx.thread_idx;

  // When enable_row_index is true, it means the inner most dimensions
  // match the block sizes.  So we can generate a simpler indexing
  // for that dimensions.  This helps LLVM generate vectorized codes
  // in that cases.
  llvm::Value* row_index = nullptr;
  if (!launch_config_.row_vectorized) {
    array_indices.emplace_back(linear_index_base, shape_, b_);
  } else {
    // Simpler index for row computation.
    // This will allow LLVM to vectorize.
    row_index = b_->CreateMul(
        thread_id_x,
        llvm::ConstantInt::get(index_type, launch_config_.unroll_factor),
        "row_index", /*HasNUW=*/true, /*HasNSW=*/true);
    std::vector<llvm::Value*> multidim(shape_.rank(), nullptr);
    multidim.back() = row_index;
    array_indices.emplace_back(linear_index_base, multidim, shape_, b_);
  }

  for (int i = 1; i < launch_config_.unroll_factor; ++i) {
    llvm::Value* linear_index =
        b_->CreateAdd(linear_index_base, llvm::ConstantInt::get(index_type, i),
                      absl::StrCat("linear_index", i),
                      /*HasNUW=*/true, /*HasNSW=*/true);
    if (!launch_config_.row_vectorized) {
      array_indices.emplace_back(linear_index, shape_, b_);
    } else {
      std::vector<llvm::Value*> multidim(shape_.rank(), nullptr);
      multidim.back() = b_->CreateAdd(
          row_index, llvm::ConstantInt::get(index_type, i),
          absl::StrCat("row_index_plus", i), /*HasNUW=*/true, /*HasNSW=*/true);
      array_indices.emplace_back(linear_index, multidim, shape_, b_);
    }
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

Status ParallelLoopEmitter::EmitSerialLoop(absl::string_view loop_name,
                                           llvm::Type* index_type,
                                           llvm::Value* base_indvar) {
  for (const llvm_ir::IrArray::Index& array_index :
       EmitIndexAndSetExitBasicBlock(loop_name, index_type, base_indvar)) {
    TF_RETURN_IF_ERROR(body_emitter_(array_index));
  }
  return Status::OK();
}

Status ParallelLoopEmitter::EmitLoop(absl::string_view loop_name,
                                     llvm::Type* index_type) {
  if (index_type == nullptr) {
    index_type = b_->getInt64Ty();
  }
  int64_t total_threads = launch_dimensions_.launch_bound();
  int64_t num_elements = ShapeUtil::ElementsIn(shape_);
  // If all the elements are handled by the current threads, no need
  // to add a loop inside the kernel.
  if (total_threads * launch_config_.unroll_factor >= num_elements) {
    VLOG(1) << "No loops inside the kernel";
    TF_RETURN_IF_ERROR(EmitSerialLoop(loop_name, index_type));
  } else {
    KernelSupportLibrary ksl(b_, llvm_ir::UnrollMode::kDefaultUnroll);
    auto constant = [&](int64_t val) {
      return llvm::ConstantInt::get(index_type, val);
    };

    TF_RETURN_IF_ERROR(ksl.ForWithStatus(
        "loop", constant(0), constant(num_elements),
        constant(total_threads * launch_config_.unroll_factor),
        [&](llvm::Value* base_indvar) {
          return EmitSerialLoop(loop_name, index_type, base_indvar);
        }));
  }

  // Set the insertion point of b_ to the loop exit, so that
  // code emitted for later instructions will be correctly placed.
  if (exit_bb_ != nullptr) {
    b_->SetInsertPoint(exit_bb_);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
