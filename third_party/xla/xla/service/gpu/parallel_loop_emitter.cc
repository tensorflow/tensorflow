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

#include "xla/service/gpu/parallel_loop_emitter.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"

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
                            static_cast<llvm::Instruction*>(block_id),
                            b_->GetInsertBlock()->getModule());
  block_id = b_->CreateZExtOrTrunc(block_id, index_type, "block_id");

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %tid.x <  %ntid.x"
  llvm::Value* thread_id_x =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.thread_counts_per_block().x,
                            static_cast<llvm::Instruction*>(thread_id_x),
                            b_->GetInsertBlock()->getModule());
  thread_id_x = b_->CreateZExtOrTrunc(thread_id_x, index_type, "thread_id_x");

  llvm::Value* linear_index_base =
      b_->CreateMul(block_id,
                    llvm::ConstantInt::get(
                        index_type, launch_dimensions_.num_threads_per_block()),
                    "",
                    /*HasNUW=*/true, /*HasNSW=*/true);

  if (launch_dimensions_.thread_counts_per_block().y > 1) {
    llvm::Value* thread_id_y =
        EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdy, {}, {}, b_);
    llvm_ir::AddRangeMetadata(0, launch_dimensions_.thread_counts_per_block().y,
                              static_cast<llvm::Instruction*>(thread_id_y),
                              b_->GetInsertBlock()->getModule());
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
  linear_index_base =
      b_->CreateAdd(linear_index_base, thread_id_x, "linear_index",
                    /*HasNUW=*/true, /*HasNSW=*/true);

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
                                 launch_dimensions_.num_threads_per_block() *
                                     launch_dimensions_.block_counts().x),
          "linear_index_in_range")},
      {}, b_);

  if (launch_config_.unroll_factor > 1) {
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

  std::vector<llvm_ir::IrArray::Index> array_indices;
  LinearBaseAndThreadIdx linear_base_and_thread_idx =
      EmitLinearBaseAndThreadIdx(index_type, base_index);

  llvm::Value* linear_index_base = linear_base_and_thread_idx.linear_base;

  llvm::Value* row_index =
      launch_config_.row_vectorized
          ? b_->CreateMul(linear_base_and_thread_idx.thread_idx,
                          llvm::ConstantInt::get(index_type,
                                                 launch_config_.unroll_factor),
                          "row_index", /*HasNUW=*/true, /*HasNSW=*/true)
          : nullptr;

  std::vector<llvm::Value*> multidim(shape_.rank(), nullptr);
  for (int i = 0; i < launch_config_.unroll_factor; ++i) {
    // The add operation is needed even if the offset is 0, since when the
    // kernel is unrolled, the following GEP instruction shares the same pointer
    // and sequential indices with others, allowing the default SLP pass to
    // optimize them into vectorized load/store operations.
    llvm::Value* linear_index =
        b_->CreateAdd(linear_index_base, llvm::ConstantInt::get(index_type, i),
                      absl::StrCat("linear_index", i),
                      /*HasNUW=*/true, /*HasNSW=*/true);
    if (launch_config_.row_vectorized) {
      // This lets us avoid emitting the division for the last dimension of the
      // index. The check for i > 0 is here for historical reasons, it might not
      // do anything.
      multidim.back() =
          i == 0 ? row_index
                 : b_->CreateAdd(
                       row_index, llvm::ConstantInt::get(index_type, i),
                       absl::StrCat("row_index_plus", i), /*HasNUW=*/true,
                       /*HasNSW=*/true);
    }
    array_indices.emplace_back(linear_index, multidim, shape_, b_);
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

absl::Status ParallelLoopEmitter::EmitSerialLoop(absl::string_view loop_name,
                                                 llvm::Type* index_type,
                                                 llvm::Value* base_indvar) {
  int64_t num_elements = ShapeUtil::ElementsIn(shape_);
  bool check_bounds = num_elements % launch_config_.unroll_factor > 0;
  for (const llvm_ir::IrArray::Index& array_index :
       EmitIndexAndSetExitBasicBlock(loop_name, index_type, base_indvar)) {
    if (!check_bounds) {
      TF_RETURN_IF_ERROR(body_emitter_(array_index));
    } else {
      // If the unroll_factor does not divide the number of elements, we must
      // check that the index is in bounds, since the last iteration of the last
      // thread might not have unroll_factor elements to write to. Normally
      // the caller of ParallelLoopEmitter ensures unroll_factor is always set
      // such that it divides num_elements, but for int4 arrays, the caller
      // always sets unroll_factor to a multiple of 2 to prevent different
      // threads from writing to adjacent elements occupying the same byte.
      CHECK(primitive_util::Is4BitType(shape_.element_type()));
      llvm_ir::LlvmIfData if_in_bounds = llvm_ir::EmitIfThenElse(
          b_->CreateICmpULT(array_index.linear(),
                            llvm::ConstantInt::get(index_type, num_elements)),
          llvm_ir::IrName(loop_name, "unrolled_in_bounds"), b_, false);
      llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, b_);
      TF_RETURN_IF_ERROR(body_emitter_(array_index));
      llvm_ir::SetToFirstInsertPoint(if_in_bounds.after_block, b_);
    }
  }
  return absl::OkStatus();
}

absl::Status ParallelLoopEmitter::EmitLoop(absl::string_view loop_name,
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
  CHECK(exit_bb_->getTerminator());
  b_->SetInsertPoint(exit_bb_->getTerminator());
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
