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

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"

#include <numeric>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

ForLoop::ForLoop(absl::string_view prefix, absl::string_view suffix,
                 llvm::Value* start_index, llvm::Value* end_index,
                 llvm::Value* step, UnrollMode unroll_mode,
                 bool prevent_vectorization)
    : prefix_(prefix),
      suffix_(suffix),
      start_index_(start_index),
      end_index_(end_index),
      step_(step),
      insert_before_bb_(nullptr),
      unroll_mode_(unroll_mode),
      prevent_vectorization_(prevent_vectorization) {}

/* static */ std::unique_ptr<ForLoop> ForLoop::EmitForLoop(
    absl::string_view prefix, llvm::Value* start_index, llvm::Value* end_index,
    llvm::Value* step, llvm::IRBuilder<>* b, UnrollMode unroll_mode,
    bool prevent_vectorization) {
  std::unique_ptr<ForLoop> loop(new ForLoop(prefix, /*suffix=*/"", start_index,
                                            end_index, step, unroll_mode,
                                            prevent_vectorization));
  loop->Emit(b);
  return loop;
}

void ForLoop::Emit(llvm::IRBuilder<>* b) {
  // The preheader block is the block the builder is currently emitting
  // code into.
  preheader_bb_ = b->GetInsertBlock();

  llvm::BasicBlock::iterator insert_point = b->GetInsertPoint();
  if (insert_point == preheader_bb_->end()) {
    // We're emitting the loop at the end of a basic block. Verify there is no
    // terminator (eg, branch) in the basic block.
    CHECK_EQ(nullptr, preheader_bb_->getTerminator());

    exit_bb_ = CreateLoopBB("loop_exit", b);
  } else {
    // We're emitting the loop into the middle of a basic block. splitBasicBlock
    // requires that this basic block be well-formed (have a terminator).
    CHECK_NE(nullptr, preheader_bb_->getTerminator());

    // Split the preheader to create an exit basic block. The exit basic block
    // will contain all instructions at or after insert_point.
    exit_bb_ = preheader_bb_->splitBasicBlock(
        insert_point, AsStringRef(GetQualifiedName("loop_exit")));

    // splitBasicBlock adds an unconditional branch between the split basic
    // blocks. Remove it. An unconditional branch will be added below from the
    // preheader to the header.
    preheader_bb_->getTerminator()->eraseFromParent();
  }
  insert_before_bb_ = exit_bb_;

  // Create remaining basic block which form the inside of the loop.
  header_bb_ = CreateLoopBB("loop_header", b);
  body_bb_ = CreateLoopBB("loop_body", b);

  // Function entry basic block.
  // Emit alloca for the induction variable. We do this at the entry to the
  // basic block to ensure the alloc only executes once per function (we could
  // be emitting a nested loop).
  llvm::Function* func = preheader_bb_->getParent();
  b->SetInsertPoint(&func->getEntryBlock(),
                    func->getEntryBlock().getFirstInsertionPt());
  llvm::Value* indvar_address =
      b->CreateAlloca(start_index_->getType(), nullptr,
                      AsStringRef(GetQualifiedName("invar_address")));

  // Preheader basic block.
  // Initialize induction variable starting index. Create branch to the header.
  b->SetInsertPoint(preheader_bb_);
  b->CreateStore(start_index_, indvar_address);
  // The preheader should not have a branch yet.
  CHECK_EQ(preheader_bb_->getTerminator(), nullptr);
  b->CreateBr(header_bb_);

  // Header basic block.
  // Emit the loop conditional branch. Load and compare indvar with ending
  // index and jump to loop exit if equal. Jump to body otherwise.
  b->SetInsertPoint(header_bb_);
  indvar_ =
      b->CreateLoad(indvar_address, AsStringRef(GetQualifiedName("indvar")));
  llvm::Value* exit_cond = b->CreateICmpUGE(indvar_, end_index_);
  b->CreateCondBr(/*Cond=*/exit_cond,
                  /*True=*/exit_bb_, /*False=*/body_bb_);

  // Body basic block.
  // Increment indvar, store indvar, and jump to header.
  b->SetInsertPoint(body_bb_);
  llvm::Value* step = step_;
  llvm::Value* indvar = indvar_;

  llvm::Value* indvar_inc = b->CreateAdd(indvar, step, "invar.inc",
                                         /*HasNUW=*/true, /*HasNSW=*/true);
  b->CreateStore(indvar_inc, indvar_address);
  llvm::BranchInst* back_branch = b->CreateBr(header_bb_);

  std::vector<llvm::Metadata*> loop_metadata = GetLoopMetadata(b);
  if (!loop_metadata.empty()) {
    llvm::LLVMContext* ctx = &start_index_->getContext();
    auto temp_node = llvm::MDNode::getTemporary(*ctx, llvm::None);
    loop_metadata.insert(loop_metadata.begin(), temp_node.get());
    auto loop_id = llvm::MDNode::get(*ctx, loop_metadata);
    loop_id->replaceOperandWith(0, loop_id);
    back_branch->setMetadata(llvm::LLVMContext::MD_loop, loop_id);
  }

  // Re-point the IR builder to the loop exit block.
  b->SetInsertPoint(exit_bb_);
}

std::vector<llvm::Metadata*> ForLoop::GetLoopMetadata(llvm::IRBuilder<>* b) {
  const char* const kLlvmLoopUnrollDisableMDName = "llvm.loop.unroll.disable";
  const char* const kLlvmLoopUnrollFullMDName = "llvm.loop.unroll.full";
  const char* const kLlvmLoopVectorizeMDName = "llvm.loop.vectorize.enable";
  llvm::LLVMContext* ctx = &start_index_->getContext();

  std::vector<llvm::Metadata*> result;
  if (unroll_mode_ == xla::llvm_ir::UnrollMode::kNoUnroll) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopUnrollDisableMDName)}));
  }

  if (prevent_vectorization_) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopVectorizeMDName),
               llvm::ConstantAsMetadata::get(b->getFalse())}));
  }

  if (unroll_mode_ == xla::llvm_ir::UnrollMode::kFullyUnroll) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopUnrollFullMDName)}));
  }
  return result;
}

string ForLoop::GetQualifiedName(absl::string_view name) {
  return llvm_ir::IrName(prefix_, llvm_ir::IrName(name, suffix_));
}

llvm::BasicBlock* ForLoop::CreateLoopBB(absl::string_view name,
                                        llvm::IRBuilder<>* b) {
  return CreateBasicBlock(insert_before_bb_, GetQualifiedName(name), b);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(absl::string_view suffix,
                                              llvm::Value* start_index,
                                              llvm::Value* end_index,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization) {
  return AddLoop(suffix, start_index, end_index, GetConstantWithIndexType(1),
                 unroll_mode, prevent_vectorization);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(
    absl::string_view suffix, llvm::Value* start_index, llvm::Value* end_index,
    llvm::Value* stride, UnrollMode unroll_mode, bool prevent_vectorization) {
  if (inner_loop_body_bb_ != nullptr) {
    // Create this loop inside the previous one.
    b_->SetInsertPoint(&*inner_loop_body_bb_->getFirstInsertionPt());
  }
  std::unique_ptr<ForLoop> loop(new ForLoop(
      /*prefix=*/name_, suffix, start_index, end_index, stride, unroll_mode,
      prevent_vectorization));
  loop->Emit(b_);

  if (outer_loop_preheader_bb_ == nullptr) {
    outer_loop_preheader_bb_ = loop->GetPreheaderBasicBlock();
  }

  if (outer_loop_exit_bb_ == nullptr) {
    outer_loop_exit_bb_ = loop->GetExitBasicBlock();
  }

  inner_loop_body_bb_ = loop->GetBodyBasicBlock();

  return loop;
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(int64 start_index,
                                              int64 end_index,
                                              absl::string_view suffix,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index), unroll_mode,
                 prevent_vectorization);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(int64 start_index,
                                              int64 end_index, int64 stride,
                                              absl::string_view suffix,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index),
                 GetConstantWithIndexType(stride), unroll_mode,
                 prevent_vectorization);
}

IrArray::Index ForLoopNest::AddLoopsForShape(const Shape& shape,
                                             absl::string_view suffix) {
  std::vector<int64> dimensions(ShapeUtil::Rank(shape));
  std::iota(dimensions.begin(), dimensions.end(), 0);
  return AddLoopsForShapeOnDimensions(shape, dimensions, suffix);
}

IrArray::Index ForLoopNest::AddLoopsForShapeOnDimensions(
    const Shape& shape, absl::Span<const int64> dimensions,
    absl::string_view suffix) {
  llvm_ir::IrArray::Index index(index_type_, shape.dimensions_size());
  for (int64 dimension : dimensions) {
    std::unique_ptr<llvm_ir::ForLoop> loop = AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape.dimensions(dimension),
        /*suffix=*/
        llvm_ir::IrName(suffix, absl::StrCat(dimension)));
    index[dimension] = loop->GetIndVarValue();
  }
  return index;
}

IrArray::Index ForLoopNest::EmitOperandArrayLoopNest(
    const llvm_ir::IrArray& operand_array, int64 dimension_to_skip,
    absl::string_view name_suffix) {
  // Prepares the dimension list we will use to emit the loop nest. Outermost
  // loops are added first. Add loops in major-to-minor order, and skip the
  // 'dimension_to_skip' dimension.
  std::vector<int64> dimensions;
  const Shape& shape = operand_array.GetShape();
  for (int64 dimension : LayoutUtil::MinorToMajor(shape)) {
    if (dimension != dimension_to_skip) {
      dimensions.push_back(dimension);
    }
  }

  // Create loop nest with one for-loop for each dimension of the
  // output.
  llvm_ir::IrArray::Index index =
      AddLoopsForShapeOnDimensions(shape, dimensions, name_suffix);
  // Verify every dimension except the 'dimension_to_skip' dimension was set in
  // the index.
  for (size_t dimension = 0; dimension < index.size(); ++dimension) {
    if (dimension == dimension_to_skip) {
      DCHECK_EQ(nullptr, index[dimension]);
    } else {
      DCHECK_NE(nullptr, index[dimension]);
    }
  }
  return index;
}

}  // namespace llvm_ir
}  // namespace xla
