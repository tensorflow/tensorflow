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

#include "external/llvm/include/llvm/IR/Constants.h"
#include "external/llvm/include/llvm/IR/Function.h"
#include "external/llvm/include/llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

ForLoop::ForLoop(tensorflow::StringPiece suffix, llvm::Value* start_index,
                 llvm::Value* end_index, llvm::Value* step)
    : suffix_(suffix.ToString()),
      start_index_(start_index),
      end_index_(end_index),
      step_(step),
      insert_before_bb_(nullptr) {}

/* static */ std::unique_ptr<ForLoop> ForLoop::EmitForLoop(
    tensorflow::StringPiece suffix, llvm::Value* start_index,
    llvm::Value* end_index, llvm::Value* step, llvm::IRBuilder<>* ir_builder) {
  std::unique_ptr<ForLoop> loop(
      new ForLoop(suffix, start_index, end_index, step));
  loop->Emit(ir_builder);
  return loop;
}

void ForLoop::Emit(llvm::IRBuilder<>* ir_builder) {
  // The preheader block is the block the builder is currently emitting
  // code into.
  preheader_bb_ = ir_builder->GetInsertBlock();

  llvm::BasicBlock::iterator insert_point = ir_builder->GetInsertPoint();
  if (insert_point == preheader_bb_->end()) {
    // We're emitting the loop at the end of a basic block. Verify there is no
    // terminator (eg, branch) in the basic block.
    CHECK_EQ(nullptr, preheader_bb_->getTerminator());

    exit_bb_ = CreateBasicBlockWithSuffix("loop_exit", ir_builder);
  } else {
    // We're emitting the loop into the middle of a basic block. splitBasicBlock
    // requires that this basic block be well-formed (have a terminator).
    CHECK_NE(nullptr, preheader_bb_->getTerminator());

    // Split the preheader to create an exit basic block. The exit basic block
    // will contain all instructions at or after insert_point.
    exit_bb_ = preheader_bb_->splitBasicBlock(
        insert_point, GetNameWithSuffix("loop_exit").c_str());

    // splitBasicBlock adds an unconditional branch between the split basic
    // blocks. Remove it. An unconditional branch will be added below from the
    // preheader to the header.
    preheader_bb_->getTerminator()->eraseFromParent();
  }
  insert_before_bb_ = exit_bb_;

  // Create remaining basic block which form the inside of the loop.
  header_bb_ = CreateBasicBlockWithSuffix("loop_header", ir_builder);
  body_bb_ = CreateBasicBlockWithSuffix("loop_body", ir_builder);

  // Function entry basic block.
  // Emit alloca for the induction variable. We do this at the entry to the
  // basic block to ensure the alloc only executes once per function (we could
  // be emitting a nested loop).
  llvm::Function* func = preheader_bb_->getParent();
  ir_builder->SetInsertPoint(&func->getEntryBlock(),
                             func->getEntryBlock().getFirstInsertionPt());
  llvm::Value* indvar_address =
      ir_builder->CreateAlloca(ir_builder->getInt64Ty(), nullptr,
                               GetNameWithSuffix("invar_address").c_str());

  // Preheader basic block.
  // Initialize induction variable starting index. Create branch to the header.
  ir_builder->SetInsertPoint(preheader_bb_);
  ir_builder->CreateStore(start_index_, indvar_address);
  // The preheader should not have a branch yet.
  CHECK_EQ(preheader_bb_->getTerminator(), nullptr);
  ir_builder->CreateBr(header_bb_);

  // Header basic block.
  // Emit the loop conditional branch. Load and compare indvar with ending
  // index and jump to loop exit if equal. Jump to body otherwise.
  ir_builder->SetInsertPoint(header_bb_);
  indvar_ = ir_builder->CreateLoad(indvar_address,
                                   GetNameWithSuffix("indvar").c_str());
  llvm::Value* exit_cond = ir_builder->CreateICmpUGE(indvar_, end_index_);
  ir_builder->CreateCondBr(/*Cond=*/exit_cond,
                           /*True=*/exit_bb_, /*False=*/body_bb_);

  // Body basic block.
  // Increment indvar, store indvar, and jump to header.
  ir_builder->SetInsertPoint(body_bb_);
  llvm::Value* step = step_;
  llvm::Value* indvar = indvar_;

  llvm::Value* indvar_inc =
      ir_builder->CreateAdd(indvar, step, "invar.inc",
                            /*HasNUW=*/true, /*HasNSW=*/true);
  ir_builder->CreateStore(indvar_inc, indvar_address);
  ir_builder->CreateBr(header_bb_);

  // Re-point the IR builder to the loop exit block.
  ir_builder->SetInsertPoint(exit_bb_);
}

string ForLoop::GetNameWithSuffix(tensorflow::StringPiece name) {
  if (suffix_.empty()) {
    return name.ToString();
  } else {
    return tensorflow::strings::StrCat(name, ".", suffix_);
  }
}

llvm::BasicBlock* ForLoop::CreateBasicBlockWithSuffix(
    tensorflow::StringPiece name, llvm::IRBuilder<>* ir_builder) {
  return CreateBasicBlock(insert_before_bb_, GetNameWithSuffix(name),
                          ir_builder);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(tensorflow::StringPiece suffix,
                                              llvm::Value* start_index,
                                              llvm::Value* end_index) {
  if (inner_loop_body_bb_ != nullptr) {
    // Create this loop inside the previous one.
    ir_builder_->SetInsertPoint(&*inner_loop_body_bb_->getFirstInsertionPt());
  }
  std::unique_ptr<ForLoop> loop = ForLoop::EmitForLoop(
      suffix, start_index, end_index, ir_builder_->getInt64(1), ir_builder_);

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
                                              tensorflow::StringPiece suffix) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, ir_builder_->getInt64(start_index),
                 ir_builder_->getInt64(end_index));
}

IrArray::Index ForLoopNest::AddLoopsForShape(const Shape& shape,
                                             tensorflow::StringPiece suffix) {
  std::vector<int64> dimensions(ShapeUtil::Rank(shape));
  std::iota(dimensions.begin(), dimensions.end(), 0);
  return AddLoopsForShapeOnDimensions(shape, dimensions, suffix);
}

IrArray::Index ForLoopNest::AddLoopsForShapeOnDimensions(
    const Shape& shape, tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::StringPiece suffix) {
  llvm_ir::IrArray::Index index(shape.dimensions_size(), nullptr);
  for (int64 dimension : dimensions) {
    std::unique_ptr<llvm_ir::ForLoop> loop = AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape.dimensions(dimension),
        /*suffix=*/tensorflow::strings::Printf(
            "%s%lld", suffix.ToString().c_str(), dimension));
    index[dimension] = loop->GetIndVarValue();
  }
  return index;
}

}  // namespace llvm_ir
}  // namespace xla
