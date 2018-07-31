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

#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

LoopEmitter::LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                         llvm::IRBuilder<>* b)
    : body_emitter_(body_emitter), shape_(shape), b_(b) {}

LoopEmitter::LoopEmitter(const ElementGenerator& target_element_generator,
                         const IrArray& target_array, llvm::IRBuilder<>* b)
    : body_emitter_([=](const llvm_ir::IrArray::Index array_index) -> Status {
        // Convert target_element_generator to a BodyEmitter.
        TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                            target_element_generator(array_index));
        target_array.EmitWriteArrayElement(array_index, target_element, b);
        return Status::OK();
      }),
      shape_(target_array.GetShape()),
      b_(b) {}

static LoopEmitter::BodyEmitter MakeBodyEmitterForMultiOutputFusion(
    const ElementGenerator& target_element_generator,
    const std::vector<IrArray>& target_arrays, llvm::IRBuilder<>* b) {
  return [=](const llvm_ir::IrArray::Index array_index) {
    TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                        target_element_generator(array_index));
    CHECK(target_element->getType()->isStructTy())
        << "This BodyEmitter is for multi-output fusion, but target element "
           "generator does not produce values of struct type.";
    CHECK_EQ(target_element->getType()->getStructNumElements(),
             target_arrays.size());

    for (int64 i = 0; i < target_arrays.size(); ++i) {
      target_arrays[i].EmitWriteArrayElement(
          array_index, b->CreateExtractValue(target_element, i), b);
    }
    return Status::OK();
  };
}

LoopEmitter::LoopEmitter(const ElementGenerator& target_element_generator,
                         tensorflow::gtl::ArraySlice<IrArray> target_arrays,
                         llvm::IRBuilder<>* b)
    : body_emitter_(MakeBodyEmitterForMultiOutputFusion(
          target_element_generator,
          std::vector<IrArray>(target_arrays.begin(), target_arrays.end()), b)),
      shape_(target_arrays[0].GetShape()),
      b_(b) {
  // Sanity check: In multi-output fusion, all shapes produced must have the
  // same dimensions.
  for (const IrArray& array : target_arrays) {
    CHECK(ShapeUtil::SameDimensions(shape_, array.GetShape()))
        << ": '" << shape_.ShortDebugString() << "' does not match '"
        << array.GetShape().ShortDebugString() << "'";
  }
}

std::vector<IrArray::Index> LoopEmitter::EmitIndexAndSetExitBasicBlock(
    tensorflow::StringPiece loop_name, llvm::Type* index_type) {
  CHECK_NE(index_type, nullptr);
  if (ShapeUtil::IsScalar(shape_)) {
    // No loop needed, so set exit_bb_ to nullptr.
    exit_bb_ = nullptr;
    return {IrArray::Index(index_type)};
  }

  // Create loop nest with one for-loop for each dimension of the target shape.
  // Loops are added from outermost to innermost order with the ForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  ForLoopNest loop_nest(loop_name, b_);
  IrArray::Index array_index(index_type, shape_.dimensions_size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64 dimension = LayoutUtil::Major(shape_.layout(), i);
    std::unique_ptr<ForLoop> loop = loop_nest.AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape_.dimensions(dimension),
        /*suffix=*/tensorflow::strings::Printf("dim.%lld", dimension));
    array_index[dimension] = loop->GetIndVarValue();
  }

  // Set IR builder insertion point to the loop body basic block of the
  // innermost loop.
  llvm::BasicBlock* innermost_body_bb = loop_nest.GetInnerLoopBodyBasicBlock();
  b_->SetInsertPoint(innermost_body_bb,
                     innermost_body_bb->getFirstInsertionPt());

  // Set exit_bb_ to the exit block of the loop nest.
  exit_bb_ = loop_nest.GetOuterLoopExitBasicBlock();
  CHECK_NOTNULL(exit_bb_);

  return {array_index};
}

Status LoopEmitter::EmitLoop(tensorflow::StringPiece loop_name,
                             llvm::Type* index_type) {
  if (index_type == nullptr) {
    index_type = b_->getInt64Ty();
  }

  for (const IrArray::Index& array_index :
       EmitIndexAndSetExitBasicBlock(loop_name, index_type)) {
    TF_RETURN_IF_ERROR(body_emitter_(array_index));
  }

  // Set the insertion point of b_ to the loop exit, so that
  // code emitted for later instructions will be correctly placed.
  if (exit_bb_ != nullptr) {
    b_->SetInsertPoint(exit_bb_);
  }
  return Status::OK();
}

}  // namespace llvm_ir
}  // namespace xla
