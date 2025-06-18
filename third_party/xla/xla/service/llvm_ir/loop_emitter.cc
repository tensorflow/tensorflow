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

#include "xla/service/llvm_ir/loop_emitter.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/layout_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace llvm_ir {

LoopEmitter::LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                         llvm::IRBuilderBase* b)
    : body_emitter_(body_emitter), shape_(shape), b_(b) {}

LoopEmitter::LoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                         std::vector<llvm::Value*> dynamic_dims,
                         llvm::IRBuilderBase* b)
    : LoopEmitter::LoopEmitter(body_emitter, shape, b) {
  CHECK_EQ(dynamic_dims.size(), shape_.dimensions().size());
  dynamic_dims_ = std::move(dynamic_dims);
}

LoopEmitter::LoopEmitter(const ElementGenerator& target_element_generator,
                         const IrArray& target_array, llvm::IRBuilderBase* b)
    : body_emitter_(MakeBodyEmitter(target_element_generator, {target_array}, b,
                                    /*is_tuple=*/false)),
      shape_(target_array.GetShape()),
      b_(b) {}

LoopEmitter::LoopEmitter(const ElementGenerator& target_element_generator,
                         absl::Span<const IrArray> target_arrays,
                         llvm::IRBuilderBase* b)
    : body_emitter_(MakeBodyEmitter(target_element_generator, target_arrays, b,
                                    /*is_tuple=*/true)),
      shape_(target_arrays[0].GetShape()),
      b_(b) {
  // Sanity check: In multi-output fusion, all shapes produced must have the
  // same dimensions.
  for (const IrArray& array : target_arrays) {
    CHECK(ShapeUtil::SameDimensions(shape_, array.GetShape()))
        << ": '" << shape_.ToString() << "' does not match '"
        << array.GetShape().ToString() << "'";
  }
}

BodyEmitter MakeBodyEmitter(const ElementGenerator& target_element_generator,
                            absl::Span<IrArray const> target_arrays,
                            llvm::IRBuilderBase* b, bool is_tuple) {
  std::vector<IrArray> target_arrays_vec(target_arrays.begin(),
                                         target_arrays.end());
  if (!is_tuple) {
    CHECK_EQ(target_arrays.size(), 1);
    return [=](const llvm_ir::IrArray::Index array_index) -> absl::Status {
      // Convert target_element_generator to a BodyEmitter.
      TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                          target_element_generator(array_index));
      target_arrays_vec[0].EmitWriteArrayElement(array_index, target_element,
                                                 b);
      return absl::OkStatus();
    };
  }

  return [=](const llvm_ir::IrArray::Index array_index) {
    TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                        target_element_generator(array_index));
    CHECK(target_element->getType()->isStructTy())
        << "This BodyEmitter is for multi-output, but target element "
           "generator does not produce values of struct type.";
    CHECK_EQ(target_element->getType()->getStructNumElements(),
             target_arrays_vec.size());

    for (int64_t i = 0; i < target_arrays_vec.size(); ++i) {
      IrArray::Index used_index = array_index;
      if (i > 0 && !ShapeUtil::EqualIgnoringElementType(
                       target_arrays_vec[i].GetShape(),
                       target_arrays_vec[0].GetShape())) {
        used_index =
            used_index.SourceIndexOfBitcast(target_arrays_vec[0].GetShape(),
                                            target_arrays_vec[i].GetShape(), b);
      }
      target_arrays_vec[i].EmitWriteArrayElement(
          used_index, b->CreateExtractValue(target_element, i), b);
    }
    return absl::OkStatus();
  };
}

IrArray::Index LoopEmitter::EmitStaticIndex(ForLoopNest* loop_nest,
                                            llvm::Type* index_type) {
  // Create loop nest with one for-loop for each dimension of the target shape.
  // Loops are added from outermost to innermost order with the ForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  std::vector<llvm::Value*> array_multi_index(shape_.dimensions().size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    // Only unroll the most minor dimension, this seems to give us good runtime
    // performance with a large improvement in compile time.
    auto unroll_mode = (i == shape_.dimensions().size() - 1)
                           ? llvm_ir::UnrollMode::kDefaultUnroll
                           : llvm_ir::UnrollMode::kNoUnroll;
    std::unique_ptr<ForLoop> loop = loop_nest->AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape_.dimensions(dimension),
        /*suffix=*/absl::StrFormat("dim.%d", dimension), unroll_mode);
    array_multi_index[dimension] = loop->GetIndVarValue();
  }
  return IrArray::Index(array_multi_index, shape_, index_type);
}

IrArray::Index LoopEmitter::EmitDynamicIndex(ForLoopNest* loop_nest,
                                             llvm::Type* index_type) {
  CHECK_EQ(shape_.is_dynamic(), true);
  // Create loop nest with one for-loop for each dynamic dimensions.
  // Loops are added from outermost to innermost order with the ForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  std::vector<llvm::Value*> array_multi_index(shape_.dimensions().size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    // Only unroll the most minor dimension, this seems to give us good runtime
    // performance with a large improvement in compile time.
    auto unroll_mode = (i == shape_.dimensions().size() - 1)
                           ? llvm_ir::UnrollMode::kDefaultUnroll
                           : llvm_ir::UnrollMode::kNoUnroll;
    std::unique_ptr<ForLoop> loop = loop_nest->AddLoop(
        /*suffix=*/absl::StrFormat("dim.%d", dimension),
        /*start_index=*/llvm::ConstantInt::get(index_type, 0),
        /*end_index=*/dynamic_dims_[dimension], unroll_mode);
    array_multi_index[dimension] = loop->GetIndVarValue();
  }
  return IrArray::Index(array_multi_index, shape_, index_type);
}

std::vector<IrArray::Index> LoopEmitter::EmitIndexAndSetExitBasicBlock(
    absl::string_view loop_name, llvm::Type* index_type,
    llvm::Value* base_index) {
  CHECK_NE(index_type, nullptr);
  CHECK_EQ(base_index, nullptr)
      << "XLA CPU implementation of"
      << " LoopEmitter::EmitIndexAndSetExitBasicBlock doesn't support"
      << " base_index, but it was requested.";

  if (ShapeUtil::IsScalar(shape_)) {
    // No loop needed, so set exit_bb_ to nullptr.
    exit_bb_ = nullptr;
    return {IrArray::Index(index_type)};
  }

  ForLoopNest loop_nest(loop_name, b_);

  IrArray::Index array_index = dynamic_dims_.empty()
                                   ? EmitStaticIndex(&loop_nest, index_type)
                                   : EmitDynamicIndex(&loop_nest, index_type);

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

absl::Status LoopEmitter::EmitLoop(absl::string_view loop_name,
                                   llvm::Type* index_type) {
  if (index_type == nullptr) {
    index_type = b_->getInt64Ty();
  }

  for (const IrArray::Index& array_index :
       EmitIndexAndSetExitBasicBlock(loop_name, index_type,
                                     /*base_index*/ nullptr)) {
    TF_RETURN_IF_ERROR(body_emitter_(array_index));
  }

  // Set the insertion point of b_ to the loop exit, so that
  // code emitted for later instructions will be correctly placed.
  if (exit_bb_ != nullptr) {
    b_->SetInsertPoint(exit_bb_);
  }
  return absl::OkStatus();
}

}  // namespace llvm_ir
}  // namespace xla
