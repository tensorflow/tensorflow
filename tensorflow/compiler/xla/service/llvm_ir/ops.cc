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

#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

void EmitTupleSelect(IrArray select, IrArray pred, llvm::Value* on_true,
                     llvm::Value* on_false, llvm::IRBuilder<>* ir_builder) {
  CHECK(ShapeUtil::IsScalar(pred.GetShape()));

  llvm::LoadInst* pred_value =
      ir_builder->CreateLoad(pred.GetBasePointer(), "load_predicate_value");
  llvm::Value* pred_cond = ir_builder->CreateICmpNE(
      pred_value,
      llvm::ConstantInt::get(PrimitiveTypeToIrType(PRED, ir_builder), 0),
      "boolean_predicate");

  VLOG(2) << "HandleSelect for tuple:";
  VLOG(2) << "  pred_value: " << DumpToString(*pred_value);
  VLOG(2) << "  pred_cond: " << DumpToString(*pred_cond);

  for (int i = 0; i < ShapeUtil::TupleElementCount(select.GetShape()); ++i) {
    std::vector<llvm::Value*> element_index = {ir_builder->getInt64(0),
                                               ir_builder->getInt64(i)};
    llvm::Value* on_true_element_address =
        ir_builder->CreateInBoundsGEP(on_true, element_index);
    llvm::Value* on_true_element = ir_builder->CreateLoad(
        on_true_element_address,
        tensorflow::strings::Printf("on_true_element_%d", i).c_str());
    llvm::Value* on_false_element_address =
        ir_builder->CreateInBoundsGEP(on_false, element_index);
    llvm::Value* on_false_element = ir_builder->CreateLoad(
        on_false_element_address,
        tensorflow::strings::Printf("on_false_element_%d", i).c_str());

    llvm::Value* output_element_address =
        ir_builder->CreateInBoundsGEP(select.GetBasePointer(), element_index);
    ir_builder->CreateStore(
        ir_builder->CreateSelect(
            pred_cond, on_true_element, on_false_element,
            tensorflow::strings::Printf("select_output_element_%d", i).c_str()),
        output_element_address);
  }
}

void EmitTuple(IrArray tuple,
               tensorflow::gtl::ArraySlice<llvm::Value*> operands,
               llvm::IRBuilder<>* ir_builder) {
  for (size_t i = 0; i < operands.size(); ++i) {
    ir_builder->CreateStore(
        ir_builder->CreatePointerCast(operands[i],
                                      PrimitiveTypeToIrType(TUPLE, ir_builder)),
        ir_builder->CreateInBoundsGEP(
            tuple.GetBasePointer(),
            {ir_builder->getInt64(0), ir_builder->getInt64(i)}));
  }
}

llvm::Value* EmitGetTupleElement(const Shape& target_shape, int64 index,
                                 int alignment, llvm::Value* operand,
                                 llvm::IRBuilder<>* ir_builder) {
  llvm::Value* element_ptr = ir_builder->CreateInBoundsGEP(
      operand, {ir_builder->getInt64(0), ir_builder->getInt64(index)});
  llvm::LoadInst* src_buffer = ir_builder->CreateLoad(element_ptr);
  SetTbaaForInstruction(src_buffer, target_shape, /*is_pointer_to=*/true);
  SetAlignmentMetadataForLoad(src_buffer, alignment);
  llvm::Type* element_type = ShapeToIrType(target_shape, ir_builder);
  llvm::Value* ret_val =
      ir_builder->CreateBitCast(src_buffer, element_type->getPointerTo());
  return ret_val;
}

}  // namespace llvm_ir
}  // namespace xla
