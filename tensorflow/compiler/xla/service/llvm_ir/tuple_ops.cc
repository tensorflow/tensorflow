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

#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"

#include <stddef.h>
#include <string>
#include <vector>

#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

void EmitTupleSelect(const IrArray& select, const IrArray& pred,
                     llvm::Value* on_true, llvm::Value* on_false,
                     llvm::IRBuilder<>* b, llvm::Module* module) {
  CHECK(ShapeUtil::IsScalar(pred.GetShape()));

  llvm::LoadInst* pred_value =
      b->CreateLoad(pred.GetBasePointer(), "load_predicate_value");
  llvm::Value* pred_cond = b->CreateICmpNE(
      pred_value,
      llvm::ConstantInt::get(PrimitiveTypeToIrType(PRED, module), 0),
      "boolean_predicate");

  VLOG(2) << "HandleSelect for tuple:";
  VLOG(2) << "  pred_value: " << DumpToString(*pred_value);
  VLOG(2) << "  pred_cond: " << DumpToString(*pred_cond);

  for (int i = 0; i < ShapeUtil::TupleElementCount(select.GetShape()); ++i) {
    llvm::Value* const element_index[] = {b->getInt64(0), b->getInt64(i)};
    llvm::Value* on_true_element_address =
        b->CreateInBoundsGEP(on_true, element_index);
    llvm::Value* on_true_element = b->CreateLoad(
        on_true_element_address, "on_true_element_" + llvm::Twine(i));
    llvm::Value* on_false_element_address =
        b->CreateInBoundsGEP(on_false, element_index);
    llvm::Value* on_false_element = b->CreateLoad(
        on_false_element_address, "on_false_element_" + llvm::Twine(i));

    llvm::Value* output_element_address =
        b->CreateInBoundsGEP(select.GetBasePointer(), element_index);
    b->CreateStore(b->CreateSelect(pred_cond, on_true_element, on_false_element,
                                   "select_output_element_" + llvm::Twine(i)),
                   output_element_address);
  }
}

void EmitTuple(const IrArray& tuple, absl::Span<llvm::Value* const> operands,
               llvm::IRBuilder<>* b, llvm::Module* module) {
  for (size_t i = 0; i < operands.size(); ++i) {
    auto* store = b->CreateStore(
        b->CreatePointerCast(operands[i], PrimitiveTypeToIrType(TUPLE, module)),
        b->CreateInBoundsGEP(tuple.GetBasePointer(),
                             {b->getInt64(0), b->getInt64(i)}));
    tuple.AnnotateLoadStoreInstructionWithMetadata(store);
  }
}

void EmitTuple(const IrArray& tuple, absl::Span<const IrArray> buffers,
               llvm::IRBuilder<>* b, llvm::Module* module) {
  std::vector<llvm::Value*> buffer_ptrs;
  buffer_ptrs.reserve(buffers.size());
  absl::c_transform(
      buffers, std::back_inserter(buffer_ptrs),
      [](const llvm_ir::IrArray& buffer) { return buffer.GetBasePointer(); });
  llvm_ir::EmitTuple(tuple, buffer_ptrs, b, module);
}

llvm::Value* EmitGetTupleElement(const Shape& target_shape, int64 index,
                                 int alignment, llvm::Value* operand,
                                 llvm::IRBuilder<>* b, llvm::Module* module) {
  llvm::Value* element_ptr =
      b->CreateInBoundsGEP(operand, {b->getInt64(0), b->getInt64(index)});
  llvm::LoadInst* src_buffer = b->CreateLoad(element_ptr);

  // Mark the loaded pointer as dereferenceable if we know its shape.
  if (!target_shape.IsOpaque()) {
    SetDereferenceableMetadataForLoad(
        src_buffer,
        ByteSizeOf(target_shape, src_buffer->getModule()->getDataLayout()));
  }
  SetAlignmentMetadataForLoad(src_buffer, alignment);

  llvm::Type* element_type = ShapeToIrType(target_shape, module);
  llvm::Value* ret_val =
      b->CreateBitCast(src_buffer, element_type->getPointerTo());
  return ret_val;
}

}  // namespace llvm_ir
}  // namespace xla
