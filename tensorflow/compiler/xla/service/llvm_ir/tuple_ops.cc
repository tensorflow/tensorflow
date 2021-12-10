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
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

static llvm::Module* getModuleFromBuilder(llvm::IRBuilder<>* b) {
  return b->GetInsertBlock()->getModule();
}

void EmitTupleSelect(const IrArray& select, const IrArray& pred,
                     llvm::Value* on_true, llvm::Value* on_false,
                     llvm::IRBuilder<>* b) {
  llvm::Module* module = getModuleFromBuilder(b);
  CHECK(ShapeUtil::IsScalar(pred.GetShape()));

  llvm::Type* pred_type = PrimitiveTypeToIrType(PRED, module);
  llvm::LoadInst* pred_value =
      b->CreateLoad(pred_type, pred.GetBasePointer(), "load_predicate_value");
  llvm::Value* pred_cond = b->CreateICmpNE(
      pred_value, llvm::ConstantInt::get(pred_type, 0), "boolean_predicate");

  VLOG(2) << "HandleSelect for tuple:";
  VLOG(2) << "  pred_value: " << DumpToString(*pred_value);
  VLOG(2) << "  pred_cond: " << DumpToString(*pred_cond);

  llvm::Value* src = b->CreateSelect(pred_cond, on_true, on_false);
  llvm::Value* dst = select.GetBasePointer();
  int64_t table_size = ShapeUtil::ByteSizeOfTupleIndexTable(
      select.GetShape(), module->getDataLayout().getPointerSize());
  b->CreateMemCpy(dst, /*DstAlign=*/llvm::Align(1), src,
                  /*SrcAlign=*/llvm::Align(1), b->getInt64(table_size));
}

void EmitTuple(const IrArray& tuple, absl::Span<llvm::Value* const> operands,
               llvm::IRBuilder<>* b) {
  llvm::Module* module = getModuleFromBuilder(b);
  for (size_t i = 0; i < operands.size(); ++i) {
    auto* cast =
        b->CreatePointerCast(operands[i], PrimitiveTypeToIrType(TUPLE, module));
    auto* store = b->CreateStore(
        cast, b->CreateInBoundsGEP(
                  tuple.GetBasePointer()->getType()->getPointerElementType(),
                  tuple.GetBasePointer(), {b->getInt64(0), b->getInt64(i)}));
    tuple.AnnotateLoadStoreInstructionWithMetadata(store);
  }
}

void EmitTuple(const IrArray& tuple, absl::Span<const IrArray> buffers,
               llvm::IRBuilder<>* b) {
  std::vector<llvm::Value*> buffer_ptrs;
  buffer_ptrs.reserve(buffers.size());
  absl::c_transform(
      buffers, std::back_inserter(buffer_ptrs),
      [](const llvm_ir::IrArray& buffer) { return buffer.GetBasePointer(); });
  llvm_ir::EmitTuple(tuple, buffer_ptrs, b);
}

std::vector<llvm::Value*> EmitTupleAllocasAtFunctionEntry(
    const Shape& tuple_shape, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();

  llvm::IRBuilder<>::InsertPointGuard guard(*b);
  llvm::Function* function = b->GetInsertBlock()->getParent();
  b->SetInsertPoint(&function->getEntryBlock(),
                    function->getEntryBlock().getFirstInsertionPt());
  CHECK(tuple_shape.IsTuple());
  int tuple_size = tuple_shape.tuple_shapes_size();

  std::vector<llvm::Value*> generated_allocas;
  for (int i = 0; i < tuple_size; i++) {
    const Shape& element_shape = tuple_shape.tuple_shapes(i);
    CHECK(ShapeUtil::IsScalar(element_shape));
    llvm::Type* type =
        llvm_ir::PrimitiveTypeToIrType(element_shape.element_type(), module);
    llvm::AllocaInst* alloca = b->CreateAlloca(
        type,
        /*ArraySize=*/nullptr, AsStringRef(absl::StrCat("tuple_element_", i)));
    generated_allocas.push_back(alloca);
  }

  return generated_allocas;
}

llvm::Value* EmitGetTupleElement(const Shape& target_shape, int64_t index,
                                 int alignment, llvm::Value* operand,
                                 llvm::IRBuilder<>* b) {
  llvm::Module* module = getModuleFromBuilder(b);
  llvm::Value* element_ptr =
      b->CreateInBoundsGEP(operand->getType()->getPointerElementType(), operand,
                           {b->getInt64(0), b->getInt64(index)});
  llvm::LoadInst* src_buffer = b->CreateLoad(
      element_ptr->getType()->getPointerElementType(), element_ptr);

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
