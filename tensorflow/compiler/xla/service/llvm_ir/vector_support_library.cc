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

#include "tensorflow/compiler/xla/service/llvm_ir/vector_support_library.h"

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
VectorSupportLibrary::VectorSupportLibrary(PrimitiveType primitive_type,
                                           int64 vector_size,
                                           llvm::IRBuilder<>* ir_builder,
                                           std::string name)
    : vector_size_(vector_size),
      primitive_type_(primitive_type),
      ir_builder_(ir_builder),
      name_(std::move(name)) {
  scalar_type_ = llvm_ir::PrimitiveTypeToIrType(
      primitive_type, ir_builder_->GetInsertBlock()->getModule());
  scalar_pointer_type_ = llvm::PointerType::getUnqual(scalar_type_);
  vector_type_ = llvm::VectorType::get(scalar_type_, vector_size);
  vector_pointer_type_ = llvm::PointerType::getUnqual(vector_type_);
}

llvm::Value* VectorSupportLibrary::Mul(llvm::Value* lhs, llvm::Value* rhs) {
  if (scalar_type_->isFloatingPointTy()) {
    return ir_builder()->CreateFMul(lhs, rhs, name());
  } else {
    return ir_builder()->CreateMul(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::Add(llvm::Value* lhs, llvm::Value* rhs) {
  if (scalar_type_->isFloatingPointTy()) {
    return ir_builder()->CreateFAdd(lhs, rhs, name());
  } else {
    return ir_builder()->CreateAdd(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::ComputeOffsetPointer(
    llvm::Value* base_pointer, llvm::Value* offset_elements) {
  if (base_pointer->getType() != scalar_pointer_type()) {
    base_pointer = ir_builder()->CreateBitCast(base_pointer,
                                               scalar_pointer_type(), name());
  }
  return ir_builder()->CreateInBoundsGEP(base_pointer, {offset_elements},
                                         name());
}

llvm::Value* VectorSupportLibrary::LoadVector(llvm::Value* pointer) {
  if (pointer->getType() != vector_pointer_type()) {
    pointer =
        ir_builder()->CreateBitCast(pointer, vector_pointer_type(), name());
  }
  return ir_builder()->CreateAlignedLoad(
      pointer, ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_), name());
}

llvm::Value* VectorSupportLibrary::LoadScalar(llvm::Value* pointer) {
  if (pointer->getType() != scalar_pointer_type()) {
    pointer =
        ir_builder()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  return ir_builder()->CreateAlignedLoad(
      pointer, ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_), name());
}

void VectorSupportLibrary::StoreVector(llvm::Value* value,
                                       llvm::Value* pointer) {
  if (pointer->getType() != vector_pointer_type()) {
    pointer = ir_builder()->CreateBitCast(pointer, vector_pointer_type());
  }
  ir_builder()->CreateAlignedStore(
      value, pointer, ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_));
}

void VectorSupportLibrary::StoreScalar(llvm::Value* value,
                                       llvm::Value* pointer) {
  if (pointer->getType() != scalar_pointer_type()) {
    pointer =
        ir_builder()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  ir_builder()->CreateAlignedStore(
      value, pointer, ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_));
}

llvm::Value* VectorSupportLibrary::LoadBroadcast(llvm::Value* pointer) {
  if (pointer->getType() != scalar_pointer_type()) {
    pointer =
        ir_builder()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  return ir_builder()->CreateVectorSplat(
      vector_size(), ir_builder()->CreateLoad(pointer), name());
}

llvm::Value* VectorSupportLibrary::AddReduce(llvm::Value* vector) {
  llvm::SmallVector<llvm::Constant*, 32> mask(vector_size(), nullptr);
  for (unsigned i = vector_size(); i != 1; i >>= 1) {
    // On every iteration, we shuffle half of the remaining lanes to the top
    // half of shuffle, and add two old and the new vector.

    for (unsigned j = 0; j < vector_size(); ++j) {
      if (j < (i / 2)) {
        mask[j] = ir_builder()->getInt32(i / 2 + j);
      } else {
        mask[j] = llvm::UndefValue::get(ir_builder()->getInt32Ty());
      }
    }

    llvm::Value* half_remaining_lanes = ir_builder()->CreateShuffleVector(
        vector, llvm::UndefValue::get(vector_type()),
        llvm::ConstantVector::get(mask), "");
    vector = Add(vector, half_remaining_lanes);
  }

  return ir_builder()->CreateExtractElement(vector, ir_builder()->getInt32(0),
                                            name());
}

llvm::Value* VectorSupportLibrary::GetZeroVector() {
  return llvm::Constant::getNullValue(vector_type());
}

llvm::Value* VectorSupportLibrary::GetZeroScalar() {
  return llvm::Constant::getNullValue(scalar_type());
}

LlvmVariable::LlvmVariable(llvm::Type* type, llvm::IRBuilder<>* ir_builder)
    : ir_builder_(ir_builder) {
  alloca_ = llvm_ir::EmitAllocaAtFunctionEntry(type, "", ir_builder_);
}

llvm::Value* LlvmVariable::Get() { return ir_builder_->CreateLoad(alloca_); }

void LlvmVariable::Set(llvm::Value* new_value) {
  ir_builder_->CreateStore(new_value, alloca_);
}
}  // namespace xla
