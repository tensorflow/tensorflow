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

#include "tensorflow/compiler/xla/service/cpu/vector_support_library.h"

#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {
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
  CHECK(lhs->getType() == scalar_type() || lhs->getType() == vector_type());
  return MulInternal(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::MulInternal(llvm::Value* lhs,
                                               llvm::Value* rhs) {
  if (scalar_type_->isFloatingPointTy()) {
    return ir_builder()->CreateFMul(lhs, rhs, name());
  } else {
    return ir_builder()->CreateMul(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::Add(llvm::Value* lhs, llvm::Value* rhs) {
  CHECK(lhs->getType() == scalar_type() || lhs->getType() == vector_type());
  return AddInternal(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::AddInternal(llvm::Value* lhs,
                                               llvm::Value* rhs) {
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

llvm::Value* VectorSupportLibrary::AvxStyleHorizontalAdd(llvm::Value* lhs,
                                                         llvm::Value* rhs) {
  CHECK_EQ(lhs->getType(), vector_type());
  CHECK_EQ(rhs->getType(), vector_type());
  CHECK_EQ(vector_size() % 2, 0);

  llvm::SmallVector<llvm::Constant*, 32> mask_a, mask_b;

  // Adding the values shuffled using mask_a and mask_b gives us the
  // AVX-style horizontal add we want.  The masks work as documented
  // in https://llvm.org/docs/LangRef.html#shufflevector-instruction
  //
  // Here are the masks for vector_width() == 8:
  //
  //    index: |0 |1 |2 | 3 |4 |5 | 6 | 7
  //   --------+--+--+--+---+--+--+---+---
  //   mask_a: |0 |2 |8 |10 |4 |6 |12 |14
  //   mask_b: |1 |3 |9 |11 |5 |7 |13 |16
  //
  // So, as an example, the value at lane 3 of the result vector is
  // the result of adding lane 10 and lane 11 in the combined lhs++rhs
  // vector, which are the lanes 2 and 3 in the rhs vector.
  for (int i = 0; i < vector_size(); i += 2) {
    int increment = i < vector_size() / 2 ? 0 : (vector_size() / 2);
    mask_a.push_back(ir_builder()->getInt32(increment + i));
    mask_b.push_back(ir_builder()->getInt32(increment + i + 1));
  }
  for (int i = 0; i < vector_size(); i += 2) {
    int increment = i < vector_size() / 2 ? (vector_size() / 2) : vector_size();
    mask_a.push_back(ir_builder()->getInt32(increment + i));
    mask_b.push_back(ir_builder()->getInt32(increment + i + 1));
  }

  llvm::Value* shuffle_0 = ir_builder()->CreateShuffleVector(
      lhs, rhs, llvm::ConstantVector::get(mask_a));
  llvm::Value* shuffle_1 = ir_builder()->CreateShuffleVector(
      lhs, rhs, llvm::ConstantVector::get(mask_b));

  return Add(shuffle_0, shuffle_1);
}

llvm::Value* VectorSupportLibrary::ExtractLowHalf(llvm::Value* vector) {
  llvm::SmallVector<llvm::Constant*, 32> mask;
  for (int i = 0; i < vector_size() / 2; i++) {
    mask.push_back(ir_builder()->getInt32(i));
  }

  return ir_builder()->CreateShuffleVector(vector,
                                           llvm::UndefValue::get(vector_type()),
                                           llvm::ConstantVector::get(mask));
}

llvm::Value* VectorSupportLibrary::ExtractHighHalf(llvm::Value* vector) {
  llvm::SmallVector<llvm::Constant*, 32> mask;
  for (int i = 0; i < vector_size() / 2; i++) {
    mask.push_back(ir_builder()->getInt32(i + vector_size() / 2));
  }

  return ir_builder()->CreateShuffleVector(vector,
                                           llvm::UndefValue::get(vector_type()),
                                           llvm::ConstantVector::get(mask));
}

std::vector<llvm::Value*> VectorSupportLibrary::ComputeHorizontalSums(
    std::vector<llvm::Value*> vectors, llvm::Value* init_values) {
  const int x86_avx_vector_elements =
      TargetMachineFeatures::kX86AvxVectorByteSize / scalar_byte_size();
  if (vector_size() == x86_avx_vector_elements &&
      vectors.size() == x86_avx_vector_elements) {
    return ComputeAvxOptimizedHorizontalSums(std::move(vectors), init_values);
  }

  std::vector<llvm::Value*> result;
  std::transform(vectors.begin(), vectors.end(), std::back_inserter(result),
                 [this](llvm::Value* vector) { return AddReduce(vector); });
  if (init_values) {
    for (int64 i = 0, e = result.size(); i < e; i++) {
      result[i] = Add(result[i], ir_builder()->CreateExtractElement(
                                     init_values, ir_builder()->getInt32(i)));
    }
  }
  return result;
}

std::vector<llvm::Value*>
VectorSupportLibrary::ComputeAvxOptimizedHorizontalSums(
    std::vector<llvm::Value*> vectors, llvm::Value* init_values) {
  while (vectors.size() != 2) {
    std::vector<llvm::Value*> new_vectors;
    for (int i = 0; i < vectors.size(); i += 2) {
      new_vectors.push_back(AvxStyleHorizontalAdd(vectors[i], vectors[i + 1]));
    }

    vectors = std::move(new_vectors);
  }

  llvm::Value* low =
      AddInternal(ExtractLowHalf(vectors[0]), ExtractHighHalf(vectors[0]));
  if (init_values) {
    low = AddInternal(ExtractLowHalf(init_values), low);
  }
  llvm::Value* high =
      AddInternal(ExtractLowHalf(vectors[1]), ExtractHighHalf(vectors[1]));
  if (init_values) {
    high = AddInternal(ExtractHighHalf(init_values), high);
  }

  std::vector<llvm::Value*> results;
  for (int i = 0; i < 8; i++) {
    llvm::Value* scalar_result = ir_builder()->CreateExtractElement(
        i < 4 ? low : high, ir_builder()->getInt32(i % 4), name());
    results.push_back(scalar_result);
  }

  return results;
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

llvm::Value* LlvmVariable::Get() const {
  return ir_builder_->CreateLoad(alloca_);
}

void LlvmVariable::Set(llvm::Value* new_value) {
  ir_builder_->CreateStore(new_value, alloca_);
}
}  // namespace cpu
}  // namespace xla
