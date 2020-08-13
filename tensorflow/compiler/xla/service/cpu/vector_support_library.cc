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

#include "absl/algorithm/container.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {
VectorSupportLibrary::VectorSupportLibrary(PrimitiveType primitive_type,
                                           int64 vector_size,
                                           llvm::IRBuilder<>* b,
                                           std::string name)
    : vector_size_(vector_size),
      primitive_type_(primitive_type),
      b_(b),
      name_(std::move(name)) {
  scalar_type_ = llvm_ir::PrimitiveTypeToIrType(
      primitive_type, b_->GetInsertBlock()->getModule());
  scalar_pointer_type_ = llvm::PointerType::getUnqual(scalar_type_);
  vector_type_ = llvm::VectorType::get(scalar_type_, vector_size);
  vector_pointer_type_ = llvm::PointerType::getUnqual(vector_type_);
}

static string TypeToString(llvm::Type* type) {
  std::string o;
  llvm::raw_string_ostream ostream(o);
  type->print(ostream);
  return ostream.str();
}

void VectorSupportLibrary::AssertCorrectTypes(
    std::initializer_list<llvm::Value*> values) {
  for (llvm::Value* v : values) {
    llvm::Type* type = v->getType();
    if (type != scalar_type() && type != vector_type()) {
      LOG(FATAL) << "Expected either " << TypeToString(scalar_type()) << " or "
                 << TypeToString(vector_type()) << " but got "
                 << TypeToString(type);
    }
  }
}

llvm::Value* VectorSupportLibrary::Mul(llvm::Value* lhs, llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  return MulInternal(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::MulInternal(llvm::Value* lhs,
                                               llvm::Value* rhs) {
  if (scalar_type_->isFloatingPointTy()) {
    return b()->CreateFMul(lhs, rhs, name());
  } else {
    return b()->CreateMul(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::Add(llvm::Value* lhs, llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  return AddInternal(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::Sub(llvm::Value* lhs, llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  return b()->CreateFSub(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::Max(llvm::Value* lhs, llvm::Value* rhs,
                                       bool enable_fast_min_max) {
  AssertCorrectTypes({lhs, rhs});
  if (scalar_type_->isFloatingPointTy()) {
    return llvm_ir::EmitFloatMax(lhs, rhs, b_, enable_fast_min_max);
  } else {
    LOG(FATAL) << "Max for integers is unimplemented";
  }
}

llvm::Value* VectorSupportLibrary::Floor(llvm::Value* a) {
  AssertCorrectTypes({a});
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::floor, {a},
                                      {a->getType()}, b());
}

llvm::Value* VectorSupportLibrary::Div(llvm::Value* lhs, llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  if (scalar_type_->isFloatingPointTy()) {
    return b()->CreateFDiv(lhs, rhs, name());
  } else {
    LOG(FATAL) << "Division for integers is unimplemented";
  }
}

llvm::Value* VectorSupportLibrary::Clamp(llvm::Value* a,
                                         const llvm::APFloat& low,
                                         const llvm::APFloat& high) {
  CHECK(!low.isNaN());
  CHECK(!high.isNaN());
  CHECK(low.compare(high) == llvm::APFloat::cmpLessThan);

  AssertCorrectTypes({a});
  llvm::Type* type = a->getType();
  CHECK(scalar_type_->isFloatingPointTy());

  llvm::Value* low_value = GetConstantFloat(type, low);
  llvm::Value* high_value = GetConstantFloat(type, high);
  a = b_->CreateSelect(b_->CreateFCmpUGE(a, low_value), a, low_value);
  a = b_->CreateSelect(b_->CreateFCmpULE(a, high_value), a, high_value);
  return a;
}

llvm::Value* VectorSupportLibrary::FCmpEQMask(llvm::Value* lhs,
                                              llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  return I1ToFloat(b()->CreateFCmpOEQ(lhs, rhs, name()));
}

llvm::Value* VectorSupportLibrary::FCmpOLTMask(llvm::Value* lhs,
                                               llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  return I1ToFloat(b()->CreateFCmpOLT(lhs, rhs, name()));
}

llvm::Value* VectorSupportLibrary::FCmpULEMask(llvm::Value* lhs,
                                               llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  return I1ToFloat(b()->CreateFCmpULE(lhs, rhs, name()));
}

llvm::Value* VectorSupportLibrary::I1ToFloat(llvm::Value* i1) {
  bool is_vector = llvm::isa<llvm::VectorType>(i1->getType());
  llvm::Type* integer_type = IntegerTypeForFloatSize(is_vector);
  return b()->CreateBitCast(b()->CreateSExt(i1, integer_type, name()),
                            is_vector ? vector_type() : scalar_type(), name());
}

llvm::Type* VectorSupportLibrary::IntegerTypeForFloatSize(bool vector) {
  CHECK(scalar_type()->isFloatingPointTy());
  const llvm::DataLayout& data_layout =
      b()->GetInsertBlock()->getModule()->getDataLayout();
  int64 float_size_bits = data_layout.getTypeSizeInBits(scalar_type());
  llvm::Type* scalar_int_type = b()->getIntNTy(float_size_bits);
  if (vector) {
    return llvm::VectorType::get(scalar_int_type, vector_size());
  } else {
    return scalar_int_type;
  }
}

llvm::Value* VectorSupportLibrary::BroadcastScalar(llvm::Value* x) {
  CHECK_EQ(x->getType(), scalar_type());
  return b()->CreateVectorSplat(vector_size(), x, name());
}

llvm::Value* VectorSupportLibrary::FloatAnd(llvm::Value* lhs,
                                            llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  llvm::Type* int_type =
      IntegerTypeForFloatSize(lhs->getType() == vector_type());
  return b()->CreateBitCast(
      b()->CreateAnd(b()->CreateBitCast(lhs, int_type, name()),
                     b()->CreateBitCast(rhs, int_type, name()), name()),
      vector_type());
}

llvm::Value* VectorSupportLibrary::FloatNot(llvm::Value* lhs) {
  AssertCorrectTypes({lhs});
  llvm::Type* int_type =
      IntegerTypeForFloatSize(lhs->getType() == vector_type());
  return b()->CreateBitCast(
      b()->CreateNot(b()->CreateBitCast(lhs, int_type, name()), name()),
      vector_type());
}

llvm::Value* VectorSupportLibrary::FloatOr(llvm::Value* lhs, llvm::Value* rhs) {
  AssertCorrectTypes({lhs, rhs});
  llvm::Type* int_type =
      IntegerTypeForFloatSize(lhs->getType() == vector_type());
  return b()->CreateBitCast(
      b()->CreateOr(b()->CreateBitCast(lhs, int_type, name()),
                    b()->CreateBitCast(rhs, int_type, name()), name()),
      vector_type(), name());
}

llvm::Value* VectorSupportLibrary::AddInternal(llvm::Value* lhs,
                                               llvm::Value* rhs) {
  if (scalar_type_->isFloatingPointTy()) {
    return b()->CreateFAdd(lhs, rhs, name());
  } else {
    return b()->CreateAdd(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::ComputeOffsetPointer(
    llvm::Value* base_pointer, llvm::Value* offset_elements) {
  if (base_pointer->getType() != scalar_pointer_type()) {
    base_pointer =
        b()->CreateBitCast(base_pointer, scalar_pointer_type(), name());
  }
  return b()->CreateInBoundsGEP(base_pointer, {offset_elements}, name());
}

llvm::Value* VectorSupportLibrary::LoadVector(llvm::Value* pointer) {
  if (pointer->getType() != vector_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, vector_pointer_type(), name());
  }
  return b()->CreateAlignedLoad(
      pointer, ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_), name());
}

llvm::Value* VectorSupportLibrary::LoadScalar(llvm::Value* pointer) {
  if (pointer->getType() != scalar_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  return b()->CreateAlignedLoad(
      pointer, ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_), name());
}

void VectorSupportLibrary::StoreVector(llvm::Value* value,
                                       llvm::Value* pointer) {
  AssertCorrectTypes({value});
  if (pointer->getType() != vector_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, vector_pointer_type());
  }
  b()->CreateAlignedStore(value, pointer,
                          ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_));
}

void VectorSupportLibrary::StoreScalar(llvm::Value* value,
                                       llvm::Value* pointer) {
  AssertCorrectTypes({value});
  if (pointer->getType() != scalar_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  b()->CreateAlignedStore(value, pointer,
                          ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_));
}

llvm::Value* VectorSupportLibrary::LoadBroadcast(llvm::Value* pointer) {
  if (pointer->getType() != scalar_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  return b()->CreateVectorSplat(vector_size(), b()->CreateLoad(pointer),
                                name());
}

llvm::Value* VectorSupportLibrary::AddReduce(llvm::Value* vector) {
  llvm::SmallVector<llvm::Constant*, 32> mask(vector_size(), nullptr);
  for (unsigned i = vector_size(); i != 1; i >>= 1) {
    // On every iteration, we shuffle half of the remaining lanes to the top
    // half of shuffle, and add two old and the new vector.

    for (unsigned j = 0; j < vector_size(); ++j) {
      if (j < (i / 2)) {
        mask[j] = b()->getInt32(i / 2 + j);
      } else {
        mask[j] = llvm::UndefValue::get(b()->getInt32Ty());
      }
    }

    llvm::Value* half_remaining_lanes =
        b()->CreateShuffleVector(vector, llvm::UndefValue::get(vector_type()),
                                 llvm::ConstantVector::get(mask), "");
    vector = Add(vector, half_remaining_lanes);
  }

  return b()->CreateExtractElement(vector, b()->getInt32(0), name());
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
    mask_a.push_back(b()->getInt32(increment + i));
    mask_b.push_back(b()->getInt32(increment + i + 1));
  }
  for (int i = 0; i < vector_size(); i += 2) {
    int increment = i < vector_size() / 2 ? (vector_size() / 2) : vector_size();
    mask_a.push_back(b()->getInt32(increment + i));
    mask_b.push_back(b()->getInt32(increment + i + 1));
  }

  llvm::Value* shuffle_0 =
      b()->CreateShuffleVector(lhs, rhs, llvm::ConstantVector::get(mask_a));
  llvm::Value* shuffle_1 =
      b()->CreateShuffleVector(lhs, rhs, llvm::ConstantVector::get(mask_b));

  return Add(shuffle_0, shuffle_1);
}

llvm::Value* VectorSupportLibrary::ExtractLowHalf(llvm::Value* vector) {
  llvm::SmallVector<llvm::Constant*, 32> mask;
  for (int i = 0; i < vector_size() / 2; i++) {
    mask.push_back(b()->getInt32(i));
  }

  return b()->CreateShuffleVector(vector, llvm::UndefValue::get(vector_type()),
                                  llvm::ConstantVector::get(mask));
}

llvm::Value* VectorSupportLibrary::ExtractHighHalf(llvm::Value* vector) {
  llvm::SmallVector<llvm::Constant*, 32> mask;
  for (int i = 0; i < vector_size() / 2; i++) {
    mask.push_back(b()->getInt32(i + vector_size() / 2));
  }

  return b()->CreateShuffleVector(vector, llvm::UndefValue::get(vector_type()),
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
      result[i] = Add(result[i],
                      b()->CreateExtractElement(init_values, b()->getInt32(i)));
    }
  }
  return result;
}

std::vector<llvm::Value*>
VectorSupportLibrary::ComputeAvxOptimizedHorizontalSums(
    std::vector<llvm::Value*> vectors, llvm::Value* init_values) {
  // vectors are N llvm vector values, each with N elements.
  int64 lane_width = vectors.size();

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

  // `low` has the first `lane_width / 2` horizontal reductions, and `high` has
  // the next `lane_width / 2` horizontal reductions.

  std::vector<llvm::Value*> results;
  for (int i = 0; i < lane_width; i++) {
    llvm::Value* scalar_result =
        b()->CreateExtractElement(i < (lane_width / 2) ? low : high,
                                  b()->getInt32(i % (lane_width / 2)), name());
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

LlvmVariable::LlvmVariable(llvm::Type* type, llvm::IRBuilder<>* b) : b_(b) {
  alloca_ = llvm_ir::EmitAllocaAtFunctionEntry(type, "", b_);
}

llvm::Value* LlvmVariable::Get() const { return b_->CreateLoad(alloca_); }

void LlvmVariable::Set(llvm::Value* new_value) {
  b_->CreateStore(new_value, alloca_);
}

TileVariable::TileVariable(VectorSupportLibrary* vector_support,
                           std::vector<llvm::Value*> initial_value) {
  for (llvm::Value* initial_vector_value : initial_value) {
    storage_.emplace_back(vector_support, initial_vector_value);
  }
}

std::vector<llvm::Value*> TileVariable::Get() const {
  std::vector<llvm::Value*> result;
  absl::c_transform(storage_, std::back_inserter(result),
                    [&](VectorVariable vect_var) { return vect_var.Get(); });
  return result;
}

void TileVariable::Set(absl::Span<llvm::Value* const> value) {
  CHECK_EQ(value.size(), storage_.size());
  for (int64 i = 0, e = value.size(); i < e; i++) {
    storage_[i].Set(value[i]);
  }
}

}  // namespace cpu
}  // namespace xla
