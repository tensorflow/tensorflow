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

#include "xla/service/llvm_ir/ir_array.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace llvm_ir {

void IrArray::Index::Delinearize(std::vector<llvm::Value*>* multidim,
                                 llvm::Value* linear, const Shape& shape,
                                 llvm::IRBuilderBase* b) const {
  int64_t divisor = 1;
  const Layout& layout = shape.layout();
  for (int64_t i = 0; i < layout.minor_to_major().size(); ++i) {
    int64_t dimension = layout.minor_to_major(i);
    int64_t size_of_current_dimension = shape.dimensions(dimension);

    // If i is not the last dimension, compute
    //   (linear_index / divisor) % current_dimension.
    // If i is the last dimension, we can skip the mod, because we assume that
    // linear is in bounds.
    //
    // TODO(jlebar): We could add bounds checks here and elsewhere in this file,
    // guarded under some sort of xla-memcheck flag.  This might be particularly
    // useful because cuda-memcheck can't help us much in XLA: Most of our
    // memory lives in one big allocation, so cuda-memcheck can't detect
    // out-of-bounds accesses.
    auto* quot = b->CreateUDiv(linear, GetConstantWithIndexType(divisor));
    if (i < layout.minor_to_major().size() - 1) {
      (*multidim)[dimension] = b->CreateURem(
          quot, GetConstantWithIndexType(size_of_current_dimension));
    } else {
      (*multidim)[dimension] = quot;
    }
    divisor *= size_of_current_dimension;
  }
}

IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      llvm::IRBuilderBase* b)
    : multidim_(shape.dimensions().size()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, b);
}

IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      const Shape& shape, llvm::Type* index_type)
    : multidim_(multidim.begin(), multidim.end()),
      linear_(nullptr),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()),
      index_type_(index_type) {
  CHECK_NE(index_type_, nullptr);
  CHECK_EQ(shape.dimensions().size(), multidim.size());
  for (const auto* dim : multidim) {
    CHECK_NE(dim, nullptr);
  }
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
}

IrArray::IrArray(llvm::Value* base_ptr, llvm::Type* pointee_type, Shape shape)
    : base_ptr_(base_ptr),
      pointee_type_(pointee_type),
      shape_(std::move(shape)) {
  CHECK_OK(ShapeUtil::ValidateShape(shape_));
  CHECK(base_ptr_->getType()->isPointerTy());
  int depth = 0;
  element_type_ = pointee_type;
  while (llvm::ArrayType* array_type =
             llvm::dyn_cast<llvm::ArrayType>(element_type_)) {
    element_type_ = array_type->getElementType();
    ++depth;
  }

  if (!shape_.IsArray() || ShapeUtil::IsScalar(shape_)) {
    DCHECK(depth == 1 || depth == 0) << depth;
  } else {
    DCHECK_EQ(depth, shape_.dimensions().size()) << shape_.ToString();
  }
}

// Returns whether the given linear index is valid on the given shape.
bool IrArray::Index::LinearValidOnShape(const Shape& a) const {
  auto b = ShapeUtil::MakeShape(a.element_type(), dims_);
  *b.mutable_layout() = layout_;
  return linear_ != nullptr &&
         ShapeUtil::ElementsIn(a) == ShapeUtil::ElementsIn(b) &&
         ShapeUtil::ReshapeIsBitcast(a, b);
}

llvm::Value* IrArray::Index::Linearize(absl::Span<const int64_t> dimensions,
                                       llvm::IRBuilderBase* builder) const {
  // Each dimension is multiplied by the product of the sizes of all
  // earlier dimensions and added to the accumulator logical_linear_index.
  CHECK_EQ(size(), dimensions.size());
  llvm::Value* logical_linear_index = GetConstantWithIndexType(0);
  int64_t multiplier = 1;
  for (ssize_t i = 0; i < size(); ++i) {
    int64_t dimension = layout_.minor_to_major(i);
    llvm::Value* addend = builder->CreateMul(
        (*this)[dimension], GetConstantWithIndexType(multiplier), "",
        /*HasNUW=*/true, /*HasNSW=*/true);
    addend = builder->CreateZExtOrTrunc(addend, index_type_);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              /*HasNUW=*/true, /*HasNSW=*/true);
    multiplier *= dimensions[dimension];
  }
  return logical_linear_index;
}

llvm::Value* IrArray::EmitArrayElementAddress(const IrArray::Index& index,
                                              llvm::IRBuilderBase* b,
                                              absl::string_view name,
                                              bool use_linear_index,
                                              llvm::Value** bit_offset) const {
  if (ShapeUtil::IsScalar(shape_)) {
    if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
      CHECK_NE(bit_offset, nullptr);
      *bit_offset = b->getInt8(0);
    }
    // Special handling of scalars: a scalar pretends to have the same value for
    // every index, thus effectively implementing broadcasting of its value
    // over higher-rank arrays.
    return base_ptr_;
  }
  CHECK_EQ(index.size(), shape_.dimensions().size());
  CHECK(index.ShapeIsCompatible(shape_))
      << "Shape " << index.AsShapeWithType(shape_.element_type()).ToString(true)
      << " is not compatible with " << shape_.ToString(true);

  if (use_linear_index && index.LinearValidOnShape(shape_)) {
    return EmitLinearArrayElementAddress(index, b, name, bit_offset);
  }

  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    // Subbyte types require the use of a linear index, because the GEP
    // multi-indexing logic below does not properly handle packed subbyte types.
    IrArray::Index linear_index = index;
    if (!index.LinearValidOnShape(shape_)) {
      // Create a valid linear index.
      std::vector<int64_t> dimensions;
      dimensions.reserve(shape_.dimensions().size());
      for (int64_t i = 0; i < shape_.dimensions().size(); ++i) {
        dimensions.push_back(shape_.dimensions(i));
      }
      llvm::Value* linearized = index.Linearize(dimensions, b);
      linear_index = IrArray::Index(linearized, shape_, b);
    }
    return EmitLinearArrayElementAddress(linear_index, b, name, bit_offset);
  }

  std::vector<llvm::Value*> actual_index;
  for (int64_t i = 0; i < index.size(); ++i) {
    // When dimension i is of size 1, LLVM optimization is able to replace
    // index[i] with 0. However, setting index[i] to 0 here still allows LLVM to
    // produce better code in some cases.
    auto dim = shape_.dimensions(i);
    actual_index.push_back(
        dim == 1 ? llvm::ConstantInt::get(index[i]->getType(), 0) : index[i]);
  }

  // "base_ptr_" has the type of "<ir_type_for_its_shape>*"
  // (e.g. [3 x [2 x float]]*). Therefore, the address of the indexed element
  // should be computed by
  //
  //   getelementptr base_ptr_, 0, most major index, ..., most minor index
  CHECK_GT(index.size(), 0);
  std::vector<llvm::Value*> gep_indices(
      1, llvm::ConstantInt::get(index[0]->getType(), 0));
  for (int64_t i = 0; i < shape_.dimensions().size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    gep_indices.push_back(actual_index[dimension]);
  }
  return b->CreateInBoundsGEP(pointee_type_, base_ptr_, gep_indices,
                              llvm_ir::AsStringRef(name));
}

llvm::Value* IrArray::EmitLinearArrayElementAddress(
    const IrArray::Index& index, llvm::IRBuilderBase* b, absl::string_view name,
    llvm::Value** bit_offset) const {
  CHECK(index.LinearValidOnShape(shape_));
  llvm::Type* type =
      PrimitiveTypeToIrType(shape_.element_type(), b->getContext());
  if (!primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    auto linear_index = llvm::dyn_cast<llvm::BinaryOperator>(index.linear());
    if (linear_index && (linear_index->getOpcode() == llvm::Instruction::Add)) {
      llvm::Value* index_operand_0 = linear_index->getOperand(0);
      llvm::Value* index_operand_1 = linear_index->getOperand(1);
      llvm::Value* ptr_address =
          b->CreateGEP(type, base_ptr_, index_operand_0, "");

      return b->CreateInBoundsGEP(type, ptr_address, index_operand_1,
                                  llvm_ir::AsStringRef(name));
    } else {
      return b->CreateInBoundsGEP(type, base_ptr_, index.linear(),
                                  llvm_ir::AsStringRef(name));
    }
  }

  // Handle sub-bytes by dividing index by the number of elements per byte.
  // Arrays are represented in LLVM IR as an array of i8 value where each i8
  // value stores multiple values.
  llvm::Type* index_type = index.linear()->getType();
  auto bit_width = primitive_util::BitWidth(shape_.element_type());
  llvm::Value* elements_per_byte =
      llvm::ConstantInt::get(index_type, 8 / bit_width);
  llvm::Value* remainder = b->CreateURem(index.linear(), elements_per_byte);
  llvm::Value* byte_offset = b->CreateUDiv(index.linear(), elements_per_byte);

  CHECK_NE(bit_offset, nullptr);
  *bit_offset = b->CreateIntCast(
      b->CreateMul(remainder, llvm::ConstantInt::get(index_type, bit_width)),
      b->getInt8Ty(), /*isSigned=*/false);
  return b->CreateInBoundsGEP(b->getInt8Ty(), base_ptr_, byte_offset,
                              llvm_ir::AsStringRef(name));
}

void IrArray::AnnotateLoadStoreInstructionWithMetadata(
    llvm::Instruction* instruction) const {
  CHECK(llvm::isa<llvm::LoadInst>(instruction) ||
        llvm::isa<llvm::StoreInst>(instruction));

  for (const auto& kind_md_pair : metadata_) {
    instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
  }
}

llvm::Value* IrArray::EmitReadArrayElement(const Index& index,
                                           llvm::IRBuilderBase* b,
                                           absl::string_view name,
                                           bool use_linear_index) const {
  llvm::Value* bit_offset = nullptr;
  llvm::Value* element_address =
      EmitArrayElementAddress(index, b, name, use_linear_index, &bit_offset);
  llvm::Type* load_type =
      primitive_util::IsSubByteNonPredType(shape_.element_type())
          ? b->getInt8Ty()
          : element_type_;
  llvm::LoadInst* load =
      b->CreateLoad(load_type, element_address, llvm_ir::AsStringRef(name));
  AnnotateLoadStoreInstructionWithMetadata(load);
  llvm::Value* elem = load;
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    llvm::Value* shifted = b->CreateLShr(load, bit_offset);
    elem = b->CreateTrunc(
        shifted, b->getIntNTy(primitive_util::BitWidth(shape_.element_type())));
  }
  return elem;
}

void IrArray::EmitWriteArrayElement(const Index& index, llvm::Value* value,
                                    llvm::IRBuilderBase* b,
                                    bool use_linear_index) const {
  llvm::Value* bit_offset = nullptr;
  llvm::Value* element_address =
      EmitArrayElementAddress(index, b, "", use_linear_index, &bit_offset);
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    // Read a byte, replace the high-order or low-order bits with 'value',
    // and write it back.
    llvm::LoadInst* load = b->CreateLoad(b->getInt8Ty(), element_address);
    AnnotateLoadStoreInstructionWithMetadata(load);
    value = b->CreateIntCast(value, b->getInt8Ty(),
                             /*isSigned=*/false);
    value = b->CreateShl(value, bit_offset);
    auto bit_width = primitive_util::BitWidth(shape_.element_type());
    // This is equivalent to:
    // mask = ~(LsbMask(bit_width) << bit_offset)
    llvm::Value* mask = b->getInt8(~LsbMask<uint8_t>(bit_width));
    mask = b->CreateIntrinsic(b->getInt8Ty(), llvm::Intrinsic::fshl,
                              {mask, mask, bit_offset});
    llvm::Value* masked_load = b->CreateAnd(load, mask);
    value = b->CreateOr(masked_load, value);
  }
  llvm::StoreInst* store = b->CreateStore(value, element_address);
  AnnotateLoadStoreInstructionWithMetadata(store);
}

bool IrArray::Index::ShapeIsCompatible(const Shape& a, const Shape& b) {
  // Compute strides for two sides of the comparison. Sometimes different shapes
  // give the same strides:
  //   [10, 20, 30, 1]{3,2,1,0} vs [10, 20, 1, 30]{3,2,1,0}
  // which should be considered compatible.
  const auto get_strides = [](const Shape& shape) {
    int rank = shape.dimensions().size();
    int64_t stride = 1;
    std::vector<int64_t> strides;
    for (int i = 0; i < rank; i++) {
      auto dim = shape.dimensions(shape.layout().minor_to_major(i));
      if (dim != 1) {
        stride *= dim;
        strides.push_back(stride);
      }
    }
    return strides;
  };

  return get_strides(a) == get_strides(b);
}

}  // namespace llvm_ir
}  // namespace xla
