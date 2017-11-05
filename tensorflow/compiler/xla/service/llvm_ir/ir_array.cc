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

#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      llvm::IRBuilder<>* ir_builder)
    : multidim_(ShapeUtil::Rank(shape)),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  int64 divisor = 1;
  for (int64 dimension : layout_.minor_to_major()) {
    int64 size_of_current_dimension = shape.dimensions(dimension);
    // Emit IR instructions that compute
    //   (linear_index / divisor) % current_dimension
    multidim_[dimension] = ir_builder->CreateURem(
        ir_builder->CreateUDiv(linear, ir_builder->getInt64(divisor)),
        ir_builder->getInt64(size_of_current_dimension));
    divisor *= size_of_current_dimension;
  }
}

IrArray::Index::Index(tensorflow::gtl::ArraySlice<llvm::Value*> multidim,
                      llvm::Value* linear, const Shape& shape)
    : multidim_(multidim.begin(), multidim.end()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
}

IrArray::Index::Index(tensorflow::gtl::ArraySlice<llvm::Value*> multidim,
                      const Shape& shape, llvm::IRBuilder<>* ir_builder)
    : multidim_(multidim.begin(), multidim.end()),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  CHECK(LayoutUtil::HasLayout(shape));
  linear_ = Linearize(AsInt64Slice(shape.dimensions()), ir_builder);
}

IrArray::IrArray(llvm::Value* base_ptr, const Shape& shape)
    : base_ptr_(base_ptr), shape_(&shape) {
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(base_ptr_->getType()->isPointerTy());
  int depth = 0;
  element_type_ =
      llvm::cast<llvm::PointerType>(base_ptr_->getType())->getElementType();
  while (llvm::ArrayType* array_type =
             llvm::dyn_cast<llvm::ArrayType>(element_type_)) {
    element_type_ = array_type->getElementType();
    ++depth;
  }

  if (!ShapeUtil::IsArray(*shape_) || ShapeUtil::IsScalar(*shape_)) {
    DCHECK(depth == 1 || depth == 0) << depth;
  } else {
    DCHECK_EQ(depth, ShapeUtil::Rank(*shape_)) << shape.ShortDebugString();
  }
}

// Returns whether given linear index valid on given shape.
bool IrArray::Index::LinearValidOnShape(const Shape& a) const {
  auto b = ShapeUtil::MakeShape(PRED /* irrelevant */, dims_);
  *b.mutable_layout() = layout_;
  return linear_ != nullptr &&
         ContainersEqual(
             ShapeUtil::StripDegenerateDimensions(a).dimensions(),
             ShapeUtil::StripDegenerateDimensions(b).dimensions()) &&
         LayoutUtil::Equal(ShapeUtil::StripDegenerateDimensions(a).layout(),
                           ShapeUtil::StripDegenerateDimensions(b).layout());
}

IrArray::Index IrArray::Index::SourceIndexOfReshape(
    const Shape& output_shape, const Shape& input_shape,
    llvm::IRBuilder<>* builder) const {
  const auto& target_index = *this;
  CHECK_EQ(target_index.size(), ShapeUtil::Rank(output_shape));
  std::vector<std::pair<int64, int64>> common_factors =
      CommonFactors(AsInt64Slice(input_shape.dimensions()),
                    AsInt64Slice(output_shape.dimensions()));
  std::vector<llvm::Value*> source_multidim_index(
      ShapeUtil::Rank(input_shape),
      llvm::UndefValue::get(builder->getInt64Ty()));
  // We compute the source indices in each common factor from only the target
  // indices in the same common factor.
  for (ssize_t k = common_factors.size() - 2; k >= 0; --k) {
    llvm::Value* logical_linear_index =
        Index(tensorflow::gtl::ArraySlice<llvm::Value*>(
                  multidim_, common_factors[k].second,
                  common_factors[k + 1].second - common_factors[k].second))
            .Linearize(
                tensorflow::gtl::ArraySlice<int64>(
                    AsInt64Slice(output_shape.dimensions()),
                    common_factors[k].second,
                    common_factors[k + 1].second - common_factors[k].second),
                builder);
    // Delinearizes logical_linear_index for the source array in row-major
    // collapsed order. The first rank-1 indices are the remainder of the
    // linear index by each dimension size.
    for (int64 i = common_factors[k + 1].first - 1;
         i >= common_factors[k].first; --i) {
      llvm::Value* divisor = builder->getInt64(input_shape.dimensions(i));
      if (input_shape.dimensions(i) == 1) {
        source_multidim_index[i] = builder->getInt64(0);
      } else if (i == common_factors[k].first) {
        source_multidim_index[i] = logical_linear_index;
      } else {
        source_multidim_index[i] =
            builder->CreateURem(logical_linear_index, divisor);
      }
      logical_linear_index = builder->CreateUDiv(logical_linear_index, divisor);
    }
  }

  if (linear() != nullptr &&
      ShapeUtil::ReshapeIsBitcast(input_shape, output_shape)) {
    return Index(source_multidim_index, linear(), input_shape);
  }
  return Index(source_multidim_index);
}

IrArray::Index IrArray::Index::SourceIndexOfSlice(
    const Shape& shape, tensorflow::gtl::ArraySlice<int64> starts,
    tensorflow::gtl::ArraySlice<int64> strides,
    llvm::IRBuilder<>* builder) const {
  Index source_index(multidim_.size());
  for (int i = 0; i < multidim_.size(); ++i) {
    int64 stride = strides[i];
    auto type = multidim_[i]->getType();

    if (stride != 1) {
      source_index[i] = builder->CreateAdd(
          builder->CreateMul(multidim_[i],
                             llvm::ConstantInt::get(type, stride)),
          llvm::ConstantInt::get(type, starts[i]));
    } else {
      source_index[i] = builder->CreateAdd(
          multidim_[i], llvm::ConstantInt::get(type, starts[i]));
    }
  }
  return source_index;
}

IrArray::Index IrArray::Index::SourceIndexOfTranspose(
    const Shape& shape, const Shape& operand_shape,
    tensorflow::gtl::ArraySlice<int64> dimension_mapping,
    llvm::IRBuilder<>* builder) const {
  std::vector<llvm::Value*> operand_multidim_index =
      Permute(dimension_mapping, multidim());
  if (linear() != nullptr &&
      ShapeUtil::TransposeIsBitcast(operand_shape, shape, dimension_mapping)) {
    return Index(operand_multidim_index, linear(), operand_shape);
  }
  return Index(operand_multidim_index);
}

llvm::Value* IrArray::Index::Linearize(
    tensorflow::gtl::ArraySlice<int64> dimensions,
    llvm::IRBuilder<>* builder) const {
  // Each dimension is multiplied by the product of the sizes of all
  // earlier dimensions and added to the accumulator logical_linear_index.
  llvm::Value* logical_linear_index = builder->getInt64(0);
  int64 multiplier = 1;
  for (ssize_t i = size() - 1; i >= 0; --i) {
    llvm::Value* addend =
        builder->CreateMul((*this)[i], builder->getInt64(multiplier), "",
                           /*HasNUW=*/true, /*HasNSW=*/true);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              /*HasNUW=*/true, /*HasNSW=*/true);
    multiplier *= dimensions[i];
  }
  return logical_linear_index;
}

llvm::Value* IrArray::EmitArrayElementAddress(
    const IrArray::Index& index, llvm::IRBuilder<>* ir_builder,
    tensorflow::StringPiece name) const {
  if (ShapeUtil::IsScalar(*shape_)) {
    // Special handling of scalars: a scalar pretends to have the same value for
    // every index, thus effectively implementing broadcasting of its value
    // over higher-rank arrays.
    return base_ptr_;
  }
  CHECK_EQ(index.size(), ShapeUtil::Rank(*shape_));

  std::vector<llvm::Value*> actual_index;
  bool is_implicit_broadcast = false;
  // We perform broadcasting when the operand shape has dimension(s) of size
  // 1. In this case we fix the index value for that dimension to zero. This
  // effectively broadcasts along this dimension.
  for (int64 i = 0; i < index.size(); ++i) {
    auto dim = shape_->dimensions(i);
    actual_index.push_back(dim == 1 ? ir_builder->getInt64(0) : index[i]);
    is_implicit_broadcast |= dim == 1;
  }

  if (!is_implicit_broadcast && index.LinearValidOnShape(*shape_)) {
    llvm::Module* module =
        ir_builder->GetInsertBlock()->getParent()->getParent();
    return ir_builder->CreateInBoundsGEP(
        ir_builder->CreateBitCast(
            base_ptr_, PrimitiveTypeToIrType(shape_->element_type(), module)
                           ->getPointerTo()),
        {index.linear()}, llvm_ir::AsStringRef(name));
  }

  // "base_ptr_" has the type of "<ir_type_for_its_shape>*"
  // (e.g. [3 x [2 x float]]*). Therefore, the address of the indexed element
  // should be computed by
  //
  //   getelementptr base_ptr_, 0, most major index, ..., most minor index
  std::vector<llvm::Value*> gep_indices(1, ir_builder->getInt64(0));
  for (int64 i = shape_->layout().minor_to_major_size() - 1; i >= 0; --i) {
    int64 dimension = shape_->layout().minor_to_major(i);
    gep_indices.push_back(actual_index[dimension]);
  }
  return ir_builder->CreateInBoundsGEP(base_ptr_, gep_indices,
                                       llvm_ir::AsStringRef(name));
}

void IrArray::AnnotateLoadStoreInstructionWithMetadata(
    llvm::Instruction* instruction) const {
  CHECK(llvm::isa<llvm::LoadInst>(instruction) ||
        llvm::isa<llvm::StoreInst>(instruction));

  for (const auto& kind_md_pair : metadata_) {
    CHECK(kind_md_pair.first != llvm::LLVMContext::MD_invariant_load ||
          llvm::isa<llvm::LoadInst>(instruction));
    instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
  }
}

llvm::Value* IrArray::EmitReadArrayElement(const Index& index,
                                           llvm::IRBuilder<>* ir_builder,
                                           tensorflow::StringPiece name) const {
  llvm::Value* element_address =
      EmitArrayElementAddress(index, ir_builder, name);
  llvm::LoadInst* load = ir_builder->CreateLoad(element_address);
  AnnotateLoadStoreInstructionWithMetadata(load);
  return load;
}

void IrArray::EmitWriteArrayElement(const Index& index, llvm::Value* value,
                                    llvm::IRBuilder<>* ir_builder) const {
  llvm::Value* element_address = EmitArrayElementAddress(index, ir_builder);
  llvm::StoreInst* store = ir_builder->CreateStore(value, element_address);
  AnnotateLoadStoreInstructionWithMetadata(store);
}

IrArray IrArray::CastToShape(const Shape& new_shape,
                             llvm::IRBuilder<>* ir_builder) const {
  llvm::Module* module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::Type* new_ir_type = llvm_ir::ShapeToIrType(new_shape, module);
  return IrArray(
      ir_builder->CreatePointerCast(base_ptr_, new_ir_type->getPointerTo()),
      new_shape);
}

/* static */ IrArray::Index IrArray::BumpIndex(const Index& index,
                                               int64 which_dimension,
                                               int64 addend,
                                               llvm::IRBuilder<>* ir_builder) {
  Index new_index = index;
  new_index[which_dimension] = ir_builder->CreateAdd(
      index[which_dimension], ir_builder->getInt64(addend), "", /*HasNUW=*/true,
      /*HasNSW=*/true);
  return new_index;
}

}  // namespace llvm_ir
}  // namespace xla
