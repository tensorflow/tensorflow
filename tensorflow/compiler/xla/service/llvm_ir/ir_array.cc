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

void IrArray::Index::Delinearize(std::vector<llvm::Value*>* multidim,
                                 llvm::Value* linear, const Shape& shape,
                                 llvm::IRBuilder<>* b) const {
  int64 divisor = 1;
  const Layout& layout = shape.layout();
  for (int64 i = 0; i < layout.minor_to_major_size(); ++i) {
    int64 dimension = layout.minor_to_major(i);
    int64 size_of_current_dimension = shape.dimensions(dimension);

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
    if (i < layout.minor_to_major_size() - 1) {
      (*multidim)[dimension] = b->CreateURem(
          quot, GetConstantWithIndexType(size_of_current_dimension));
    } else {
      (*multidim)[dimension] = quot;
    }
    divisor *= size_of_current_dimension;
  }
}

IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      llvm::IRBuilder<>* b)
    : multidim_(ShapeUtil::Rank(shape)),
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

IrArray::Index::Index(tensorflow::gtl::ArraySlice<llvm::Value*> multidim,
                      llvm::Value* linear, const Shape& shape)
    : multidim_(multidim.begin(), multidim.end()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  if (size()) {
    index_type_ = multidim_[0]->getType();
  } else {
    CHECK_NE(linear_, nullptr);
    index_type_ = linear_->getType();
  }
  CHECK_NE(index_type_, nullptr);
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
}

IrArray::Index::Index(tensorflow::gtl::ArraySlice<llvm::Value*> multidim,
                      const Shape& shape, llvm::IRBuilder<>* b)
    : multidim_(multidim.begin(), multidim.end()),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_GT(multidim_.size(), 0);
  index_type_ = multidim[0]->getType();
  CHECK_NE(index_type_, nullptr);
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  CHECK(LayoutUtil::HasLayout(shape));
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

// Returns whether the given linear index is valid on the given shape.
bool IrArray::Index::LinearValidOnShape(const Shape& a) const {
  auto b = ShapeUtil::MakeShape(a.element_type(), dims_);
  *b.mutable_layout() = layout_;
  return linear_ != nullptr &&
         ShapeUtil::ElementsIn(a) == ShapeUtil::ElementsIn(b) &&
         ShapeUtil::ReshapeIsBitcast(a, b);
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
      ShapeUtil::Rank(input_shape), llvm::UndefValue::get(index_type_));
  // We compute the source indices in each common factor from only the target
  // indices in the same common factor.
  for (ssize_t k = common_factors.size() - 2; k >= 0; --k) {
    llvm::Value* logical_linear_index =
        Index(tensorflow::gtl::ArraySlice<llvm::Value*>(
                  multidim_, common_factors[k].second,
                  common_factors[k + 1].second - common_factors[k].second),
              index_type_)
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
      llvm::Value* divisor =
          GetConstantWithIndexType(input_shape.dimensions(i));
      if (input_shape.dimensions(i) == 1) {
        source_multidim_index[i] = GetConstantWithIndexType(0);
      } else if (i == common_factors[k].first) {
        source_multidim_index[i] = logical_linear_index;
      } else {
        source_multidim_index[i] =
            builder->CreateURem(logical_linear_index, divisor);
      }
      logical_linear_index = builder->CreateUDiv(logical_linear_index, divisor);
    }
  }

  if (linear() != nullptr && LayoutUtil::HasLayout(input_shape) &&
      LayoutUtil::HasLayout(output_shape) &&
      ShapeUtil::ReshapeIsBitcast(input_shape, output_shape)) {
    return Index(source_multidim_index, linear(), input_shape);
  }
  return Index(source_multidim_index, index_type_);
}

IrArray::Index IrArray::Index::SourceIndexOfSlice(
    const Shape& shape, tensorflow::gtl::ArraySlice<int64> starts,
    tensorflow::gtl::ArraySlice<int64> strides,
    llvm::IRBuilder<>* builder) const {
  Index source_index(index_type_, multidim_.size());
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

  if (linear() != nullptr && LayoutUtil::HasLayout(operand_shape) &&
      LayoutUtil::HasLayout(shape) &&
      ShapeUtil::TransposeIsBitcast(operand_shape, shape, dimension_mapping)) {
    return Index(operand_multidim_index, linear(), operand_shape);
  }

  return Index(operand_multidim_index);
}

IrArray::Index IrArray::Index::SourceIndexOfBitcast(
    const Shape& shape, const Shape& operand_shape,
    llvm::IRBuilder<>* builder) const {
  CHECK(LayoutUtil::HasLayout(shape) && LayoutUtil::HasLayout(operand_shape));
  // In case the bitcast is just a reshape, we can use SourceIndexOfReshape()
  // instead. This will reuse linear() if possible, so we don't have to build a
  // new 'linear_index'.
  if (ShapeUtil::ReshapeIsBitcast(operand_shape, shape)) {
    return SourceIndexOfReshape(shape, operand_shape, builder);
  }

  // First linearize the index coming from the output of the bitcast. We want
  // the physical index of the element in the buffer. This is like Linearize,
  // but takes the layout into account.
  int64 scale = 1;
  llvm::Value* linear_index = GetConstantWithIndexType(0);
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    linear_index = builder->CreateAdd(
        linear_index,
        builder->CreateMul(multidim_[dimension],
                           GetConstantWithIndexType(scale), "",
                           /*HasNUW=*/true, /*HasNSW=*/true),
        "", /*HasNUW=*/true, /*HasNSW=*/true);
    scale *= shape.dimensions(dimension);
  }

  // Now delinearize it for the input of the bitcast.
  std::vector<llvm::Value*> multi_index(operand_shape.dimensions_size());
  Delinearize(&multi_index, linear_index, operand_shape, builder);

  return Index(multi_index, linear_index, operand_shape);
}

IrArray::Index IrArray::Index::SourceIndexOfBroadcast(
    const Shape& shape, const Shape& operand_shape,
    tensorflow::gtl::ArraySlice<int64> dimension_mapping,
    llvm::IRBuilder<>* builder) const {
  int64 rank = ShapeUtil::Rank(operand_shape);
  std::vector<llvm::Value*> source_index(rank);
  for (int64 i = 0; i < rank; ++i) {
    source_index[i] = multidim_[dimension_mapping[i]];
  }
  if (linear_ == nullptr || !LayoutUtil::HasLayout(operand_shape) ||
      !LayoutUtil::HasLayout(shape)) {
    return Index(source_index, index_type_);
  }
  // High-level idea: we can reuse the linear index if the broadcasted
  // dimensions are contiguous, and this part of the operation is a bitcast.
  // The other dimensions can be masked out with a div and a mod operation.
  std::vector<int64> logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(shape.layout());
  int64 output_rank = ShapeUtil::Rank(shape);
  // The minimum physical dimension that is broadcasted.
  int64 min_broadcasted_dimension = output_rank;
  // The maximum physical dimension that is broadcasted.
  int64 max_broadcasted_dimension = -1;
  for (int64 i = 0; i < rank; ++i) {
    int64 physical_dim = logical_to_physical[dimension_mapping[i]];
    min_broadcasted_dimension =
        std::min(min_broadcasted_dimension, physical_dim);
    max_broadcasted_dimension =
        std::max(max_broadcasted_dimension, physical_dim);
  }
  bool contiguous_broadcast_dimensions =
      max_broadcasted_dimension - min_broadcasted_dimension == rank - 1;
  if (!contiguous_broadcast_dimensions) {
    return Index(source_index, index_type_);
  }
  // Check if the mapped dimensions are a bitcast.
  std::vector<int64> operand_logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(operand_shape.layout());
  for (int64 i = 0; i < rank; ++i) {
    if (operand_logical_to_physical[i] !=
        logical_to_physical[dimension_mapping[i]] - min_broadcasted_dimension) {
      return Index(source_index, index_type_);
    }
  }
  llvm::Value* linear = linear_;
  int64 divisor = 1;
  for (int64 i = max_broadcasted_dimension + 1; i < output_rank; ++i) {
    divisor *= shape.dimensions(LayoutUtil::Major(shape.layout(), i));
  }
  if (divisor > 1) {
    linear = builder->CreateUDiv(
        linear,
        IrArray::Index(linear->getType()).GetConstantWithIndexType(divisor));
  }
  if (min_broadcasted_dimension > 0) {
    int64 mod = 1;
    for (int64 i = min_broadcasted_dimension; i <= max_broadcasted_dimension;
         ++i) {
      mod *= shape.dimensions(LayoutUtil::Major(shape.layout(), i));
    }
    linear = builder->CreateURem(
        linear,
        IrArray::Index(linear->getType()).GetConstantWithIndexType(mod));
  }
  return Index(source_index, linear, operand_shape);
}

llvm::Value* IrArray::Index::Linearize(
    tensorflow::gtl::ArraySlice<int64> dimensions,
    llvm::IRBuilder<>* builder) const {
  // Each dimension is multiplied by the product of the sizes of all
  // earlier dimensions and added to the accumulator logical_linear_index.
  CHECK_EQ(size(), dimensions.size());
  llvm::Value* logical_linear_index = GetConstantWithIndexType(0);
  int64 multiplier = 1;
  for (ssize_t i = size() - 1; i >= 0; --i) {
    llvm::Value* addend =
        builder->CreateMul((*this)[i], GetConstantWithIndexType(multiplier), "",
                           /*HasNUW=*/true, /*HasNSW=*/true);
    addend = builder->CreateZExtOrTrunc(addend, index_type_);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              /*HasNUW=*/true, /*HasNSW=*/true);
    multiplier *= dimensions[i];
  }
  return logical_linear_index;
}

llvm::Value* IrArray::EmitArrayElementAddress(
    const IrArray::Index& index, llvm::IRBuilder<>* b,
    tensorflow::StringPiece name) const {
  if (ShapeUtil::IsScalar(*shape_)) {
    // Special handling of scalars: a scalar pretends to have the same value for
    // every index, thus effectively implementing broadcasting of its value
    // over higher-rank arrays.
    return base_ptr_;
  }
  CHECK_EQ(index.size(), ShapeUtil::Rank(*shape_));

  if (index.LinearValidOnShape(*shape_)) {
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    return b->CreateInBoundsGEP(
        b->CreateBitCast(base_ptr_,
                         PrimitiveTypeToIrType(shape_->element_type(), module)
                             ->getPointerTo()),
        {index.linear()}, llvm_ir::AsStringRef(name));
  }

  std::vector<llvm::Value*> actual_index;
  for (int64 i = 0; i < index.size(); ++i) {
    // When dimension i is of size 1, LLVM optimization is able to replace
    // index[i] with 0. However, setting index[i] to 0 here still allows LLVM to
    // produce better code in some cases.
    auto dim = shape_->dimensions(i);
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
  for (int64 i = 0; i < LayoutUtil::MinorToMajor(*shape_).size(); ++i) {
    int64 dimension = LayoutUtil::Major(shape_->layout(), i);
    gep_indices.push_back(actual_index[dimension]);
  }
  return b->CreateInBoundsGEP(base_ptr_, gep_indices,
                              llvm_ir::AsStringRef(name));
}

void IrArray::AnnotateLoadStoreInstructionWithMetadata(
    llvm::Instruction* instruction) const {
  CHECK(llvm::isa<llvm::LoadInst>(instruction) ||
        llvm::isa<llvm::StoreInst>(instruction));
  CHECK(!llvm::isa<llvm::StoreInst>(instruction) || !is_invariant_)
      << "Trying to create a store to an invariant IRArray.";

  for (const auto& kind_md_pair : metadata_) {
    instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
  }
}

llvm::Value* IrArray::EmitReadArrayElement(const Index& index,
                                           llvm::IRBuilder<>* b,
                                           tensorflow::StringPiece name) const {
  llvm::Value* element_address = EmitArrayElementAddress(index, b, name);
  llvm::LoadInst* load = b->CreateLoad(element_address);
  AnnotateLoadStoreInstructionWithMetadata(load);
  return load;
}

void IrArray::EmitWriteArrayElement(const Index& index, llvm::Value* value,
                                    llvm::IRBuilder<>* b) const {
  llvm::Value* element_address = EmitArrayElementAddress(index, b);
  llvm::StoreInst* store = b->CreateStore(value, element_address);
  AnnotateLoadStoreInstructionWithMetadata(store);
}

IrArray IrArray::CastToShape(const Shape& new_shape,
                             llvm::IRBuilder<>* b) const {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* new_ir_type = llvm_ir::ShapeToIrType(new_shape, module);
  IrArray new_irarray(
      b->CreatePointerCast(base_ptr_, new_ir_type->getPointerTo()), new_shape);
  new_irarray.metadata_ = metadata_;
  return new_irarray;
}

/* static */ IrArray::Index IrArray::BumpIndex(const Index& index,
                                               int64 which_dimension,
                                               int64 addend,
                                               llvm::IRBuilder<>* b) {
  Index new_index = index;
  new_index[which_dimension] = b->CreateAdd(
      index[which_dimension],
      llvm::ConstantInt::get(index[which_dimension]->getType(), addend), "",
      /*HasNUW=*/true,
      /*HasNSW=*/true);
  return new_index;
}

}  // namespace llvm_ir
}  // namespace xla
