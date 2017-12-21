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

#include "tensorflow/compiler/xla/literal_util.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
namespace {
using tensorflow::int64;

constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

// Converts between little and big endian, assuming elements in the array are 16
// bits long.
void ConvertEndianShort(char* bytes, int64 size) {
  CHECK_EQ(size / 2, 0);
  for (int64 i = 0; i < size; i += 2) {
    std::swap(bytes[i], bytes[i + 1]);
  }
}
}  // namespace

namespace xla {

std::ostream& operator<<(std::ostream& out, const Literal& literal) {
  out << literal.ToString();
  return out;
}

Literal::StrideConfig::StrideConfig(
    const Shape& source_shape, const Shape& dest_shape,
    tensorflow::gtl::ArraySlice<int64> dimensions)
    : dimensions(dimensions),
      base(dimensions.size(), 0),
      step(dimensions.size(), 1) {
  if (!dimensions.empty()) {
    // Selects the shape with the largest minor dimension as the one upon
    // which to run the tight stride loop.
    if (dimensions[LayoutUtil::Minor(source_shape.layout(), 0)] >=
        dimensions[LayoutUtil::Minor(dest_shape.layout(), 0)]) {
      minor_dimension = LayoutUtil::Minor(source_shape.layout(), 0);
      dest_stride = IndexUtil::GetDimensionStride(dest_shape, minor_dimension);
    } else {
      minor_dimension = LayoutUtil::Minor(dest_shape.layout(), 0);
      source_stride =
          IndexUtil::GetDimensionStride(source_shape, minor_dimension);
    }
    minor_loop_size = dimensions[minor_dimension];
    step[minor_dimension] = minor_loop_size;
  }
}

std::unique_ptr<Literal> Literal::CreateFromShape(const Shape& shape) {
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() = shape;
  if (ShapeUtil::IsTuple(shape)) {
    int64 num_elements = ShapeUtil::TupleElementCount(shape);
    literal->tuple_literals_.resize(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      std::unique_ptr<Literal> elem =
          CreateFromShape(ShapeUtil::GetTupleElementShape(shape, i));
      literal->tuple_literals_[i] = std::move(*elem);
    }
  } else {
    literal->Reserve(ShapeUtil::ElementsIn(literal->shape()));
  }
  return literal;
}

/* static */ std::unique_ptr<Literal> Literal::CreateFromDimensions(
    PrimitiveType primitive_type,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  return CreateFromShape(ShapeUtil::MakeShape(primitive_type, dimensions));
}

template <typename T>
Status Literal::CopyRange(const Literal& src_literal,
                          tensorflow::gtl::ArraySlice<int64> src_base,
                          tensorflow::gtl::ArraySlice<int64> dest_base,
                          tensorflow::gtl::ArraySlice<int64> copy_size) {
  const Shape& src_shape = src_literal.shape();
  const Shape& dest_shape = shape();
  tensorflow::gtl::ArraySlice<T> src_data = src_literal.GetArraySlice<T>();
  tensorflow::gtl::MutableArraySlice<T> dest_data = GetMutableArraySlice<T>();

  TF_RET_CHECK(ShapeUtil::Rank(src_shape) == src_base.size());
  TF_RET_CHECK(ShapeUtil::Rank(dest_shape) == dest_base.size());

  if (ShapeUtil::Rank(src_shape) == 0 || ShapeUtil::Rank(dest_shape) == 0) {
    // If any of the two shapes are scalars, we can just call the StridedCopy()
    // directly, and we know we will be copying only one value.
    TF_RET_CHECK(copy_size.empty());
    StridedCopy(dest_data, LinearIndex(dest_base), 0, src_data,
                src_literal.LinearIndex(src_base), 0, 1);
  } else if (!ShapeUtil::HasZeroElements(dest_shape) &&
             !ShapeUtil::HasZeroElements(src_shape)) {
    // Perform copy if neither src literal nor dest literal has dimensions with
    // zero element, otherwise it's a no-op.
    TF_RET_CHECK(src_base.size() == dest_base.size());
    TF_RET_CHECK(src_base.size() == copy_size.size());

    // Scan the source from minor, stepping in copy size blocks, then within
    // the index enumaration functor, do a strided copy advancing source index
    // by one (walking through the minor dimension), and destination index by
    // proper stride size at the matching dimension.
    DimensionVector src_indexes(src_base.size(), 0);
    DimensionVector dest_indexes(dest_base.size(), 0);
    StrideConfig stride_config(src_shape, dest_shape, copy_size);

    auto copy_proc = [&](const std::vector<int64>& indexes) {
      // Map from multi-dimensional index, to source index.
      std::transform(indexes.begin(), indexes.end(), src_base.begin(),
                     src_indexes.begin(), std::plus<int64>());
      // Map from multi-dimensional index, to destination index.
      std::transform(indexes.begin(), indexes.end(), dest_base.begin(),
                     dest_indexes.begin(), std::plus<int64>());

      int64 src_index = src_literal.LinearIndex(src_indexes);
      int64 dest_index = LinearIndex(dest_indexes);

      StridedCopy(dest_data, dest_index, stride_config.dest_stride, src_data,
                  src_index, stride_config.source_stride,
                  stride_config.minor_loop_size);
      return true;
    };

    ShapeUtil::ForEachIndex(src_shape, stride_config.base,
                            stride_config.dimensions, stride_config.step,
                            copy_proc);
  }
  return Status::OK();
}

Status Literal::Copy(const Literal& src_literal,
                     tensorflow::gtl::ArraySlice<int64> src_base,
                     tensorflow::gtl::ArraySlice<int64> dest_base,
                     tensorflow::gtl::ArraySlice<int64> copy_size) {
  TF_RET_CHECK(ShapeUtil::SameElementType(src_literal.shape(), shape()));
  switch (src_literal.shape().element_type()) {
    case U8:
      return CopyRange<uint8>(src_literal, src_base, dest_base, copy_size);
    case U16:
      return CopyRange<uint16>(src_literal, src_base, dest_base, copy_size);
    case U32:
      return CopyRange<uint32>(src_literal, src_base, dest_base, copy_size);
    case U64:
      return CopyRange<uint64>(src_literal, src_base, dest_base, copy_size);
    case S8:
      return CopyRange<int8>(src_literal, src_base, dest_base, copy_size);
    case S16:
      return CopyRange<int16>(src_literal, src_base, dest_base, copy_size);
    case S32:
      return CopyRange<int32>(src_literal, src_base, dest_base, copy_size);
    case S64:
      return CopyRange<int64>(src_literal, src_base, dest_base, copy_size);
    case F16:
      return CopyRange<half>(src_literal, src_base, dest_base, copy_size);
    case BF16:
      return CopyRange<bfloat16>(src_literal, src_base, dest_base, copy_size);
    case F32:
      return CopyRange<float>(src_literal, src_base, dest_base, copy_size);
    case F64:
      return CopyRange<double>(src_literal, src_base, dest_base, copy_size);
    case C64:
      return CopyRange<complex64>(src_literal, src_base, dest_base, copy_size);
    case PRED:
      return CopyRange<bool>(src_literal, src_base, dest_base, copy_size);
    default:
      break;
  }
  return Unimplemented("Unhandled primitive type %d",
                       src_literal.shape().element_type());
}

/* static */ Literal Literal::Zero(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *Literal::CreateR0<uint8>(0);
    case U32:
      return *Literal::CreateR0<uint32>(0);
    case U64:
      return *Literal::CreateR0<uint64>(0);
    case S8:
      return *Literal::CreateR0<int8>(0);
    case S32:
      return *Literal::CreateR0<int32>(0);
    case S64:
      return *Literal::CreateR0<int64>(0);
    case F16:
      return *Literal::CreateR0<half>(static_cast<half>(0.0f));
    case BF16:
      return *Literal::CreateR0<bfloat16>(static_cast<bfloat16>(0.0f));
    case F32:
      return *Literal::CreateR0<float>(0);
    case F64:
      return *Literal::CreateR0<double>(0);
    case C64:
      return *Literal::CreateR0<complex64>(0);
    case PRED:
      return *Literal::CreateR0<bool>(false);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type cannot take on value of 0";
    case OPAQUE:
      LOG(FATAL) << "opaque element type cannot take on value of 0";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal Literal::One(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *Literal::CreateR0<uint8>(1);
    case U32:
      return *Literal::CreateR0<uint32>(1);
    case U64:
      return *Literal::CreateR0<uint64>(1);
    case S8:
      return *Literal::CreateR0<int8>(1);
    case S32:
      return *Literal::CreateR0<int32>(1);
    case S64:
      return *Literal::CreateR0<int64>(1);
    case F16:
      return *Literal::CreateR0<half>(static_cast<half>(1.0f));
    case BF16:
      return *Literal::CreateR0<bfloat16>(static_cast<bfloat16>(1.0f));
    case F32:
      return *Literal::CreateR0<float>(1);
    case F64:
      return *Literal::CreateR0<double>(1);
    case C64:
      return *Literal::CreateR0<complex64>(1);
    case PRED:
      return *Literal::CreateR0<bool>(true);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type cannot take on value of 1";
    case OPAQUE:
      LOG(FATAL) << "opaque element type cannot take on value of 1";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal Literal::MinValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *Literal::CreateR0<uint8>(std::numeric_limits<uint8>::min());
    case U32:
      return *Literal::CreateR0<uint32>(std::numeric_limits<uint32>::min());
    case U64:
      return *Literal::CreateR0<uint64>(std::numeric_limits<uint64>::min());
    case S8:
      return *Literal::CreateR0<int8>(std::numeric_limits<int8>::min());
    case S32:
      return *Literal::CreateR0<int32>(std::numeric_limits<int32>::min());
    case S64:
      return *Literal::CreateR0<int64>(std::numeric_limits<int64>::min());
    case F32:
      return *Literal::CreateR0<float>(-std::numeric_limits<float>::infinity());
    case F64:
      return *Literal::CreateR0<double>(
          -std::numeric_limits<double>::infinity());
    case C64:
      LOG(FATAL) << "C64 element type has no minimum value";
    case PRED:
      return *Literal::CreateR0<bool>(false);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      return *Literal::CreateR0<half>(
          static_cast<half>(-std::numeric_limits<float>::infinity()));
    case BF16:
      return *Literal::CreateR0<bfloat16>(
          static_cast<bfloat16>(-std::numeric_limits<float>::infinity()));
    case TUPLE:
      LOG(FATAL) << "tuple element type has no minimum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no minimum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal Literal::MaxValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *Literal::CreateR0<uint8>(std::numeric_limits<uint8>::max());
    case U32:
      return *Literal::CreateR0<uint32>(std::numeric_limits<uint32>::max());
    case U64:
      return *Literal::CreateR0<uint64>(std::numeric_limits<uint64>::max());
    case S8:
      return *Literal::CreateR0<int8>(std::numeric_limits<int8>::max());
    case S32:
      return *Literal::CreateR0<int32>(std::numeric_limits<int32>::max());
    case S64:
      return *Literal::CreateR0<int64>(std::numeric_limits<int64>::max());
    case F32:
      return *Literal::CreateR0<float>(std::numeric_limits<float>::infinity());
    case F64:
      return *Literal::CreateR0<double>(
          std::numeric_limits<double>::infinity());
    case PRED:
      return *Literal::CreateR0<bool>(true);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      return *Literal::CreateR0<half>(
          static_cast<half>(std::numeric_limits<float>::infinity()));
    case BF16:
      return *Literal::CreateR0<bfloat16>(
          static_cast<bfloat16>(std::numeric_limits<float>::infinity()));
    case TUPLE:
      LOG(FATAL) << "tuple element type has no maximum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no maximum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ std::unique_ptr<Literal> Literal::CreateR1(
    const tensorflow::core::Bitmap& values) {
  auto literal = MakeUnique<Literal>();
  literal->PopulateR1(values);
  return literal;
}

/* static */ std::unique_ptr<Literal> Literal::CreateR1U8(
    tensorflow::StringPiece value) {
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(U8, {static_cast<int64>(value.size())});
  literal->set_u8s(tensorflow::StringPiece(value.ToString()));
  return literal;
}

/* static */ std::unique_ptr<Literal> Literal::CreateR2F32Linspace(float from,
                                                                   float to,
                                                                   int64 rows,
                                                                   int64 cols) {
  auto value = MakeLinspaceArray2D(from, to, rows, cols);
  return CreateR2FromArray2D(*value);
}

std::unique_ptr<Literal> Literal::Relayout(
    const Layout& new_layout, const ShapeIndex& shape_index) const {
  std::unique_ptr<Literal> outer_result = CloneToUnique();

  const Literal* copy_from = this;
  Literal* copy_to = outer_result.get();
  for (int64 i = 0; i < shape_index.size(); i++) {
    *ShapeUtil::GetMutableSubshape(copy_to->mutable_shape(), {shape_index, i})
         ->mutable_layout() = new_layout;
    copy_from = &copy_from->tuple_literals_[shape_index[i]];
    copy_to = &copy_to->tuple_literals_[shape_index[i]];
  }

  DimensionVector base(ShapeUtil::Rank(copy_from->shape()), 0);
  DimensionVector copy_size(copy_from->shape().dimensions().begin(),
                            copy_from->shape().dimensions().end());

  CHECK(ShapeUtil::IsArray(copy_from->shape()));
  CHECK(ShapeUtil::IsArray(copy_to->shape()));
  *copy_to->mutable_shape()->mutable_layout() = new_layout;
  TF_CHECK_OK(copy_to->Copy(*copy_from, base, base, copy_size));
  return outer_result;
}

std::unique_ptr<Literal> Literal::Relayout(
    const Shape& shape_with_layout) const {
  CHECK(ShapeUtil::Compatible(shape_with_layout, shape()))
      << "Given shape_with_layout " << ShapeUtil::HumanString(shape_with_layout)
      << " not compatible with literal shape "
      << ShapeUtil::HumanString(shape());
  std::unique_ptr<Literal> result = CreateFromShape(shape_with_layout);
  ShapeUtil::ForEachSubshape(
      result->shape(),
      [this, &result](const Shape& subshape, const ShapeIndex& index) {
        if (ShapeUtil::IsArray(subshape)) {
          DimensionVector base(ShapeUtil::Rank(subshape), 0);
          DimensionVector copy_size(subshape.dimensions().begin(),
                                    subshape.dimensions().end());
          TF_CHECK_OK(result->GetSubliteral(index).Copy(GetSubliteral(index),
                                                        base, base, copy_size));
        }
      });
  return result;
}

StatusOr<std::unique_ptr<Literal>> Literal::Reshape(
    tensorflow::gtl::ArraySlice<int64> dimensions) const {
  if (ShapeUtil::IsTuple(shape())) {
    return InvalidArgument("Reshape does not support tuples.");
  }
  std::unique_ptr<Literal> output;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape().layout())) {
    output =
        Relayout(LayoutUtil::GetDefaultLayoutForRank(ShapeUtil::Rank(shape())));
  } else {
    output = CloneToUnique();
  }
  // Because the layout is monotonic, we can simply reuse the same sequence of
  // values without changing their order.
  *output->mutable_shape() =
      ShapeUtil::MakeShape(shape().element_type(), dimensions);

  int64 elements_before = ShapeUtil::ElementsIn(shape());
  int64 elements_after = ShapeUtil::ElementsIn(output->shape());
  if (elements_before != elements_after) {
    return InvalidArgument(
        "Shapes before and after Literal::Reshape have different numbers "
        "of elements: %s vs %s.",
        ShapeUtil::HumanString(shape()).c_str(),
        ShapeUtil::HumanString(output->shape()).c_str());
  }
  return std::move(output);
}

std::unique_ptr<Literal> Literal::Transpose(
    tensorflow::gtl::ArraySlice<int64> permutation) const {
  CHECK(!ShapeUtil::IsTuple(shape())) << "Tuple is not supported for transpose";
  CHECK(IsPermutation(permutation, ShapeUtil::Rank(shape())))
      << "Given permutation is not a permutation of dimension numbers";
  // To transpose the array, we just permute the dimensions and layout, and
  // do a straight memory copy of the raw data set.
  // This is considerably faster than iterating over every array element using
  // the EachCell<>() and Set<>() APIs.
  std::vector<int64> inverse_permutation = InversePermutation(permutation);
  Shape permuted_shape =
      ShapeUtil::PermuteDimensions(inverse_permutation, shape());
  // Replace the layout with one affine to this shape, such that a
  // transpose operation can be performed by leaving the flat values
  // representation intact.
  // For example, consider the shape F32[11,8]{1,0} under a {1,0} permutation.
  // The shape with affine layout resulting from that operation will be
  // F32[8,11]{0,1}, since it leaves the original most minor (the 8 sized), the
  // most minor.
  //
  // Essentially, given MinMaj(Di) the position of the Di dimension within the
  // minor to major vector, and given T(Di) the index that the original Di
  // dimension has within the transposed array, a layout is affine if
  // MinMaj(Di) == TMinMaj(T(Di)), with TMinMaj() being the minor to major
  // vector of the affine layout.
  CHECK(LayoutUtil::IsDense(permuted_shape));
  Layout* layout = permuted_shape.mutable_layout();
  layout->clear_minor_to_major();
  for (auto index : LayoutUtil::MinorToMajor(shape())) {
    layout->add_minor_to_major(inverse_permutation[index]);
  }
  std::unique_ptr<Literal> new_literal = CreateFromShape(permuted_shape);
  DCHECK_GE(ShapeUtil::ByteSizeOf(new_literal->shape()),
            ShapeUtil::ByteSizeOf(shape()));
  std::memcpy(new_literal->MutableInternalData(), InternalData(),
              ShapeUtil::ByteSizeOf(shape()));
  return new_literal;
}

std::unique_ptr<Literal> Literal::Slice(
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices) const {
  CHECK(!ShapeUtil::IsTuple(shape())) << "tuple is not supported for reshape";

  DimensionVector result_dimensions;
  for (int64 dnum = 0; dnum < ShapeUtil::Rank(shape()); ++dnum) {
    CHECK_GE(start_indices[dnum], 0);
    CHECK_LE(limit_indices[dnum], shape().dimensions(dnum));
    int64 dimension = limit_indices[dnum] - start_indices[dnum];
    CHECK_GT(dimension, 0);
    result_dimensions.push_back(dimension);
  }
  const auto result_shape =
      ShapeUtil::MakeShapeWithLayout(shape().element_type(), result_dimensions,
                                     LayoutUtil::MinorToMajor(shape()));

  auto result_literal = MakeUnique<Literal>();
  *result_literal->mutable_shape() = result_shape;
  result_literal->Reserve(ShapeUtil::ElementsIn(result_shape));

  DimensionVector new_indices(ShapeUtil::Rank(result_shape));
  switch (result_shape.element_type()) {
    case F32:
      result_literal->EachCell<float>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, float /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            float value = Get<float>(new_indices);
            result_literal->Set<float>(indices, value);
          });
      return result_literal;
    case S32:
      result_literal->EachCell<int32>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, int32 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            int32 value = Get<int32>(new_indices);
            result_literal->Set<int32>(indices, value);
          });
      return result_literal;
    case U32:
      result_literal->EachCell<uint32>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, uint32 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            uint32 value = Get<uint32>(new_indices);
            result_literal->Set<uint32>(indices, value);
          });
      return result_literal;
    default:
      LOG(FATAL) << "not yet implemented: "
                 << PrimitiveType_Name(result_shape.element_type());
  }
}

std::unique_ptr<Literal> Literal::CloneToUnique() const {
  auto unique = MakeUnique<Literal>();
  *unique = *this;
  return unique;
}

string Literal::GetAsString(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  switch (shape().element_type()) {
    case PRED:
      return Get<bool>(multi_index) ? "true" : "false";
    case U8:
      return tensorflow::strings::StrCat(Get<uint8>(multi_index));
    case S32:
      return tensorflow::strings::StrCat(Get<int32>(multi_index));
    case S64:
      return tensorflow::strings::StrCat(Get<int64>(multi_index));
    case U32:
      return tensorflow::strings::StrCat(Get<uint32>(multi_index));
    case U64:
      return tensorflow::strings::StrCat(Get<uint64>(multi_index));
    case F32:
      return tensorflow::strings::StrCat(Get<float>(multi_index));
    case F64:
      return tensorflow::strings::StrCat(Get<double>(multi_index));
    case C64: {
      complex64 c = Get<complex64>(multi_index);
      return tensorflow::strings::StrCat("(", c.real(), ", ", c.imag(), ")");
    }
    case F16:
      return tensorflow::strings::StrCat(Get<half>(multi_index));
    case BF16:
      return tensorflow::strings::StrCat(
          static_cast<float>(Get<bfloat16>(multi_index)));
    default:
      return tensorflow::strings::StrCat(
          "[", PrimitiveType_Name(shape().element_type()), "]");
  }
}

StatusOr<int64> Literal::GetIntegralAsS64(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  switch (shape().element_type()) {
    case PRED:
      return Get<bool>(multi_index);
    case U8:
      return Get<uint8>(multi_index);
    case S32:
      return Get<int32>(multi_index);
    case S64:
      return Get<int64>(multi_index);
    case U32:
      return Get<uint32>(multi_index);
    case U64:
      return Get<uint64>(multi_index);
    default:
      return FailedPrecondition(
          "Array element type is not integral: %s",
          PrimitiveType_Name(shape().element_type()).c_str());
  }
}

int64 Literal::LinearIndex(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  return IndexUtil::MultidimensionalIndexToLinearIndex(shape(), multi_index);
}

string Literal::ToString(bool print_layout) const {
  std::vector<string> pieces;

  auto shape_to_string = [print_layout](const Shape& shape) {
    if (print_layout) {
      return ShapeUtil::HumanStringWithLayout(shape);
    } else {
      return ShapeUtil::HumanString(shape);
    }
  };

  auto element_to_string =
      [this](tensorflow::gtl::ArraySlice<int64> indices) -> string {
    PrimitiveType element_type = shape().element_type();
    if (element_type == PRED) {
      // We display predicates in a densely packed form.
      return Get<bool>(indices) ? "1" : "0";
    }
    return ((!indices.empty() && indices.back() > 0) ? ", " : "") +
           GetAsString(indices);
  };

  // TODO(b/32894291): refactor this code to reduce code duplication.
  if (ShapeUtil::IsTuple(shape())) {
    pieces.push_back(shape_to_string(shape()));
    pieces.push_back(" (\n");
    pieces.push_back(tensorflow::str_util::Join(
        tuple_literals(), ",\n", [](string* out, const Literal& element) {
          tensorflow::strings::StrAppend(out, element.ToString());
        }));
    pieces.push_back("\n)");
  } else if (ShapeUtil::Rank(shape()) == 0) {
    pieces.push_back(GetAsString({}));
  } else if (ShapeUtil::Rank(shape()) == 1) {
    pieces.push_back("{");
    for (int64 i0 = 0; i0 < shape().dimensions(0); ++i0) {
      pieces.push_back(element_to_string({i0}));
    }
    pieces.push_back("}");
  } else if (ShapeUtil::Rank(shape()) == 2) {
    pieces.push_back(shape_to_string(shape()));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape().dimensions(0); ++i0) {
      pieces.push_back("  { ");
      for (int64 i1 = 0; i1 < shape().dimensions(1); ++i1) {
        pieces.push_back(element_to_string({i0, i1}));
      }
      pieces.push_back(" ");
      pieces.push_back(i0 == shape().dimensions(0) - 1 ? "}\n" : "},\n");
    }
    pieces.push_back("}");
  } else if (ShapeUtil::Rank(shape()) == 3) {
    pieces.push_back(shape_to_string(shape()));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape().dimensions(0); ++i0) {
      pieces.push_back(i0 > 0 ? ",\n{" : "{");
      for (int64 i1 = 0; i1 < shape().dimensions(1); ++i1) {
        pieces.push_back(i1 > 0 ? ",\n  { " : " { ");
        for (int64 i2 = 0; i2 < shape().dimensions(2); ++i2) {
          pieces.push_back(element_to_string({i0, i1, i2}));
        }
        pieces.push_back(" }");
      }
      pieces.push_back(" }");
    }
    pieces.push_back("\n}");
  } else if (ShapeUtil::Rank(shape()) == 4) {
    pieces.push_back(shape_to_string(shape()));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape().dimensions(0); ++i0) {
      pieces.push_back(tensorflow::strings::Printf("  {  /*i0=%lld*/\n", i0));
      for (int64 i1 = 0; i1 < shape().dimensions(1); ++i1) {
        pieces.push_back(
            tensorflow::strings::Printf("    {  /*i1=%lld*/\n", i1));
        for (int64 i2 = 0; i2 < shape().dimensions(2); ++i2) {
          pieces.push_back("      {");
          for (int64 i3 = 0; i3 < shape().dimensions(3); ++i3) {
            pieces.push_back(element_to_string({i0, i1, i2, i3}));
          }
          pieces.push_back(i2 == shape().dimensions(2) - 1 ? "}\n" : "},\n");
        }
        pieces.push_back(i1 == shape().dimensions(1) - 1 ? "    }\n"
                                                         : "    },\n");
      }
      pieces.push_back(i0 == shape().dimensions(0) - 1 ? "  }\n" : "  },\n");
    }
    pieces.push_back("}");
  } else if (ShapeUtil::Rank(shape()) == 5) {
    pieces.push_back(shape_to_string(shape()));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape().dimensions(0); ++i0) {
      pieces.push_back(tensorflow::strings::Printf("  {  /*i0=%lld*/\n", i0));
      for (int64 i1 = 0; i1 < shape().dimensions(1); ++i1) {
        pieces.push_back(
            tensorflow::strings::Printf("    {  /*i1=%lld*/\n", i1));
        for (int64 i2 = 0; i2 < shape().dimensions(2); ++i2) {
          pieces.push_back(
              tensorflow::strings::Printf("      {  /*i2=%lld*/\n", i2));
          for (int64 i3 = 0; i3 < shape().dimensions(3); ++i3) {
            pieces.push_back("        {");
            for (int64 i4 = 0; i4 < shape().dimensions(4); ++i4) {
              pieces.push_back(element_to_string({i0, i1, i2, i3, i4}));
            }
            pieces.push_back(i3 == shape().dimensions(3) - 1 ? "}\n" : "},\n");
          }
          pieces.push_back(i2 == shape().dimensions(2) - 1 ? "      }\n"
                                                           : "      },\n");
        }
        pieces.push_back(i1 == shape().dimensions(1) - 1 ? "    }\n"
                                                         : "    },\n");
      }
      pieces.push_back(i0 == shape().dimensions(0) - 1 ? "  }\n" : "  },\n");
    }
    pieces.push_back("}");
  } else {
    pieces.push_back(shape_to_string(shape()));
    pieces.push_back(" {");
    EachCellAsString(
        [&](tensorflow::gtl::ArraySlice<int64> indices, const string& value) {
          pieces.push_back(" ");
          pieces.push_back(value);
        });
    pieces.push_back("}");
  }

  return tensorflow::str_util::Join(pieces, "");
}

/* static */ std::unique_ptr<Literal> Literal::MakeTuple(
    tensorflow::gtl::ArraySlice<const Literal*> elements) {
  auto literal = MakeUnique<Literal>();
  std::vector<Shape> shape;
  for (const Literal* tuple_element : elements) {
    *literal->add_tuple_literals() = *tuple_element;
    shape.push_back(tuple_element->shape());
  }
  *literal->mutable_shape() = ShapeUtil::MakeTupleShape(shape);
  return literal;
}

/* static */ std::unique_ptr<Literal> Literal::MakeTupleOwned(
    std::vector<std::unique_ptr<Literal>> elements) {
  auto literal = MakeUnique<Literal>();
  std::vector<Shape> shape;
  for (auto& tuple_element : elements) {
    shape.push_back(tuple_element->shape());
    *literal->add_tuple_literals() = std::move(*tuple_element);
  }
  *literal->mutable_shape() = ShapeUtil::MakeTupleShape(shape);
  return literal;
}

const void* Literal::InternalData() const {
  return const_cast<const void*>(
      const_cast<Literal*>(this)->MutableInternalData());
}

void* Literal::MutableInternalData() {
  // NOTE: We access the vectors directly to avoid the const reference
  // created by the accessor functions.
  switch (shape().element_type()) {
    case PRED:
    case U8:
      return reinterpret_cast<void*>(u8s_.data());
    case S32:
      return reinterpret_cast<void*>(s32s_.data());
    case S64:
      return reinterpret_cast<void*>(s64s_.data());
    case U32:
      return reinterpret_cast<void*>(u32s_.data());
    case U64:
      return reinterpret_cast<void*>(u64s_.data());
    case F32:
      return reinterpret_cast<void*>(f32s_.data());
    case F64:
      return reinterpret_cast<void*>(f64s_.data());
    case C64:
      return reinterpret_cast<void*>(c64s_.data());
    case F16:
      return reinterpret_cast<void*>(f16s_.data());
    case BF16:
      return reinterpret_cast<void*>(bf16s_.data());
    default:
      LOG(FATAL) << "primitive type not supported in literals: "
                 << PrimitiveType_Name(shape().element_type());
  }
}

void Literal::Reserve(int64 num_elements) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  switch (shape().element_type()) {
    case PRED:
      Resize<bool>(num_elements, false);
      break;
    case S8:
      Resize<int8>(num_elements, 0);
      break;
    case U8:
      Resize<uint8>(num_elements, 0);
      break;
    case S32:
      Resize<int32>(num_elements, 0);
      break;
    case S64:
      Resize<int64>(num_elements, 0);
      break;
    case U32:
      Resize<uint32>(num_elements, 0);
      break;
    case U64:
      Resize<uint64>(num_elements, 0);
      break;
    case F32:
      Resize<float>(num_elements, 0);
      break;
    case F64:
      Resize<double>(num_elements, 0);
      break;
    case C64:
      Resize<complex64>(num_elements, 0);
      break;
    case F16:
      Resize<half>(num_elements, static_cast<half>(0.0f));
      break;
    case BF16:
      Resize<bfloat16>(num_elements, static_cast<bfloat16>(0.0f));
      break;
    default:
      LOG(FATAL) << "primitive type not supported in literals: "
                 << PrimitiveType_Name(shape().element_type());
  }
}

tensorflow::Status Literal::ValidateLiteral() const {
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape()));
  int64 expected = ShapeUtil::ElementsIn(shape());
  int64 actual = -1;
  switch (shape().element_type()) {
    case PRED:
    case U8:
      actual = u8s_size();
      break;
    case S32:
      actual = s32s_size();
      break;
    case U32:
      actual = u32s_size();
      break;
    case S64:
      actual = s64s_size();
      break;
    case U64:
      actual = u64s_size();
      break;
    case F32:
      actual = f32s_size();
      break;
    case F64:
      actual = f64s_size();
      break;
    case C64:
      actual = c64s_size();
      break;
    case F16:
      actual = f16s().size() / sizeof(half);
      break;
    case BF16:
      actual = bf16s().size();
      break;
    default:
      return tensorflow::errors::Unimplemented(
          "unhandled element type for literal validation: " +
          PrimitiveType_Name(shape().element_type()));
  }

  if (expected != actual) {
    return tensorflow::errors::InvalidArgument(tensorflow::strings::Printf(
        "literal has bad number of elements for its shape %s: want %lld "
        "got %lld",
        ShapeUtil::HumanString(shape()).c_str(), expected, actual));
  }

  return tensorflow::Status::OK();
}

void Literal::EachCellAsString(
    const std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                             const string& value)>& per_cell) const {
  if (ShapeUtil::HasZeroElements(shape())) {
    return;
  }
  std::vector<int64> indices = IndexUtil::LinearIndexToMultidimensionalIndex(
      shape(), /*linear_index=*/0);
  do {
    per_cell(indices, GetAsString(indices));
  } while (IndexUtil::BumpIndices(shape(), &indices));
}

namespace {
template <typename NativeSrcT, typename NativeDestT>
std::unique_ptr<Literal> ConvertBetweenNativeTypes(const Literal& src_literal) {
  auto result_literal = MakeUnique<Literal>();
  Shape* result_shape = result_literal->mutable_shape();
  *result_shape = src_literal.shape();
  result_shape->set_element_type(
      primitive_util::NativeToPrimitiveType<NativeDestT>());
  result_literal->Reserve(ShapeUtil::ElementsIn(*result_shape));
  tensorflow::gtl::ArraySlice<NativeSrcT> src_data =
      src_literal.GetArraySlice<NativeSrcT>();
  tensorflow::gtl::MutableArraySlice<NativeDestT> dest_data =
      result_literal->GetMutableArraySlice<NativeDestT>();
  int64 num_elements = ShapeUtil::ElementsIn(src_literal.shape());

  for (int64 i = 0; i < num_elements; ++i) {
    dest_data[i] = static_cast<NativeDestT>(src_data[i]);
  }
  return result_literal;
}

template <PrimitiveType primitive_src_type>
std::unique_ptr<Literal> ConvertToC64(const Literal& src_literal) {
  auto result_literal = MakeUnique<Literal>();
  Shape* result_shape = result_literal->mutable_shape();
  *result_shape = src_literal.shape();
  result_shape->set_element_type(C64);
  result_literal->Reserve(ShapeUtil::ElementsIn(*result_shape));
  using NativeSrcT =
      typename primitive_util::PrimitiveTypeToNative<primitive_src_type>::type;
  tensorflow::gtl::ArraySlice<NativeSrcT> src_data =
      src_literal.GetArraySlice<NativeSrcT>();
  tensorflow::gtl::MutableArraySlice<complex64> dest_data =
      result_literal->GetMutableArraySlice<complex64>();
  int64 num_elements = ShapeUtil::ElementsIn(src_literal.shape());
  for (int64 i = 0; i < num_elements; ++i) {
    dest_data[i] = complex64(static_cast<float>(src_data[i]), 0);
  }
  return result_literal;
}

template <PrimitiveType primitive_src_type, PrimitiveType primitive_dest_type>
std::unique_ptr<Literal> ConvertIfTypesMatch(const Literal& src_literal) {
  CHECK_EQ(primitive_src_type, src_literal.shape().element_type());
  return ConvertBetweenNativeTypes<
      typename primitive_util::PrimitiveTypeToNative<primitive_src_type>::type,
      typename primitive_util::PrimitiveTypeToNative<
          primitive_dest_type>::type>(src_literal);
}

template <PrimitiveType primitive_src_type>
StatusOr<std::unique_ptr<Literal>> ConvertIfDestTypeMatches(
    const Literal& src_literal, PrimitiveType primitive_dest_type) {
  switch (primitive_dest_type) {
#define CONVERT_IF_TYPES_MATCH(type) \
  case (type):                       \
    return ConvertIfTypesMatch<primitive_src_type, (type)>(src_literal);
    CONVERT_IF_TYPES_MATCH(PRED)
    CONVERT_IF_TYPES_MATCH(S8)
    CONVERT_IF_TYPES_MATCH(S32)
    CONVERT_IF_TYPES_MATCH(S64)
    CONVERT_IF_TYPES_MATCH(U8)
    CONVERT_IF_TYPES_MATCH(U32)
    CONVERT_IF_TYPES_MATCH(U64)
    CONVERT_IF_TYPES_MATCH(F16)
    CONVERT_IF_TYPES_MATCH(F32)
    CONVERT_IF_TYPES_MATCH(F64)
    CONVERT_IF_TYPES_MATCH(BF16)
#undef CONVERT_IF_TYPES_MATCH
    case C64:
      return ConvertToC64<primitive_src_type>(src_literal);
    // Other types are not yet supported.
    default:
      return InvalidArgument(
          "Unimplemented: Convert from type %s to type %s",
          PrimitiveType_Name(src_literal.shape().element_type()).c_str(),
          PrimitiveType_Name(primitive_dest_type).c_str());
  }
}
}  // namespace

StatusOr<std::unique_ptr<Literal>> Literal::Convert(
    PrimitiveType primitive_dest_type) const {
  switch (shape().element_type()) {
#define CONVERT_IF_DEST_TYPE_MATCHES(type) \
  case (type):                             \
    return ConvertIfDestTypeMatches<(type)>(*this, primitive_dest_type);
    CONVERT_IF_DEST_TYPE_MATCHES(PRED)
    CONVERT_IF_DEST_TYPE_MATCHES(S8)
    CONVERT_IF_DEST_TYPE_MATCHES(S32)
    CONVERT_IF_DEST_TYPE_MATCHES(S64)
    CONVERT_IF_DEST_TYPE_MATCHES(U8)
    CONVERT_IF_DEST_TYPE_MATCHES(U32)
    CONVERT_IF_DEST_TYPE_MATCHES(U64)
    CONVERT_IF_DEST_TYPE_MATCHES(F16)
    CONVERT_IF_DEST_TYPE_MATCHES(F32)
    CONVERT_IF_DEST_TYPE_MATCHES(F64)
    CONVERT_IF_DEST_TYPE_MATCHES(BF16)
#undef CONVERT_IF_DEST_TYPE_MATCHES
      // Other types are not yet supported.
    default:
      return InvalidArgument("Unimplemented: Convert from type %s to type %s",
                             PrimitiveType_Name(shape().element_type()).c_str(),
                             PrimitiveType_Name(primitive_dest_type).c_str());
  }
}

namespace {

// Helper function which compares whether the elements of literal1 are equal to
// the elements of literal2. Recursively iterates through the entire
// multidimensional index space and compares the literal elements
// one-by-one. literal1 and literal2 must be compatible (same dimensions and
// type).
template <typename NativeT>
bool EqualElements(const Literal& literal1, const Literal& literal2,
                   int dimension, std::vector<int64>* multi_index) {
  if (dimension == ShapeUtil::Rank(literal1.shape())) {
    return (literal1.Get<NativeT>(*multi_index) ==
            literal2.Get<NativeT>(*multi_index));
  }
  for (int64 i = 0; i < literal1.shape().dimensions(dimension); ++i) {
    (*multi_index)[dimension] = i;
    if (!EqualElements<NativeT>(literal1, literal2, dimension + 1,
                                multi_index)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool Literal::operator==(const Literal& other) const {
  if (!ShapeUtil::Compatible(shape(), other.shape())) {
    return false;
  }
  if (ShapeUtil::IsTuple(shape())) {
    // Because the shapes are compatible, they must have the same number of
    // tuple elements.
    CHECK_EQ(tuple_literals_size(), other.tuple_literals_size());
    for (int i = 0; i < tuple_literals_size(); ++i) {
      if (tuple_literals(i) != other.tuple_literals(i)) {
        return false;
      }
    }
    return true;
  } else {
    std::vector<int64> multi_index(ShapeUtil::Rank(shape()), 0);
    switch (shape().element_type()) {
      case PRED:
        return EqualElements<bool>(*this, other, 0, &multi_index);
      case U8:
        return EqualElements<uint8>(*this, other, 0, &multi_index);
      case S32:
        return EqualElements<int32>(*this, other, 0, &multi_index);
      case S64:
        return EqualElements<int64>(*this, other, 0, &multi_index);
      case U32:
        return EqualElements<uint32>(*this, other, 0, &multi_index);
      case U64:
        return EqualElements<uint64>(*this, other, 0, &multi_index);
      case F32:
        return EqualElements<float>(*this, other, 0, &multi_index);
      case F64:
        return EqualElements<double>(*this, other, 0, &multi_index);
      case F16:
        return EqualElements<half>(*this, other, 0, &multi_index);
      case BF16:
        return EqualElements<bfloat16>(*this, other, 0, &multi_index);
      case C64:
        return EqualElements<complex64>(*this, other, 0, &multi_index);
      default:
        LOG(FATAL) << "Unimplemented: Literal::Equal for type "
                   << PrimitiveType_Name(shape().element_type());
    }
  }
}

template <>
tensorflow::gtl::MutableArraySlice<bool> Literal::GetMutableArraySlice() {
  auto values = mutable_preds();
  return tensorflow::gtl::MutableArraySlice<bool>(
      reinterpret_cast<bool*>(values->data()), values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<int8> Literal::GetMutableArraySlice() {
  auto values = mutable_u8s();
  return tensorflow::gtl::MutableArraySlice<int8>(
      reinterpret_cast<int8*>(values->data()), values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<uint8> Literal::GetMutableArraySlice() {
  auto values = mutable_u8s();
  return tensorflow::gtl::MutableArraySlice<uint8>(values->data(),
                                                   values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<int16> Literal::GetMutableArraySlice() {
  auto values = mutable_s16s();
  return tensorflow::gtl::MutableArraySlice<int16>(values->data(),
                                                   values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<uint16> Literal::GetMutableArraySlice() {
  auto values = mutable_u16s();
  return tensorflow::gtl::MutableArraySlice<uint16>(values->data(),
                                                    values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<int32> Literal::GetMutableArraySlice() {
  auto values = mutable_s32s();
  return tensorflow::gtl::MutableArraySlice<int32>(values->data(),
                                                   values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<uint32> Literal::GetMutableArraySlice() {
  auto values = mutable_u32s();
  return tensorflow::gtl::MutableArraySlice<uint32>(values->data(),
                                                    values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<int64> Literal::GetMutableArraySlice() {
  static_assert(sizeof(int64) == sizeof(tensorflow::protobuf_int64) &&
                    alignof(int64) == alignof(tensorflow::protobuf_int64),
                "The int64 and tensorflow::protobuf_int64 types are not "
                "compatible");
  auto values = mutable_s64s();
  // Because of the fact that tensorflow::protobuf_int64 is defined as int64_t
  // while tensorflow::int64 is defined as long long, a reinterpret_cast<> is
  // necessary from the raw data pointer returned by the mutable_data() API.
  return tensorflow::gtl::MutableArraySlice<int64>(
      reinterpret_cast<int64*>(values->data()), values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<uint64> Literal::GetMutableArraySlice() {
  static_assert(sizeof(uint64) == sizeof(tensorflow::protobuf_uint64) &&
                    alignof(uint64) == alignof(tensorflow::protobuf_uint64),
                "The uint64 and tensorflow::protobuf_uint64 types are not "
                "compatible");
  auto values = mutable_u64s();
  // Because of the fact that tensorflow::protobuf_uint64 is defined as uint64_t
  // while tensorflow::uint64 is defined as unsigned long long, a
  // reinterpret_cast<> is necessary from the raw data pointer returned by the
  // mutable_data() API.
  return tensorflow::gtl::MutableArraySlice<uint64>(
      reinterpret_cast<uint64*>(values->data()), values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<float> Literal::GetMutableArraySlice() {
  auto values = mutable_f32s();
  return tensorflow::gtl::MutableArraySlice<float>(values->data(),
                                                   values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<double> Literal::GetMutableArraySlice() {
  auto values = mutable_f64s();
  return tensorflow::gtl::MutableArraySlice<double>(values->data(),
                                                    values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<complex64> Literal::GetMutableArraySlice() {
  auto values = mutable_c64s();
  return {values->data(), values->size()};
}

template <>
tensorflow::gtl::MutableArraySlice<half> Literal::GetMutableArraySlice<half>() {
  auto values = mutable_f16s();
  return tensorflow::gtl::MutableArraySlice<half>(values->data(),
                                                  values->size());
}

template <>
tensorflow::gtl::MutableArraySlice<bfloat16>
Literal::GetMutableArraySlice<bfloat16>() {
  auto values = mutable_bf16s();
  return {values->data(), values->size()};
}

template <>
tensorflow::gtl::ArraySlice<bool> Literal::GetArraySlice<bool>() const {
  CHECK_EQ(shape().element_type(), PRED);
  return tensorflow::gtl::ArraySlice<bool>(
      reinterpret_cast<const bool*>(preds().data()), preds().size());
}

template <>
tensorflow::gtl::ArraySlice<uint8> Literal::GetArraySlice<uint8>() const {
  CHECK_EQ(shape().element_type(), U8);
  return tensorflow::gtl::ArraySlice<uint8>(
      reinterpret_cast<const uint8*>(u8s().data()), u8s().size());
}

template <>
tensorflow::gtl::ArraySlice<int8> Literal::GetArraySlice<int8>() const {
  CHECK_EQ(shape().element_type(), S8);
  return tensorflow::gtl::ArraySlice<int8>(
      reinterpret_cast<const int8*>(u8s().data()), u8s().size());
}

template <>
tensorflow::gtl::ArraySlice<uint16> Literal::GetArraySlice<uint16>() const {
  CHECK_EQ(shape().element_type(), U16);
  return tensorflow::gtl::ArraySlice<uint16>(u16s().data(), u16s().size());
}

template <>
tensorflow::gtl::ArraySlice<int16> Literal::GetArraySlice<int16>() const {
  CHECK_EQ(shape().element_type(), S16);
  return tensorflow::gtl::ArraySlice<int16>(s16s().data(), s16s().size());
}

template <>
tensorflow::gtl::ArraySlice<uint32> Literal::GetArraySlice<uint32>() const {
  CHECK_EQ(shape().element_type(), U32);
  return u32s();
}

template <>
tensorflow::gtl::ArraySlice<uint64> Literal::GetArraySlice<uint64>() const {
  CHECK_EQ(shape().element_type(), U64);
  return u64s();
}

template <>
tensorflow::gtl::ArraySlice<int32> Literal::GetArraySlice<int32>() const {
  CHECK_EQ(shape().element_type(), S32);
  return s32s();
}

template <>
tensorflow::gtl::ArraySlice<int64> Literal::GetArraySlice<int64>() const {
  CHECK_EQ(shape().element_type(), S64);
  return s64s();
}

template <>
tensorflow::gtl::ArraySlice<double> Literal::GetArraySlice<double>() const {
  CHECK_EQ(shape().element_type(), F64);
  return f64s();
}

template <>
tensorflow::gtl::ArraySlice<half> Literal::GetArraySlice<half>() const {
  CHECK_EQ(shape().element_type(), F16);
  return tensorflow::gtl::ArraySlice<half>(f16s().data(),
                                           f16s().size() / sizeof(half));
}

template <>
tensorflow::gtl::ArraySlice<bfloat16> Literal::GetArraySlice<bfloat16>() const {
  CHECK_EQ(shape().element_type(), BF16);
  return {bf16s().data(), bf16s().size()};
}

template <>
tensorflow::gtl::ArraySlice<complex64> Literal::GetArraySlice<complex64>()
    const {
  CHECK_EQ(shape().element_type(), C64);
  return c64s();
}

template <typename NativeT>
static bool AllElementsEqualValue(const Literal& literal, NativeT value) {
  for (int64 i = 0; i < ShapeUtil::ElementsIn(literal.shape()); ++i) {
    auto multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
    if (literal.Get<NativeT>(multi_index) != value) {
      return false;
    }
  }
  return true;
}

bool Literal::IsAll(int8 value) const {
  switch (shape().element_type()) {
    case U8:
      if (value >= 0) {
        return AllElementsEqualValue<uint8>(*this, value);
      }
      return false;
    case U32:
      if (value >= 0) {
        return AllElementsEqualValue<uint32>(*this, value);
      }
      return false;
    case U64:
      if (value >= 0) {
        return AllElementsEqualValue<uint64>(*this, value);
      }
      return false;
    case S8:
      return AllElementsEqualValue<int8>(*this, value);
    case S32:
      return AllElementsEqualValue<int32>(*this, value);
    case S64:
      return AllElementsEqualValue<int64>(*this, value);
    case F32:
      return AllElementsEqualValue<float>(*this, value);
    case F64:
      return AllElementsEqualValue<double>(*this, value);
    case F16:
      return AllElementsEqualValue<half>(*this, static_cast<half>(value));
    case BF16:
      return AllElementsEqualValue<bfloat16>(*this,
                                             static_cast<bfloat16>(value));
    case PRED:
      if (value == 0) {
        return AllElementsEqualValue<bool>(*this, false);
      }
      if (value == 1) {
        return AllElementsEqualValue<bool>(*this, true);
      }
      return false;
    default:
      return false;
  }
}

bool Literal::IsAllFloat(float value) const {
  switch (shape().element_type()) {
    case F32:
      return AllElementsEqualValue<float>(*this, value);
    case F64:
      return AllElementsEqualValue<double>(*this, value);
    case F16:
      return AllElementsEqualValue<half>(*this, static_cast<half>(value));
    case BF16:
      return AllElementsEqualValue<bfloat16>(*this,
                                             static_cast<bfloat16>(value));
    default:
      return false;
  }
}

bool Literal::IsAllComplex(complex64 value) const {
  switch (shape().element_type()) {
    case C64:
      return AllElementsEqualValue<complex64>(*this, value);
    default:
      return false;
  }
}

bool Literal::IsZero(tensorflow::gtl::ArraySlice<int64> indices) const {
  switch (shape().element_type()) {
    case U8:
      return Get<uint8>(indices) == 0;
    case U32:
      return Get<uint32>(indices) == 0;
    case U64:
      return Get<uint64>(indices) == 0;
    case S8:
      return Get<int8>(indices) == 0;
    case S32:
      return Get<int32>(indices) == 0;
    case S64:
      return Get<int64>(indices) == 0;
    case F32:
      return Get<float>(indices) == 0.0f;
    case F64:
      return Get<double>(indices) == 0.0;
    case C64:
      return Get<complex64>(indices) == complex64(0.0f, 0.0f);
    case F16:
      return Get<half>(indices) == static_cast<half>(0.0f);
    case BF16:
      return Get<bfloat16>(indices) == static_cast<bfloat16>(0.0f);
    case PRED:
      return Get<bool>(indices) == false;
    default:
      LOG(FATAL) << "Input literal must be an array.";
  }
}

template <>
/* static */ void Literal::Resize<bool>(int64 num_elements, bool value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_preds()->resize(num_elements, value);
}

template <>
void Literal::Resize<int8>(int64 num_elements, int8 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_u8s()->resize(num_elements, value);
}

template <>
void Literal::Resize<uint8>(int64 num_elements, uint8 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_u8s()->resize(num_elements, value);
}

template <>
void Literal::Resize<int32>(int64 num_elements, int32 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_s32s()->resize(num_elements, value);
}

template <>
void Literal::Resize<uint32>(int64 num_elements, uint32 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_u32s()->resize(num_elements, value);
}

template <>
void Literal::Resize<int64>(int64 num_elements, int64 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_s64s()->resize(num_elements, value);
}

template <>
void Literal::Resize<uint64>(int64 num_elements, uint64 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_u64s()->resize(num_elements, value);
}

template <>
void Literal::Resize<float>(int64 num_elements, float value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_f32s()->resize(num_elements, value);
}

template <>
void Literal::Resize<double>(int64 num_elements, double value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_f64s()->resize(num_elements, value);
}

template <>
void Literal::Resize<half>(int64 num_elements, half value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_f16s()->resize(num_elements, value);
}

template <>
void Literal::Resize<bfloat16>(int64 num_elements, bfloat16 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_bf16s()->resize(num_elements, value);
}

template <>
void Literal::Resize<complex64>(int64 num_elements, complex64 value) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), num_elements);
  mutable_c64s()->resize(num_elements, value);
}

template <typename RepeatedFieldT, typename NativeT>
void CopyToRepeatedField(RepeatedFieldT* dest,
                         const std::vector<NativeT>& src) {
  *dest = RepeatedFieldT(src.begin(), src.end());
}

template <>
void CopyToRepeatedField<tensorflow::protobuf::RepeatedField<float>, complex64>(
    tensorflow::protobuf::RepeatedField<float>* dest,
    const std::vector<complex64>& src) {
  *dest = tensorflow::protobuf::RepeatedField<float>(
      reinterpret_cast<const float*>(src.data()),
      reinterpret_cast<const float*>(src.data()) + src.size() * 2);
}

LiteralProto Literal::ToProto() const {
  LiteralProto proto;
  proto.Clear();
  *proto.mutable_shape() = shape();
  switch (shape().element_type()) {
    case PRED:
      CopyToRepeatedField(proto.mutable_preds(), preds());
      break;
    case U8:
      *proto.mutable_u8s() = u8s_string();
      break;
    case S32:
      CopyToRepeatedField(proto.mutable_s32s(), s32s());
      break;
    case S64:
      CopyToRepeatedField(proto.mutable_s64s(), s64s());
      break;
    case U32:
      CopyToRepeatedField(proto.mutable_u32s(), u32s());
      break;
    case U64:
      CopyToRepeatedField(proto.mutable_u64s(), u64s());
      break;
    case F16:
      *proto.mutable_f16s() =
          string(reinterpret_cast<const char*>(f16s_.data()),
                 f16s_.size() * sizeof(half));
      if (!kLittleEndian) {
        ConvertEndianShort(const_cast<char*>(proto.mutable_f16s()->data()),
                           proto.f16s().size());
      }
      break;
    case BF16:
      *proto.mutable_bf16s() =
          string(reinterpret_cast<const char*>(bf16s_.data()),
                 bf16s_.size() * sizeof(bfloat16));
      if (!kLittleEndian) {
        ConvertEndianShort(const_cast<char*>(proto.mutable_bf16s()->data()),
                           proto.bf16s().size());
      }
      break;
    case F32:
      CopyToRepeatedField(proto.mutable_f32s(), f32s());
      break;
    case F64:
      CopyToRepeatedField(proto.mutable_f64s(), f64s());
      break;
    case C64:
      CopyToRepeatedField(proto.mutable_c64s(), c64s());
      break;
    case TUPLE:
      for (const auto& tuple : tuple_literals()) {
        *proto.add_tuple_literals() = tuple.ToProto();
      }
      break;
    default:
      LOG(FATAL) << "Unhandled primitive type " << shape().element_type();
  }

  return proto;
}

template <typename RepeatedFieldT, typename NativeT>
void CopyFromRepeatedField(std::vector<NativeT>* dest,
                           const RepeatedFieldT& src) {
  *dest = std::vector<NativeT>(src.begin(), src.end());
}

template <>
void CopyFromRepeatedField<tensorflow::protobuf::RepeatedField<float>,
                           complex64>(
    std::vector<complex64>* dest,
    const tensorflow::protobuf::RepeatedField<float>& src) {
  *dest = std::vector<complex64>(
      reinterpret_cast<const complex64*>(src.data()),
      reinterpret_cast<const complex64*>(src.data()) + src.size() / 2);
}

void Literal::CopyFromProto(const LiteralProto& literal_proto) {
  if (!literal_proto.has_shape()) {
    return;
  }

  *mutable_shape() = literal_proto.shape();
  switch (shape().element_type()) {
    case PRED:
      CopyFromRepeatedField(mutable_preds(), literal_proto.preds());
      break;
    case U8:
      set_u8s(literal_proto.u8s());
      break;
    case S32:
      CopyFromRepeatedField(mutable_s32s(), literal_proto.s32s());
      break;
    case S64:
      CopyFromRepeatedField(mutable_s64s(), literal_proto.s64s());
      break;
    case U32:
      CopyFromRepeatedField(mutable_u32s(), literal_proto.u32s());
      break;
    case U64:
      CopyFromRepeatedField(mutable_u64s(), literal_proto.u64s());
      break;
    case F16: {
      const string& s(literal_proto.f16s());
      CHECK_EQ(0, s.size() % sizeof(half));
      f16s_ = std::vector<half>(s.size() / sizeof(half));
      memcpy(f16s_.data(), s.data(), s.size());

      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(f16s_.data()), s.size());
      }
      break;
    }
    case BF16: {
      const string& s(literal_proto.bf16s());
      CHECK_EQ(0, s.size() % sizeof(bfloat16));
      bf16s_ = std::vector<bfloat16>(s.size() / sizeof(bfloat16));
      memcpy(bf16s_.data(), s.data(), s.size());

      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(bf16s_.data()), s.size());
      }
      break;
    }
    case F32:
      CopyFromRepeatedField(mutable_f32s(), literal_proto.f32s());
      break;
    case F64:
      CopyFromRepeatedField(mutable_f64s(), literal_proto.f64s());
      break;
    case C64:
      CopyFromRepeatedField(mutable_c64s(), literal_proto.c64s());
      break;
    case TUPLE:
      for (const auto& proto : literal_proto.tuple_literals()) {
        mutable_tuple_literals()->push_back(Literal(proto));
      }
      break;
    default:
      LOG(FATAL) << "Unhandled primitive type " << shape().element_type();
  }
}

const Literal& Literal::GetSubliteral(const ShapeIndex& index) const {
  return const_cast<Literal*>(this)->GetSubliteral(index);
}

Literal& Literal::GetSubliteral(const ShapeIndex& index) {
  Literal* subliteral = this;
  for (int64 i : index) {
    subliteral = &subliteral->tuple_literals_.at(i);
  }
  return *subliteral;
}

}  // namespace xla
