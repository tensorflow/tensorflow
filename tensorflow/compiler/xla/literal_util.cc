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

namespace xla {

LiteralUtil::StrideConfig::StrideConfig(
    const Shape& source_shape, const Shape& dest_shape,
    tensorflow::gtl::ArraySlice<int64> dimensions)
    : dimensions(dimensions),
      base(dimensions.size(), 0),
      step(dimensions.size(), 1) {
  if (!dimensions.empty()) {
    // Selects the shape with the highest minor dimension as the one upon
    // where to run the tight stride loop.
    if (source_shape.layout().minor_to_major()[0] >=
        dest_shape.layout().minor_to_major()[0]) {
      minor_dimension = source_shape.layout().minor_to_major()[0];
      dest_stride = IndexUtil::GetDimensionStride(dest_shape, minor_dimension);
    } else {
      minor_dimension = dest_shape.layout().minor_to_major()[0];
      source_stride =
          IndexUtil::GetDimensionStride(source_shape, minor_dimension);
    }
    minor_loop_size = dimensions[minor_dimension];
    step[minor_dimension] = minor_loop_size;
  }
}

/* static */ std::unique_ptr<Literal> LiteralUtil::CreateFromShape(
    const Shape& shape) {
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() = shape;
  Reserve(ShapeUtil::ElementsIn(literal->shape()), literal.get());
  return literal;
}

/* static */ std::unique_ptr<Literal> LiteralUtil::CreateFromDimensions(
    PrimitiveType primitive_type,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  return CreateFromShape(ShapeUtil::MakeShape(primitive_type, dimensions));
}

template <typename T>
/* static */ Status LiteralUtil::CopyRange(
    const Literal& src_literal, tensorflow::gtl::ArraySlice<int64> src_base,
    Literal* dest_literal, tensorflow::gtl::ArraySlice<int64> dest_base,
    tensorflow::gtl::ArraySlice<int64> copy_size) {
  const Shape& src_shape = src_literal.shape();
  const Shape& dest_shape = dest_literal->shape();
  tensorflow::gtl::ArraySlice<T> src_data = GetArraySlice<T>(src_literal);
  tensorflow::gtl::MutableArraySlice<T> dest_data =
      GetMutableArraySlice<T>(dest_literal);

  TF_RET_CHECK(ShapeUtil::Rank(src_shape) == src_base.size());
  TF_RET_CHECK(ShapeUtil::Rank(dest_shape) == dest_base.size());
  if (ShapeUtil::Rank(src_shape) == 0 || ShapeUtil::Rank(dest_shape) == 0) {
    // If any of the two shapes are scalars, we can just call the StridedCopy()
    // directly, and we know we will be copying only one value.
    TF_RET_CHECK(copy_size.empty());
    StridedCopy(dest_data, LinearIndex(*dest_literal, dest_base), 0, src_data,
                LinearIndex(src_literal, src_base), 0, 1);
  } else if (!ShapeUtil::HasZeroElements(dest_shape)) {
    TF_RET_CHECK(!ShapeUtil::HasZeroElements(src_shape));
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

      int64 src_index = LinearIndex(src_literal, src_indexes);
      int64 dest_index = LinearIndex(*dest_literal, dest_indexes);

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

/* static */ Status LiteralUtil::Copy(
    const Literal& src_literal, tensorflow::gtl::ArraySlice<int64> src_base,
    Literal* dest_literal, tensorflow::gtl::ArraySlice<int64> dest_base,
    tensorflow::gtl::ArraySlice<int64> copy_size) {
  TF_RET_CHECK(
      ShapeUtil::SameElementType(src_literal.shape(), dest_literal->shape()));
  switch (src_literal.shape().element_type()) {
    case U32:
      return CopyRange<uint32>(src_literal, src_base, dest_literal, dest_base,
                               copy_size);
    case U64:
      return CopyRange<uint64>(src_literal, src_base, dest_literal, dest_base,
                               copy_size);
    case S32:
      return CopyRange<int32>(src_literal, src_base, dest_literal, dest_base,
                              copy_size);
    case S64:
      return CopyRange<int64>(src_literal, src_base, dest_literal, dest_base,
                              copy_size);
    case F32:
      return CopyRange<float>(src_literal, src_base, dest_literal, dest_base,
                              copy_size);
    case F64:
      return CopyRange<double>(src_literal, src_base, dest_literal, dest_base,
                               copy_size);
    case PRED:
      return CopyRange<bool>(src_literal, src_base, dest_literal, dest_base,
                             copy_size);
    default:
      break;
  }
  return Unimplemented("Unhandled primitive type %d",
                       src_literal.shape().element_type());
}

/* static */ Literal LiteralUtil::Zero(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *LiteralUtil::CreateR0<uint8>(0);
    case U32:
      return *LiteralUtil::CreateR0<uint32>(0);
    case U64:
      return *LiteralUtil::CreateR0<uint64>(0);
    case S8:
      return *LiteralUtil::CreateR0<int8>(0);
    case S32:
      return *LiteralUtil::CreateR0<int32>(0);
    case S64:
      return *LiteralUtil::CreateR0<int64>(0);
    case F32:
      return *LiteralUtil::CreateR0<float>(0);
    case F64:
      return *LiteralUtil::CreateR0<double>(0);
    case PRED:
      return *LiteralUtil::CreateR0<bool>(false);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      LOG(FATAL) << "f16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type cannot take on value of 0";
    case OPAQUE:
      LOG(FATAL) << "opaque element type cannot take on value of 0";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal LiteralUtil::One(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *LiteralUtil::CreateR0<uint8>(1);
    case U32:
      return *LiteralUtil::CreateR0<uint32>(1);
    case U64:
      return *LiteralUtil::CreateR0<uint64>(1);
    case S8:
      return *LiteralUtil::CreateR0<int8>(1);
    case S32:
      return *LiteralUtil::CreateR0<int32>(1);
    case S64:
      return *LiteralUtil::CreateR0<int64>(1);
    case F32:
      return *LiteralUtil::CreateR0<float>(1);
    case F64:
      return *LiteralUtil::CreateR0<double>(1);
    case PRED:
      return *LiteralUtil::CreateR0<bool>(true);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      LOG(FATAL) << "f16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type cannot take on value of 1";
    case OPAQUE:
      LOG(FATAL) << "opaque element type cannot take on value of 1";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal LiteralUtil::MinValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *LiteralUtil::CreateR0<uint8>(std::numeric_limits<uint8>::min());
    case U32:
      return *LiteralUtil::CreateR0<uint32>(std::numeric_limits<uint32>::min());
    case U64:
      return *LiteralUtil::CreateR0<uint64>(std::numeric_limits<uint64>::min());
    case S8:
      return *LiteralUtil::CreateR0<int8>(std::numeric_limits<int8>::min());
    case S32:
      return *LiteralUtil::CreateR0<int32>(std::numeric_limits<int32>::min());
    case S64:
      return *LiteralUtil::CreateR0<int64>(std::numeric_limits<int64>::min());
    case F32:
      return *LiteralUtil::CreateR0<float>(
          -std::numeric_limits<float>::infinity());
    case F64:
      return *LiteralUtil::CreateR0<double>(
          -std::numeric_limits<double>::infinity());
    case PRED:
      return *LiteralUtil::CreateR0<bool>(false);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      LOG(FATAL) << "f16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type has no minimum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no minimum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal LiteralUtil::MaxValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return *LiteralUtil::CreateR0<uint8>(std::numeric_limits<uint8>::max());
    case U32:
      return *LiteralUtil::CreateR0<uint32>(std::numeric_limits<uint32>::max());
    case U64:
      return *LiteralUtil::CreateR0<uint64>(std::numeric_limits<uint64>::max());
    case S8:
      return *LiteralUtil::CreateR0<int8>(std::numeric_limits<int8>::max());
    case S32:
      return *LiteralUtil::CreateR0<int32>(std::numeric_limits<int32>::max());
    case S64:
      return *LiteralUtil::CreateR0<int64>(std::numeric_limits<int64>::max());
    case F32:
      return *LiteralUtil::CreateR0<float>(
          std::numeric_limits<float>::infinity());
    case F64:
      return *LiteralUtil::CreateR0<double>(
          std::numeric_limits<double>::infinity());
    case PRED:
      return *LiteralUtil::CreateR0<bool>(true);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      LOG(FATAL) << "f16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type has no maximum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no maximum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR1(
    const tensorflow::core::Bitmap& values) {
  auto literal = MakeUnique<Literal>();
  PopulateR1(values, literal.get());
  return literal;
}

/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR1U8(
    tensorflow::StringPiece value) {
  auto literal = MakeUnique<Literal>();
  *literal->mutable_shape() =
      ShapeUtil::MakeShape(U8, {static_cast<int64>(value.size())});
  literal->set_u8s(value.ToString());
  return literal;
}

/* static */ std::unique_ptr<Literal> LiteralUtil::CreateR2F32Linspace(
    float from, float to, int64 rows, int64 cols) {
  auto value = MakeLinspaceArray2D(from, to, rows, cols);
  return CreateR2FromArray2D(*value);
}

/* static */ std::unique_ptr<Literal> LiteralUtil::Relayout(
    const Literal& original, const Layout& layout) {
  std::unique_ptr<Literal> result = CloneToUnique(original);
  *result->mutable_shape()->mutable_layout() = layout;

  const Shape& shape = original.shape();
  DimensionVector base(ShapeUtil::Rank(shape), 0);
  DimensionVector copy_size(shape.dimensions().begin(),
                            shape.dimensions().end());

  TF_CHECK_OK(Copy(original, base, result.get(), base, copy_size));
  return result;
}

/* static */ StatusOr<std::unique_ptr<Literal>> LiteralUtil::Reshape(
    const xla::Literal& input, tensorflow::gtl::ArraySlice<int64> dimensions) {
  if (ShapeUtil::IsTuple(input.shape())) {
    return InvalidArgument("Reshape does not support tuples.");
  }
  std::unique_ptr<Literal> output;
  if (!LayoutUtil::IsMonotonicWithDim0Major(input.shape().layout())) {
    std::vector<int64> minor_to_major(ShapeUtil::Rank(input.shape()));
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(),
              static_cast<int64>(0));
    output = Relayout(input, LayoutUtil::MakeLayout(minor_to_major));
  } else {
    output = CloneToUnique(input);
  }
  // Because the layout is monotonic, we can simply reuse the same sequence of
  // values without changing their order.
  *output->mutable_shape() =
      ShapeUtil::MakeShape(input.shape().element_type(), dimensions);

  int64 elements_before = ShapeUtil::ElementsIn(input.shape());
  int64 elements_after = ShapeUtil::ElementsIn(output->shape());
  if (elements_before != elements_after) {
    return InvalidArgument(
        "Shapes before and after LiteralUtil::Reshape have different numbers "
        "of elements: %s vs %s.",
        ShapeUtil::HumanString(input.shape()).c_str(),
        ShapeUtil::HumanString(output->shape()).c_str());
  }
  return std::move(output);
}

/* static */ std::unique_ptr<Literal> LiteralUtil::Transpose(
    const Literal& original, tensorflow::gtl::ArraySlice<int64> permutation) {
  CHECK(!ShapeUtil::IsTuple(original.shape()))
      << "Tuple is not supported for transpose";
  CHECK(IsPermutation(permutation, ShapeUtil::Rank(original.shape())))
      << "Given permutation is not a permutation of dimension numbers";
  // To transpose the array, we just permute the dimensions and layout, and
  // do a straight memory copy of the raw data set.
  // This is considerably faster than iterating over every array element using
  // the EachCell<>() and Set<>() APIs.
  std::vector<int64> inverse_permutation = InversePermutation(permutation);
  Shape shape =
      ShapeUtil::PermuteDimensions(inverse_permutation, original.shape());
  // Replace the layout with one affine to the original shape, such that a
  // transpose operation can be performed by leaving the flat values
  // representation intact.
  // For example, consider the shape F32[11,8]{1,0} under a {1,0} permutation.
  // The shape with affine layout resulting from that operation will be
  // F32[8,11]{0,1}, since it leave the original most minor (the 8 sized), the
  // most minor.
  // Essentially, given MinMaj(Di) the position of the Di dimension within the
  // minor to major vector, and given T(Di) the index that the original Di
  // dimension has within the transposed array, a layout is affine if
  // MinMaj(Di) == TMinMaj(T(Di)), with TMinMaj() being the minor to major
  // vector of the affine layout.
  Layout* layout = shape.mutable_layout();
  layout->clear_minor_to_major();
  for (auto index : original.shape().layout().minor_to_major()) {
    layout->add_minor_to_major(inverse_permutation[index]);
  }
  std::unique_ptr<Literal> new_literal = CreateFromShape(shape);
  DCHECK_GE(ShapeUtil::ByteSizeOf(new_literal->shape()),
            ShapeUtil::ByteSizeOf(original.shape()));
  std::memcpy(MutableInternalData(new_literal.get()), InternalData(original),
              ShapeUtil::ByteSizeOf(original.shape()));
  return new_literal;
}

/* static */ std::unique_ptr<Literal> LiteralUtil::Slice(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices) {
  CHECK(!ShapeUtil::IsTuple(literal.shape()))
      << "tuple is not supported for reshape";

  DimensionVector result_dimensions;
  for (int64 dnum = 0; dnum < ShapeUtil::Rank(literal.shape()); ++dnum) {
    CHECK_GE(start_indices[dnum], 0);
    CHECK_LE(limit_indices[dnum], literal.shape().dimensions(dnum));
    int64 dimension = limit_indices[dnum] - start_indices[dnum];
    CHECK_GT(dimension, 0);
    result_dimensions.push_back(dimension);
  }
  const auto result_shape = ShapeUtil::MakeShapeWithLayout(
      literal.shape().element_type(), result_dimensions,
      AsInt64Slice(literal.shape().layout().minor_to_major()));

  auto result_literal = MakeUnique<Literal>();
  *result_literal->mutable_shape() = result_shape;
  Reserve(ShapeUtil::ElementsIn(result_shape), result_literal.get());

  DimensionVector new_indices(ShapeUtil::Rank(result_shape));
  switch (result_shape.element_type()) {
    case F32:
      LiteralUtil::EachCell<float>(
          *result_literal,
          [&](tensorflow::gtl::ArraySlice<int64> indices, float /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            float value = LiteralUtil::Get<float>(literal, new_indices);
            LiteralUtil::Set<float>(result_literal.get(), indices, value);
          });
      return result_literal;
    case S32:
      LiteralUtil::EachCell<int32>(
          *result_literal,
          [&](tensorflow::gtl::ArraySlice<int64> indices, int32 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            int32 value = LiteralUtil::Get<int32>(literal, new_indices);
            LiteralUtil::Set<int32>(result_literal.get(), indices, value);
          });
      return result_literal;
    case U32:
      LiteralUtil::EachCell<uint32>(
          *result_literal,
          [&](tensorflow::gtl::ArraySlice<int64> indices, uint32 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            uint32 value = LiteralUtil::Get<uint32>(literal, new_indices);
            LiteralUtil::Set<uint32>(result_literal.get(), indices, value);
          });
      return result_literal;
    default:
      LOG(FATAL) << "not yet implemented: "
                 << PrimitiveType_Name(result_shape.element_type());
  }
}

/* static */ std::unique_ptr<Literal> LiteralUtil::CloneToUnique(
    const Literal& literal) {
  auto unique = MakeUnique<Literal>();
  *unique = literal;
  return unique;
}

/* static */ string LiteralUtil::GetAsString(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> multi_index) {
  switch (literal.shape().element_type()) {
    case PRED:
      return Get<bool>(literal, multi_index) ? "true" : "false";
    case U8:
      return tensorflow::strings::StrCat(Get<uint8>(literal, multi_index));
    case S32:
      return tensorflow::strings::StrCat(Get<int32>(literal, multi_index));
    case S64:
      return tensorflow::strings::StrCat(Get<int64>(literal, multi_index));
    case U32:
      return tensorflow::strings::StrCat(Get<uint32>(literal, multi_index));
    case U64:
      return tensorflow::strings::StrCat(Get<uint64>(literal, multi_index));
    case F32:
      return tensorflow::strings::StrCat(Get<float>(literal, multi_index));
    case F64:
      return tensorflow::strings::StrCat(Get<double>(literal, multi_index));
    default:
      return tensorflow::strings::StrCat(
          "[", PrimitiveType_Name(literal.shape().element_type()), "]");
  }
}

/* static */ int64 LiteralUtil::LinearIndex(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> multi_index) {
  return IndexUtil::MultidimensionalIndexToLinearIndex(literal.shape(),
                                                       multi_index);
}

/* static */ string LiteralUtil::ToString(const Literal& literal) {
  const Shape& shape = literal.shape();
  std::vector<string> pieces;

  auto element_to_string =
      [&literal](tensorflow::gtl::ArraySlice<int64> indices) -> string {
    PrimitiveType element_type = literal.shape().element_type();
    if (element_type == PRED) {
      // We display predicates in a densely packed form.
      return Get<bool>(literal, indices) ? "1" : "0";
    }
    return ((!indices.empty() && indices.back() > 0) ? ", " : "") +
           GetAsString(literal, indices);
  };

  // TODO(b/32894291): refactor this code to reduce code duplication.
  if (ShapeUtil::IsTuple(shape)) {
    pieces.push_back(ShapeUtil::HumanString(shape));
    pieces.push_back(" (\n");
    for (const auto& element_literal : literal.tuple_literals()) {
      pieces.push_back(ToString(element_literal));
      pieces.push_back(",\n");
    }
    pieces.push_back(")");
  } else if (ShapeUtil::Rank(shape) == 0) {
    pieces.push_back(GetAsString(literal, {}));
  } else if (ShapeUtil::Rank(shape) == 1) {
    pieces.push_back("{");
    for (int64 i0 = 0; i0 < shape.dimensions(0); ++i0) {
      pieces.push_back(element_to_string({i0}));
    }
    pieces.push_back("}");
  } else if (ShapeUtil::Rank(shape) == 2) {
    pieces.push_back(ShapeUtil::HumanString(shape));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape.dimensions(0); ++i0) {
      pieces.push_back("  { ");
      for (int64 i1 = 0; i1 < shape.dimensions(1); ++i1) {
        pieces.push_back(element_to_string({i0, i1}));
      }
      pieces.push_back(" ");
      pieces.push_back("},\n");
    }
    pieces.push_back("}");
  } else if (ShapeUtil::Rank(shape) == 3) {
    pieces.push_back(ShapeUtil::HumanString(shape));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape.dimensions(0); ++i0) {
      pieces.push_back(i0 > 0 ? ",\n{" : "{");
      for (int64 i1 = 0; i1 < shape.dimensions(1); ++i1) {
        pieces.push_back(i1 > 0 ? ",\n  { " : " { ");
        for (int64 i2 = 0; i2 < shape.dimensions(2); ++i2) {
          pieces.push_back(element_to_string({i0, i1, i2}));
        }
        pieces.push_back(" }");
      }
      pieces.push_back(" }");
    }
    pieces.push_back("\n}");
  } else if (ShapeUtil::Rank(shape) == 4) {
    pieces.push_back(ShapeUtil::HumanString(shape));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape.dimensions(0); ++i0) {
      pieces.push_back(tensorflow::strings::Printf("  {  // i0=%lld\n", i0));
      for (int64 i1 = 0; i1 < shape.dimensions(1); ++i1) {
        pieces.push_back(
            tensorflow::strings::Printf("    {  // i1=%lld\n", i1));
        for (int64 i2 = 0; i2 < shape.dimensions(2); ++i2) {
          pieces.push_back("      {");
          for (int64 i3 = 0; i3 < shape.dimensions(3); ++i3) {
            pieces.push_back(element_to_string({i0, i1, i2, i3}));
          }
          pieces.push_back("},\n");
        }
        pieces.push_back("    },\n");
      }
      pieces.push_back("  },\n");
    }
    pieces.push_back("}");
  } else if (ShapeUtil::Rank(shape) == 5) {
    pieces.push_back(ShapeUtil::HumanString(shape));
    pieces.push_back(" {\n");
    for (int64 i0 = 0; i0 < shape.dimensions(0); ++i0) {
      pieces.push_back(tensorflow::strings::Printf("  {  // i0=%lld\n", i0));
      for (int64 i1 = 0; i1 < shape.dimensions(1); ++i1) {
        pieces.push_back(
            tensorflow::strings::Printf("    {  // i1=%lld\n", i1));
        for (int64 i2 = 0; i2 < shape.dimensions(2); ++i2) {
          pieces.push_back(
              tensorflow::strings::Printf("      {  // i2=%lld\n", i2));
          for (int64 i3 = 0; i3 < shape.dimensions(3); ++i3) {
            pieces.push_back("        {");
            for (int64 i4 = 0; i4 < shape.dimensions(4); ++i4) {
              pieces.push_back(element_to_string({i0, i1, i2, i3, i4}));
            }
            pieces.push_back("},\n");
          }
          pieces.push_back("      },\n");
        }
        pieces.push_back("    },\n");
      }
      pieces.push_back("  },\n");
    }
    pieces.push_back("}");
  } else {
    pieces.push_back(ShapeUtil::HumanString(shape));
    pieces.push_back(" {...}");
  }

  return tensorflow::str_util::Join(pieces, "");
}

/* static */ std::unique_ptr<Literal> LiteralUtil::MakeTuple(
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

/* static */ const void* LiteralUtil::InternalData(const Literal& literal) {
  switch (literal.shape().element_type()) {
    case PRED:
      return reinterpret_cast<const void*>(literal.preds().data());
    case U8:
      return reinterpret_cast<const void*>(literal.u8s().data());
    case S32:
      return reinterpret_cast<const void*>(literal.s32s().data());
    case S64:
      return reinterpret_cast<const void*>(literal.s64s().data());
    case U32:
      return reinterpret_cast<const void*>(literal.u32s().data());
    case U64:
      return reinterpret_cast<const void*>(literal.u64s().data());
    case F32:
      return reinterpret_cast<const void*>(literal.f32s().data());
    case F64:
      return reinterpret_cast<const void*>(literal.f64s().data());
    default:
      LOG(FATAL) << "primitive type not supported in literals: "
                 << PrimitiveType_Name(literal.shape().element_type());
  }
}

/* static */ void* LiteralUtil::MutableInternalData(Literal* literal) {
  return const_cast<void*>(LiteralUtil::InternalData(*literal));
}

/* static */ void LiteralUtil::Reserve(int64 num_elements, Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  switch (literal->shape().element_type()) {
    case PRED:
      Resize<bool>(num_elements, false, literal);
      break;
    case S8:
      Resize<int8>(num_elements, 0, literal);
      break;
    case U8:
      Resize<uint8>(num_elements, 0, literal);
      break;
    case S32:
      Resize<int32>(num_elements, 0, literal);
      break;
    case S64:
      Resize<int64>(num_elements, 0, literal);
      break;
    case U32:
      Resize<uint32>(num_elements, 0, literal);
      break;
    case U64:
      Resize<uint64>(num_elements, 0, literal);
      break;
    case F32:
      Resize<float>(num_elements, 0, literal);
      break;
    case F64:
      Resize<double>(num_elements, 0, literal);
      break;
    default:
      LOG(FATAL) << "primitive type not supported in literals: "
                 << PrimitiveType_Name(literal->shape().element_type());
  }
}

/* static */ tensorflow::Status LiteralUtil::ValidateLiteral(
    const Literal& literal) {
  TF_CHECK_OK(ShapeUtil::ValidateShape(literal.shape()));
  int64 expected = ShapeUtil::ElementsIn(literal.shape());
  int64 actual = -1;
  switch (literal.shape().element_type()) {
    case PRED:
      actual = literal.preds().size();
      break;
    case U8:
      actual = literal.u8s().size();
      break;
    case S32:
      actual = literal.s32s_size();
      break;
    case U32:
      actual = literal.u32s_size();
      break;
    case S64:
      actual = literal.s64s_size();
      break;
    case U64:
      actual = literal.u64s_size();
      break;
    case F32:
      actual = literal.f32s_size();
      break;
    case F64:
      actual = literal.f64s_size();
      break;
    default:
      return tensorflow::errors::Unimplemented(
          "unhandled element type for literal validation: " +
          PrimitiveType_Name(literal.shape().element_type()));
  }

  if (expected != actual) {
    return tensorflow::errors::InvalidArgument(tensorflow::strings::Printf(
        "literal has bad number of elements for its shape %s: want %lld "
        "got %lld",
        ShapeUtil::HumanString(literal.shape()).c_str(), expected, actual));
  }

  return tensorflow::Status::OK();
}

/* static */ void LiteralUtil::EachCellAsString(
    const Literal& literal,
    const std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                             const string& value)>& per_cell) {
  if (ShapeUtil::HasZeroElements(literal.shape())) {
    return;
  }
  std::vector<int64> indices = IndexUtil::LinearIndexToMultidimensionalIndex(
      literal.shape(), /*linear_index=*/0);
  do {
    per_cell(indices, GetAsString(literal, indices));
  } while (IndexUtil::BumpIndices(literal.shape(), &indices));
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
    return (LiteralUtil::Get<NativeT>(literal1, *multi_index) ==
            LiteralUtil::Get<NativeT>(literal2, *multi_index));
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

/* static */ bool LiteralUtil::Equal(const Literal& literal1,
                                     const Literal& literal2) {
  if (!ShapeUtil::Compatible(literal1.shape(), literal2.shape())) {
    return false;
  }
  if (ShapeUtil::IsTuple(literal1.shape())) {
    // Because the shapes are compatible, they must have the same number of
    // tuple elements.
    CHECK_EQ(literal1.tuple_literals_size(), literal2.tuple_literals_size());
    for (int i = 0; i < literal1.tuple_literals_size(); ++i) {
      if (!Equal(literal1.tuple_literals(i), literal2.tuple_literals(i))) {
        return false;
      }
    }
    return true;
  } else {
    std::vector<int64> multi_index(ShapeUtil::Rank(literal1.shape()), 0);
    switch (literal1.shape().element_type()) {
      case PRED:
        return EqualElements<bool>(literal1, literal2, 0, &multi_index);
      case U8:
        return EqualElements<uint8>(literal1, literal2, 0, &multi_index);
      case S32:
        return EqualElements<int32>(literal1, literal2, 0, &multi_index);
      case S64:
        return EqualElements<int64>(literal1, literal2, 0, &multi_index);
      case U32:
        return EqualElements<uint32>(literal1, literal2, 0, &multi_index);
      case U64:
        return EqualElements<uint64>(literal1, literal2, 0, &multi_index);
      case F32:
        return EqualElements<float>(literal1, literal2, 0, &multi_index);
      case F64:
        return EqualElements<double>(literal1, literal2, 0, &multi_index);
      default:
        LOG(FATAL) << "Unimplemented: LiteralUtil::Equal for type "
                   << PrimitiveType_Name(literal1.shape().element_type());
    }
  }
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<bool>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  auto values = literal->mutable_preds();
  return tensorflow::gtl::MutableArraySlice<bool>(values->mutable_data(),
                                                  values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<int8>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  // C++11 standard, basic_string 21.4.1.5, values should be stored
  // contiguously. From C++17 a mutable data() member will be provided.
  auto values = literal->mutable_u8s();
  return tensorflow::gtl::MutableArraySlice<int8>(
      reinterpret_cast<int8*>(&(*values)[0]), values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<uint8>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  // C++11 standard, basic_string 21.4.1.5, values should be stored
  // contiguously. From C++17 a mutable data() member will be provided.
  auto values = literal->mutable_u8s();
  return tensorflow::gtl::MutableArraySlice<uint8>(
      reinterpret_cast<uint8*>(&(*values)[0]), values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<int32>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  auto values = literal->mutable_s32s();
  return tensorflow::gtl::MutableArraySlice<int32>(values->mutable_data(),
                                                   values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<uint32>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  auto values = literal->mutable_u32s();
  return tensorflow::gtl::MutableArraySlice<uint32>(values->mutable_data(),
                                                    values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<int64>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  static_assert(sizeof(int64) == sizeof(tensorflow::protobuf_int64) &&
                    alignof(int64) == alignof(tensorflow::protobuf_int64),
                "The int64 and tensorflow::protobuf_int64 types are not "
                "compatible");
  auto values = literal->mutable_s64s();
  // Because of the fact that tensorflow::protobuf_int64 is defined as int64_t
  // while tensorflow::int64 is defined as long long, a reinterpret_cast<> is
  // necessary from the raw data pointer returned by the mutable_data() API.
  return tensorflow::gtl::MutableArraySlice<int64>(
      reinterpret_cast<int64*>(values->mutable_data()), values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<uint64>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  static_assert(sizeof(uint64) == sizeof(tensorflow::protobuf_uint64) &&
                    alignof(uint64) == alignof(tensorflow::protobuf_uint64),
                "The uint64 and tensorflow::protobuf_uint64 types are not "
                "compatible");
  auto values = literal->mutable_u64s();
  // Because of the fact that tensorflow::protobuf_uint64 is defined as uint64_t
  // while tensorflow::uint64 is defined as unsigned long long, a
  // reinterpret_cast<> is necessary from the raw data pointer returned by the
  // mutable_data() API.
  return tensorflow::gtl::MutableArraySlice<uint64>(
      reinterpret_cast<uint64*>(values->mutable_data()), values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<float>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  auto values = literal->mutable_f32s();
  return tensorflow::gtl::MutableArraySlice<float>(values->mutable_data(),
                                                   values->size());
}

template <>
/* static */ tensorflow::gtl::MutableArraySlice<double>
LiteralUtil::GetMutableArraySlice(Literal* literal) {
  auto values = literal->mutable_f64s();
  return tensorflow::gtl::MutableArraySlice<double>(values->mutable_data(),
                                                    values->size());
}

template <>
/* static */ tensorflow::gtl::ArraySlice<bool> LiteralUtil::GetArraySlice<bool>(
    const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), PRED);
  return literal.preds();
}

template <>
/* static */ tensorflow::gtl::ArraySlice<uint8>
LiteralUtil::GetArraySlice<uint8>(const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), U8);
  return tensorflow::gtl::ArraySlice<uint8>(
      reinterpret_cast<const uint8*>(literal.u8s().data()),
      literal.u8s().size());
}

template <>
/* static */ tensorflow::gtl::ArraySlice<int8> LiteralUtil::GetArraySlice<int8>(
    const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), S8);
  return tensorflow::gtl::ArraySlice<int8>(
      reinterpret_cast<const int8*>(literal.u8s().data()),
      literal.u8s().size());
}

template <>
/* static */ tensorflow::gtl::ArraySlice<uint32>
LiteralUtil::GetArraySlice<uint32>(const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), U32);
  return literal.u32s();
}

template <>
/* static */ tensorflow::gtl::ArraySlice<uint64>
LiteralUtil::GetArraySlice<uint64>(const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), U64);
  return AsUInt64Slice(literal.u64s());
}

template <>
/* static */ tensorflow::gtl::ArraySlice<int32>
LiteralUtil::GetArraySlice<int32>(const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), S32);
  return literal.s32s();
}

template <>
/* static */ tensorflow::gtl::ArraySlice<int64>
LiteralUtil::GetArraySlice<int64>(const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), S64);
  return AsInt64Slice(literal.s64s());
}

template <>
/* static */ tensorflow::gtl::ArraySlice<double>
LiteralUtil::GetArraySlice<double>(const Literal& literal) {
  CHECK_EQ(literal.shape().element_type(), F64);
  return literal.f64s();
}

template <typename NativeT>
static bool AllElementsEqualValue(const Literal& literal, NativeT value) {
  for (int64 i = 0; i < ShapeUtil::ElementsIn(literal.shape()); ++i) {
    auto multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
    if (LiteralUtil::Get<NativeT>(literal, multi_index) != value) {
      return false;
    }
  }
  return true;
}

/* static */ bool LiteralUtil::IsAll(const Literal& literal, int8 value) {
  switch (literal.shape().element_type()) {
    case U8:
      if (value >= 0) {
        return AllElementsEqualValue<uint8>(literal, value);
      }
      return false;
    case U32:
      if (value >= 0) {
        return AllElementsEqualValue<uint32>(literal, value);
      }
      return false;
    case U64:
      if (value >= 0) {
        return AllElementsEqualValue<uint64>(literal, value);
      }
      return false;
    case S8:
      return AllElementsEqualValue<int8>(literal, value);
    case S32:
      return AllElementsEqualValue<int32>(literal, value);
    case S64:
      return AllElementsEqualValue<int64>(literal, value);
    case F32:
      return AllElementsEqualValue<float>(literal, value);
    case F64:
      return AllElementsEqualValue<double>(literal, value);
    case PRED:
      if (value == 0) {
        return AllElementsEqualValue<bool>(literal, false);
      }
      if (value == 1) {
        return AllElementsEqualValue<bool>(literal, true);
      }
      return false;
    default:
      return false;
  }
}

/* static */ bool LiteralUtil::IsAllFloat(const Literal& literal, float value) {
  switch (literal.shape().element_type()) {
    case F32:
      return AllElementsEqualValue<float>(literal, value);
    case F64:
      return AllElementsEqualValue<double>(literal, value);
    default:
      return false;
  }
}

/* static */ bool LiteralUtil::IsZero(
    const Literal& literal, tensorflow::gtl::ArraySlice<int64> indices) {
  switch (literal.shape().element_type()) {
    case U8:
      return Get<uint8>(literal, indices) == 0;
    case U32:
      return Get<uint32>(literal, indices) == 0;
    case U64:
      return Get<uint64>(literal, indices) == 0;
    case S8:
      return Get<int8>(literal, indices) == 0;
    case S32:
      return Get<int32>(literal, indices) == 0;
    case S64:
      return Get<int64>(literal, indices) == 0;
    case F32:
      return Get<float>(literal, indices) == 0.0f;
    case F64:
      return Get<double>(literal, indices) == 0.0;
    case PRED:
      return Get<bool>(literal, indices) == false;
    default:
      LOG(FATAL) << "Input literal must be an array.";
  }
}

template <>
/* static */ void LiteralUtil::Resize<bool>(int64 num_elements, bool value,
                                            Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_preds()->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<int8>(int64 num_elements, int8 value,
                                            Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_u8s()->resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<uint8>(int64 num_elements, uint8 value,
                                             Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_u8s()->resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<int32>(int64 num_elements, int32 value,
                                             Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_s32s()->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<uint32>(int64 num_elements, uint32 value,
                                              Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_u32s()->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<int64>(int64 num_elements, int64 value,
                                             Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_s64s()->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<uint64>(int64 num_elements, uint64 value,
                                              Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_u64s()->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<float>(int64 num_elements, float value,
                                             Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_f32s()->Resize(num_elements, value);
}

template <>
/* static */ void LiteralUtil::Resize<double>(int64 num_elements, double value,
                                              Literal* literal) {
  CHECK_EQ(ShapeUtil::ElementsIn(literal->shape()), num_elements);
  literal->mutable_f64s()->Resize(num_elements, value);
}

}  // namespace xla
