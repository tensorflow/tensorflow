/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using absl::StrCat;

// Return a literal with all arrays of type FromNativeT converted to type
// ToNativeT in the given literal.
template <typename FromNativeT, typename ToNativeT>
Literal ConvertType(LiteralSlice literal) {
  // First construct shape of the result.
  Shape result_shape(literal.shape());
  ShapeUtil::ForEachMutableSubshape(
      &result_shape, [](Shape* subshape, const ShapeIndex&) {
        if (subshape->element_type() ==
            primitive_util::NativeToPrimitiveType<FromNativeT>()) {
          subshape->set_element_type(
              primitive_util::NativeToPrimitiveType<ToNativeT>());
        }
      });
  Literal result(result_shape);

  // Then copy over the data from 'literal' converting FromNativeT values to
  // ToNativeT values as necessary.
  ShapeUtil::ForEachSubshape(
      literal.shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (subshape.IsArray()) {
          if (subshape.element_type() ==
              primitive_util::NativeToPrimitiveType<FromNativeT>()) {
            auto src = literal.data<FromNativeT>(shape_index);
            auto dest = result.data<ToNativeT>(shape_index);
            for (int64 i = 0; i < src.size(); ++i) {
              dest[i] = static_cast<ToNativeT>(src[i]);
            }
          } else {
            TF_CHECK_OK(result.CopyFrom(literal,
                                        /*dest_shape_index=*/shape_index,
                                        /*src_shape_index=*/shape_index));
          }
        }
      });
  return result;
}

}  // namespace

/* static */ Literal LiteralUtil::CreateFromDimensions(
    PrimitiveType primitive_type, absl::Span<const int64> dimensions) {
  return Literal::CreateFromShape(
      ShapeUtil::MakeShape(primitive_type, dimensions));
}

/* static */ Literal LiteralUtil::ConvertBF16ToF32(
    const LiteralSlice& bf16_literal) {
  return ConvertType<bfloat16, float>(bf16_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToBF16(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, bfloat16>(f32_literal);
}

/* static */ Literal LiteralUtil::CreateToken() {
  return Literal(ShapeUtil::MakeTokenShape());
}

/* static */ Literal LiteralUtil::Zero(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return LiteralUtil::CreateR0<uint8>(0);
    case U32:
      return LiteralUtil::CreateR0<uint32>(0);
    case U64:
      return LiteralUtil::CreateR0<uint64>(0);
    case S8:
      return LiteralUtil::CreateR0<int8>(0);
    case S32:
      return LiteralUtil::CreateR0<int32>(0);
    case S64:
      return LiteralUtil::CreateR0<int64>(0);
    case F16:
      return LiteralUtil::CreateR0<half>(static_cast<half>(0.0f));
    case BF16:
      return LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(0.0f));
    case F32:
      return LiteralUtil::CreateR0<float>(0);
    case F64:
      return LiteralUtil::CreateR0<double>(0);
    case C64:
      return LiteralUtil::CreateR0<complex64>(0);
    case PRED:
      return LiteralUtil::CreateR0<bool>(false);
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

/* static */ Literal LiteralUtil::One(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return LiteralUtil::CreateR0<uint8>(1);
    case U32:
      return LiteralUtil::CreateR0<uint32>(1);
    case U64:
      return LiteralUtil::CreateR0<uint64>(1);
    case S8:
      return LiteralUtil::CreateR0<int8>(1);
    case S32:
      return LiteralUtil::CreateR0<int32>(1);
    case S64:
      return LiteralUtil::CreateR0<int64>(1);
    case F16:
      return LiteralUtil::CreateR0<half>(static_cast<half>(1.0f));
    case BF16:
      return LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(1.0f));
    case F32:
      return LiteralUtil::CreateR0<float>(1);
    case F64:
      return LiteralUtil::CreateR0<double>(1);
    case C64:
      return LiteralUtil::CreateR0<complex64>(1);
    case PRED:
      return LiteralUtil::CreateR0<bool>(true);
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

/* static */ Literal LiteralUtil::MinValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return LiteralUtil::CreateR0<uint8>(std::numeric_limits<uint8>::min());
    case U32:
      return LiteralUtil::CreateR0<uint32>(std::numeric_limits<uint32>::min());
    case U64:
      return LiteralUtil::CreateR0<uint64>(std::numeric_limits<uint64>::min());
    case S8:
      return LiteralUtil::CreateR0<int8>(std::numeric_limits<int8>::min());
    case S32:
      return LiteralUtil::CreateR0<int32>(std::numeric_limits<int32>::min());
    case S64:
      return LiteralUtil::CreateR0<int64>(std::numeric_limits<int64>::min());
    case F32:
      return LiteralUtil::CreateR0<float>(
          -std::numeric_limits<float>::infinity());
    case F64:
      return LiteralUtil::CreateR0<double>(
          -std::numeric_limits<double>::infinity());
    case C64:
      LOG(FATAL) << "C64 element type has no minimum value";
    case PRED:
      return LiteralUtil::CreateR0<bool>(false);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      return LiteralUtil::CreateR0<half>(
          static_cast<half>(-std::numeric_limits<float>::infinity()));
    case BF16:
      return LiteralUtil::CreateR0<bfloat16>(
          static_cast<bfloat16>(-std::numeric_limits<float>::infinity()));
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
      return LiteralUtil::CreateR0<uint8>(std::numeric_limits<uint8>::max());
    case U32:
      return LiteralUtil::CreateR0<uint32>(std::numeric_limits<uint32>::max());
    case U64:
      return LiteralUtil::CreateR0<uint64>(std::numeric_limits<uint64>::max());
    case S8:
      return LiteralUtil::CreateR0<int8>(std::numeric_limits<int8>::max());
    case S32:
      return LiteralUtil::CreateR0<int32>(std::numeric_limits<int32>::max());
    case S64:
      return LiteralUtil::CreateR0<int64>(std::numeric_limits<int64>::max());
    case F32:
      return LiteralUtil::CreateR0<float>(
          std::numeric_limits<float>::infinity());
    case F64:
      return LiteralUtil::CreateR0<double>(
          std::numeric_limits<double>::infinity());
    case PRED:
      return LiteralUtil::CreateR0<bool>(true);
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      return LiteralUtil::CreateR0<half>(
          static_cast<half>(std::numeric_limits<float>::infinity()));
    case BF16:
      return LiteralUtil::CreateR0<bfloat16>(
          static_cast<bfloat16>(std::numeric_limits<float>::infinity()));
    case TUPLE:
      LOG(FATAL) << "tuple element type has no maximum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no maximum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal LiteralUtil::CreateR1(
    const tensorflow::core::Bitmap& values) {
  Literal literal(
      ShapeUtil::MakeShape(PRED, {static_cast<int64>(values.bits())}));
  literal.PopulateR1(values);
  return literal;
}

/* static */ Literal LiteralUtil::CreateR1U8(absl::string_view value) {
  Literal literal(ShapeUtil::MakeShape(U8, {static_cast<int64>(value.size())}));
  for (int i = 0; i < value.size(); ++i) {
    literal.Set<uint8>({i}, value[i]);
  }
  return literal;
}

/* static */ Literal LiteralUtil::CreateR2F32Linspace(float from, float to,
                                                      int64 rows, int64 cols) {
  auto value = MakeLinspaceArray2D(from, to, rows, cols);
  return CreateR2FromArray2D(*value);
}

/* static */ Literal LiteralUtil::ReshapeSlice(
    absl::Span<const int64> new_dimensions,
    absl::Span<const int64> minor_to_major, const LiteralSlice& literal) {
  int64 new_num_elements = 1;
  for (int64 i = 0; i < new_dimensions.size(); ++i) {
    new_num_elements *= new_dimensions[i];
  }
  CHECK_EQ(ShapeUtil::ElementsIn(literal.shape()), new_num_elements);
  CHECK_EQ(new_dimensions.size(), minor_to_major.size());

  Literal new_literal(
      ShapeUtil::MakeShape(literal.shape().element_type(), new_dimensions));

  // Create a new shape with the given minor-to-major layout. This shape is used
  // solely for converting linear address to multi-dimensional addresses when
  // writing elements to the new literal.
  Shape shape_with_layout = new_literal.shape();
  *shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);

  // Copy data into new literal, element-by-element.
  for (int64 i = 0; i < ShapeUtil::ElementsIn(literal.shape()); ++i) {
    std::vector<int64> from_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
    std::vector<int64> to_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(shape_with_layout, i);
    switch (literal.shape().element_type()) {
      case PRED:
        new_literal.Set<bool>(to_multi_index,
                              literal.Get<bool>(from_multi_index));
        break;
      case U8:
        new_literal.Set<uint8>(to_multi_index,
                               literal.Get<uint8>(from_multi_index));
        break;
      case U32:
        new_literal.Set<uint32>(to_multi_index,
                                literal.Get<uint32>(from_multi_index));
        break;
      case S32:
        new_literal.Set<int32>(to_multi_index,
                               literal.Get<int32>(from_multi_index));
        break;
      case U64:
        new_literal.Set<uint64>(to_multi_index,
                                literal.Get<uint64>(from_multi_index));
        break;
      case S64:
        new_literal.Set<int64>(to_multi_index,
                               literal.Get<int64>(from_multi_index));
        break;
      case F32:
        new_literal.Set<float>(to_multi_index,
                               literal.Get<float>(from_multi_index));
        break;
      case F64:
        new_literal.Set<double>(to_multi_index,
                                literal.Get<double>(from_multi_index));
        break;
      case C64:
        new_literal.Set<complex64>(to_multi_index,
                                   literal.Get<complex64>(from_multi_index));
        break;
      default:
        LOG(FATAL) << "Unhandled primitive element type: "
                   << PrimitiveType_Name(literal.shape().element_type());
    }
  }

  return new_literal;
}

/* static */ Literal LiteralUtil::GetFirstScalarLiteral(
    const LiteralSlice& literal) {
  CHECK(literal.shape().IsArray());
  CHECK_GT(ShapeUtil::ElementsIn(literal.shape()), 0);
  switch (literal.shape().element_type()) {
    case PRED:
      return LiteralUtil::CreateR0<bool>(literal.GetFirstElement<bool>());
    // 8 bit types.
    case S8:
      return LiteralUtil::CreateR0<int8>(literal.GetFirstElement<int8>());
    case U8:
      return LiteralUtil::CreateR0<uint8>(literal.GetFirstElement<uint8>());
    // 16 bit types.
    case BF16:
      return LiteralUtil::CreateR0<bfloat16>(
          literal.GetFirstElement<bfloat16>());
    case F16:
      return LiteralUtil::CreateR0<half>(literal.GetFirstElement<half>());
    case S16:
      return LiteralUtil::CreateR0<int16>(literal.GetFirstElement<int16>());
    case U16:
      return LiteralUtil::CreateR0<uint16>(literal.GetFirstElement<uint16>());
    // 32 bit types.
    case F32:
      return LiteralUtil::CreateR0<float>(literal.GetFirstElement<float>());
    case S32:
      return LiteralUtil::CreateR0<int32>(literal.GetFirstElement<int32>());
    case U32:
      return LiteralUtil::CreateR0<uint32>(literal.GetFirstElement<uint32>());
    // 64 bit types.
    case C64:
      return LiteralUtil::CreateR0<complex64>(
          literal.GetFirstElement<complex64>());
    case F64:
      return LiteralUtil::CreateR0<double>(literal.GetFirstElement<double>());
    case S64:
      return LiteralUtil::CreateR0<int64>(literal.GetFirstElement<int64>());
    case U64:
      return LiteralUtil::CreateR0<uint64>(literal.GetFirstElement<uint64>());
    default:
      LOG(FATAL) << "Unhandled primitive type "
                 << literal.shape().element_type();
  }
}

/* static */ Literal LiteralUtil::MakeTuple(
    absl::Span<const Literal* const> elements) {
  std::vector<Shape> element_shapes;
  for (const auto* element : elements) {
    element_shapes.push_back(element->shape());
  }
  Literal literal(ShapeUtil::MakeTupleShape(element_shapes));
  for (int i = 0; i < elements.size(); ++i) {
    TF_CHECK_OK(literal.CopyFrom(*elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ Literal LiteralUtil::MakeTupleFromSlices(
    absl::Span<const LiteralSlice> elements) {
  std::vector<Shape> element_shapes;
  for (const auto& element : elements) {
    element_shapes.push_back(element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShape(element_shapes));
  for (int i = 0; i < elements.size(); ++i) {
    TF_CHECK_OK(literal.CopyFrom(elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ Literal LiteralUtil::MakeTupleOwned(
    std::vector<Literal> elements) {
  std::vector<Shape> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShape(element_shapes));
  for (int64 i = 0; i < elements.size(); ++i) {
    TF_CHECK_OK(
        literal.MoveFrom(std::move(elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ string LiteralUtil::MultiIndexAsString(
    absl::Span<const int64> multi_index) {
  return StrCat("{", absl::StrJoin(multi_index, ","), "}");
}

}  // namespace xla
