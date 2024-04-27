/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/literal_util.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/index_util.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/bitmap.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/status.h"

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
            for (int64_t i = 0, end = src.size(); i < end; ++i) {
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

template <PrimitiveType kType>
using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;

template <PrimitiveType kType, typename F, typename... Args>
Literal CreateScalarImpl(F&& value_provider, Args... args) {
  return LiteralUtil::CreateR0<NativeT<kType>>(
      value_provider(std::forward<Args>(args)...));
}

template <template <PrimitiveType> class F, typename... Args>
Literal CreateScalar(PrimitiveType primitive_type, Args... args) {
  return primitive_util::PrimitiveTypeSwitch<Literal>(
      [&](auto primitive_type_constant) -> Literal {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return CreateScalarImpl<primitive_type_constant>(
              F<primitive_type_constant>{}, std::forward<Args>(args)...);
        }
        LOG(FATAL) << "Unhandled primitive type " << primitive_type;
      },
      primitive_type);
}

template <PrimitiveType kType>
struct ZeroProvider {
  NativeT<kType> operator()() const { return static_cast<NativeT<kType>>(0); }
};

template <PrimitiveType kType>
struct OneProvider {
  NativeT<kType> operator()() const { return static_cast<NativeT<kType>>(1); }
};

template <typename T>
struct IsReal {
  static constexpr bool value = std::numeric_limits<T>::is_specialized;
};

template <typename T>
struct IsValidScalarType {
  static constexpr bool value = IsReal<T>::value || is_complex_v<T>;
};

template <typename NativeT>
NativeT GetMaxImpl() {
  if constexpr (IsReal<NativeT>::value) {
    if constexpr (std::numeric_limits<NativeT>::has_infinity) {
      return std::numeric_limits<NativeT>::infinity();
    }
    return std::numeric_limits<NativeT>::max();
  }
  LOG(FATAL) << "No max value for given type.";
}

template <typename NativeT>
NativeT GetMinImpl() {
  if constexpr (IsReal<NativeT>::value) {
    if constexpr (std::numeric_limits<NativeT>::has_infinity) {
      return -std::numeric_limits<NativeT>::infinity();
    }
    return std::numeric_limits<NativeT>::lowest();
  }
  LOG(FATAL) << "No min value for given type.";
}

template <PrimitiveType kType>
struct MaxProvider {
  NativeT<kType> operator()() const { return GetMaxImpl<NativeT<kType>>(); }
};

template <PrimitiveType kType>
struct MinProvider {
  NativeT<kType> operator()() const { return GetMinImpl<NativeT<kType>>(); }
};

template <PrimitiveType kType>
struct FirstElementProvider {
  NativeT<kType> operator()(const LiteralBase& literal) const {
    return literal.GetFirstElement<NativeT<kType>>();
  }
};

template <typename NativeT>
std::enable_if_t<IsReal<NativeT>::value, NativeT> GetMaxElementImpl(
    const LiteralBase& literal) {
  auto view = literal.data<NativeT>();
  return *absl::c_max_element(view);
}

template <typename NativeT>
std::enable_if_t<!IsReal<NativeT>::value, NativeT> GetMaxElementImpl(
    const LiteralBase& literal) {
  LOG(FATAL) << "Unsupported type.";
}

template <PrimitiveType kType>
struct MaxElementProvider {
  NativeT<kType> operator()(const LiteralBase& literal) const {
    return GetMaxElementImpl<NativeT<kType>>(literal);
  }
};

template <typename NativeT>
std::enable_if_t<IsValidScalarType<NativeT>::value, NativeT>
GetElementAtIndexImpl(const LiteralBase* literal,
                      absl::Span<const int64_t> multi_index) {
  return literal->Get<NativeT>(multi_index);
}

template <typename NativeT>
std::enable_if_t<!IsValidScalarType<NativeT>::value, NativeT>
GetElementAtIndexImpl(const LiteralBase* literal,
                      absl::Span<const int64_t> multi_index) {
  LOG(FATAL) << "Not a valid scalar element type.";
}

template <PrimitiveType kType>
struct GetElementAtIndexProvider {
  NativeT<kType> operator()(const LiteralBase* literal,
                            absl::Span<const int64_t> multi_index) const {
    DCHECK_EQ(literal->shape().element_type(), kType);
    return GetElementAtIndexImpl<NativeT<kType>>(literal, multi_index);
  }
};

template <PrimitiveType kType>
void SetScalarAtIndexImpl(MutableLiteralBase& literal,
                          absl::Span<const int64_t> multi_index,
                          const LiteralBase& scalar) {
  DCHECK_EQ(literal.shape().element_type(), kType);
  using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;
  literal.Set<NativeT>(multi_index, scalar.Get<NativeT>({}));
}

}  // namespace

/* static */ Literal LiteralUtil::CreateFromDimensions(
    PrimitiveType primitive_type, absl::Span<const int64_t> dimensions) {
  return Literal::CreateFromShape(
      ShapeUtil::MakeShape(primitive_type, dimensions));
}

/* static */ Literal LiteralUtil::ConvertBF16ToF32(
    const LiteralSlice& bf16_literal) {
  return ConvertType<bfloat16, float>(bf16_literal);
}

/* static */ Literal LiteralUtil::ConvertBF16ToF64(
    const LiteralSlice& bf16_literal) {
  return ConvertType<bfloat16, double>(bf16_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF8E4M3FNUZ(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, tsl::float8_e4m3fnuz>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF8E5M2FNUZ(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, tsl::float8_e5m2fnuz>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToBF16(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, bfloat16>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToS8(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, int8_t>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF64(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, double>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF64ToBF16(
    const LiteralSlice& f64_literal) {
  return ConvertType<double, bfloat16>(f64_literal);
}

/* static */ Literal LiteralUtil::ConvertF64ToF32(
    const LiteralSlice& f64_literal) {
  return ConvertType<double, float>(f64_literal);
}

/* static */ Literal LiteralUtil::ConvertS32ToF32(
    const LiteralSlice& s32_literal) {
  return ConvertType<int32_t, float>(s32_literal);
}

/* static */ Literal LiteralUtil::CreateToken() {
  return Literal(ShapeUtil::MakeTokenShape());
}

/* static */ Literal LiteralUtil::Zero(PrimitiveType primitive_type) {
  return CreateScalar<ZeroProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::One(PrimitiveType primitive_type) {
  return CreateScalar<OneProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::MinValue(PrimitiveType primitive_type) {
  return CreateScalar<MinProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::MaxValue(PrimitiveType primitive_type) {
  return CreateScalar<MaxProvider>(primitive_type);
}

/* static */ absl::StatusOr<Literal> LiteralUtil::NanValue(
    PrimitiveType primitive_type) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return LiteralUtil::CreateR0<NativeT>(
              std::numeric_limits<NativeT>::quiet_NaN());
        }
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          auto nan =
              std::numeric_limits<typename NativeT::value_type>::quiet_NaN();
          return LiteralUtil::CreateR0<NativeT>(NativeT(nan, nan));
        }
        return InvalidArgument("Invalid type for NanValue: %s",
                               PrimitiveType_Name(primitive_type));
      },
      primitive_type);
}

/* static */ Literal LiteralUtil::CreateR1(const tsl::core::Bitmap& values) {
  Literal literal(
      ShapeUtil::MakeShape(PRED, {static_cast<int64_t>(values.bits())}));
  literal.PopulateR1(values);
  return literal;
}

/* static */ Literal LiteralUtil::CreateR1U8(absl::string_view value) {
  Literal literal(
      ShapeUtil::MakeShape(U8, {static_cast<int64_t>(value.size())}));
  for (int i = 0, end = value.size(); i < end; ++i) {
    literal.Set<uint8_t>({i}, value[i]);
  }
  return literal;
}

/* static */ Literal LiteralUtil::CreateR2F32Linspace(float from, float to,
                                                      int64_t rows,
                                                      int64_t cols) {
  auto value = MakeLinspaceArray2D(from, to, rows, cols);
  return CreateR2FromArray2D(*value);
}

/* static */ Literal LiteralUtil::ReshapeSlice(
    absl::Span<const int64_t> new_dimensions,
    absl::Span<const int64_t> minor_to_major, const LiteralSlice& literal) {
  int64_t new_num_elements = 1;
  for (int64_t i = 0, end = new_dimensions.size(); i < end; ++i) {
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
  for (int64_t i = 0; i < ShapeUtil::ElementsIn(literal.shape()); ++i) {
    auto from_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
    auto to_multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(shape_with_layout, i);
    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type_constant) -> void {
          if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
            using NativeT = typename primitive_util::PrimitiveTypeToNative<
                primitive_type_constant>::type;
            new_literal.Set<NativeT>(to_multi_index,
                                     literal.Get<NativeT>(from_multi_index));
            return;
          }
          LOG(FATAL) << "Unhandled primitive element type: "
                     << PrimitiveType_Name(literal.shape().element_type());
        },
        literal.shape().element_type());
  }

  return new_literal;
}

/* static */ Literal LiteralUtil::GetFirstScalarLiteral(
    const LiteralSlice& literal) {
  CHECK(literal.shape().IsArray());
  CHECK_GT(ShapeUtil::ElementsIn(literal.shape()), 0);
  return CreateScalar<FirstElementProvider>(literal.shape().element_type(),
                                            literal);
}

/*static*/ Literal LiteralUtil::GetScalarLiteral(
    const LiteralBase& literal, absl::Span<const int64_t> multi_index) {
  return CreateScalar<GetElementAtIndexProvider>(literal.shape().element_type(),
                                                 &literal, multi_index);
}

/*static*/ void LiteralUtil::SetScalarLiteral(
    MutableLiteralBase& literal, absl::Span<const int64_t> multi_index,
    const LiteralBase& scalar) {
  primitive_util::PrimitiveTypeSwitch<void>(
      [&](auto primitive_type_constant) -> void {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          SetScalarAtIndexImpl<primitive_type_constant>(literal, multi_index,
                                                        scalar);
          return;
        }
        LOG(FATAL) << "Unsupported element type: "
                   << literal.shape().element_type();
      },
      literal.shape().element_type());
}

/* static */ Literal LiteralUtil::MaxElement(const LiteralSlice& literal) {
  CHECK(literal.shape().IsArray());
  CHECK_GT(ShapeUtil::ElementsIn(literal.shape()), 0);
  return CreateScalar<MaxElementProvider>(literal.shape().element_type(),
                                          literal);
}

/* static */ Literal LiteralUtil::MakeTuple(
    absl::Span<const Literal* const> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto* element : elements) {
    element_shapes.push_back(&element->shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = elements.size(); i < end; ++i) {
    TF_CHECK_OK(literal.CopyFrom(*elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ Literal LiteralUtil::MakeTupleFromSlices(
    absl::Span<const LiteralSlice> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = elements.size(); i < end; ++i) {
    TF_CHECK_OK(literal.CopyFrom(elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ Literal LiteralUtil::MakeTupleOwned(
    std::vector<Literal> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int64_t i = 0, end = elements.size(); i < end; ++i) {
    TF_CHECK_OK(
        literal.MoveFrom(std::move(elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ std::string LiteralUtil::MultiIndexAsString(
    absl::Span<const int64_t> multi_index) {
  return StrCat("{", absl::StrJoin(multi_index, ","), "}");
}

/* static */ std::optional<int64_t> LiteralUtil::LiteralAsScalarInt64(
    const Literal& l) {
  if (!ShapeUtil::IsEffectiveScalar(l.shape())) {
    VLOG(2) << "literal is not an effective scalar: " << l.ToString();
    return std::nullopt;
  }
  return l.GetFirstInteger();
}

}  // namespace xla
