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

#ifndef XLA_CLIENT_LIB_CONSTANTS_H_
#define XLA_CLIENT_LIB_CONSTANTS_H_

#include <type_traits>

#include "absl/status/statusor.h"
#include "xla/client/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Returns scalar 'value' as a scalar of 'type'. Unlike ConstantR0, 'type' is
// determined at C++ run-time, rather than C++ compile-time.
// If 'value' is floating point but 'type' is not, or if 'value' is complex but
// 'type' is not, an error will be returned. This is to catch accidental
// truncation; in such cases, use an explicit cast.
template <typename T>
XlaOp ConstantR0WithType(XlaBuilder* builder, PrimitiveType type, T value) {
  if (std::is_floating_point<T>::value &&
      !(primitive_util::IsFloatingPointType(type) ||
        primitive_util::IsComplexType(type))) {
    return builder->ReportError(InvalidArgument(
        "Invalid cast from floating point type to %s in ConstantR0WithType.",
        PrimitiveType_Name(type)));
  }
  if (std::is_same<T, complex64>::value &&
      !primitive_util::IsComplexType(type)) {
    return builder->ReportError(InvalidArgument(
        "Invalid cast from complex type to %s in ConstantR0WithType.",
        PrimitiveType_Name(type)));
  }
  return primitive_util::PrimitiveTypeSwitch<XlaOp>(
      [&](auto primitive_type_constant) -> XlaOp {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return ConstantR0<NativeT>(builder, static_cast<NativeT>(value));
        }
        return builder->ReportError(
            InvalidArgument("Invalid type for ConstantR0WithType (%s).",
                            PrimitiveType_Name(type)));
      },
      type);
}

// Returns a scalar containing 'value' cast to the same run-time type as
// 'prototype'.
// If 'value' is floating point but 'prototype' is not, or if 'value' is complex
// 'prototype' is not, an error will be returned.
template <typename T>
XlaOp ScalarLike(XlaOp prototype, T value) {
  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return ConstantR0WithType(builder, shape.element_type(), value);
  });
}

// Returns an array or scalar containing copies of `value` cast to the same
// run-type type as `prototype` and broadcast to the same dimensions as
// `prototype`.
//
// If `prototype` is not a scalar or array, returns an error.
template <typename T>
XlaOp FullLike(XlaOp prototype, T value) {
  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    if (ShapeUtil::IsScalar(shape) || shape.IsArray()) {
      return Broadcast(ScalarLike(prototype, value), shape.dimensions());
    } else {
      return InvalidArgument(
          "Prototype shape for BroadcastConstantLike must be a scalar or "
          "array, but was %s",
          shape.ToString());
    }
  });
}

// Returns a scalar with value '0' of 'type'.
XlaOp Zero(XlaBuilder* builder, PrimitiveType type);

// Returns a zero-filled tensor with shape `shape`.
XlaOp Zeros(XlaBuilder* builder, const Shape& shape);

// Returns a zero-filled tensor with the same shape as `prototype`.
XlaOp ZerosLike(XlaOp prototype);

// Returns a scalar with value '1' of 'type'.
XlaOp One(XlaBuilder* builder, PrimitiveType type);

// Returns the machine epsilon for floating-point type `type`, i.e.,
// the difference between 1.0 and the next representable value.
XlaOp Epsilon(XlaBuilder* builder, PrimitiveType type);

// Returns the minimum representable finite or infinite value for 'type'.
// Returns '-inf' for floating-point types.
XlaOp MinValue(XlaBuilder* builder, PrimitiveType type);

// Returns the minimum representable finite value for 'type'. For a floating
// point type, this is equal to -MaxFiniteValue().
XlaOp MinFiniteValue(XlaBuilder* builder, PrimitiveType type);

// Returns the minimum positive normal value for floating-point type `type`.
XlaOp MinPositiveNormalValue(XlaBuilder* builder, PrimitiveType type);

// Returns the maximum representable finite or infinite value for 'type'.
// Returns 'inf' for floating-point types.
XlaOp MaxValue(XlaBuilder* builder, PrimitiveType type);

// Returns the maximum representable finite value for 'type'.
XlaOp MaxFiniteValue(XlaBuilder* builder, PrimitiveType type);

// Returns a nan for the given type.  Only valid for real-valued fp types.
XlaOp NanValue(XlaBuilder* builder, PrimitiveType type);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_CONSTANTS_H_
