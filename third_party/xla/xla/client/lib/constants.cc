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

#include "xla/client/lib/constants.h"

#include <limits>

#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {

XlaOp Zero(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::Zero(type));
}

XlaOp Zeros(XlaBuilder* builder, const Shape& shape) {
  return Broadcast(Zero(builder, shape.element_type()), shape.dimensions());
}

XlaOp ZerosLike(XlaOp prototype) {
  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return Zeros(builder, shape);
  });
}

XlaOp One(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::One(type));
}

XlaOp Epsilon(XlaBuilder* builder, PrimitiveType type) {
  return primitive_util::PrimitiveTypeSwitch<XlaOp>(
      [&](auto primitive_type_constant) -> XlaOp {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return ConstantR0<NativeT>(builder,
                                     std::numeric_limits<NativeT>::epsilon());
        }
        return builder->ReportError(InvalidArgument(
            "Invalid type for Epsilon (%s).", PrimitiveType_Name(type)));
      },
      type);
}

XlaOp MinValue(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::MinValue(type));
}

XlaOp MinFiniteValue(XlaBuilder* builder, PrimitiveType type) {
  return primitive_util::PrimitiveTypeSwitch<XlaOp>(
      [&](auto primitive_type_constant) -> XlaOp {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return ConstantR0<NativeT>(builder,
                                     std::numeric_limits<NativeT>::lowest());
        }
        return MinValue(builder, type);
      },
      type);
}

XlaOp MinPositiveNormalValue(XlaBuilder* builder, PrimitiveType type) {
  return primitive_util::PrimitiveTypeSwitch<XlaOp>(
      [&](auto primitive_type_constant) -> XlaOp {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return ConstantR0<NativeT>(builder,
                                     std::numeric_limits<NativeT>::min());
        }
        return builder->ReportError(
            InvalidArgument("Invalid type for MinPositiveNormalValue (%s).",
                            PrimitiveType_Name(type)));
      },
      type);
}

XlaOp MaxValue(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::MaxValue(type));
}

XlaOp MaxFiniteValue(XlaBuilder* builder, PrimitiveType type) {
  return primitive_util::PrimitiveTypeSwitch<XlaOp>(
      [&](auto primitive_type_constant) -> XlaOp {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return ConstantR0<NativeT>(builder,
                                     std::numeric_limits<NativeT>::max());
        }
        return MaxValue(builder, type);
      },
      type);
}

XlaOp NanValue(XlaBuilder* builder, PrimitiveType type) {
  return primitive_util::PrimitiveTypeSwitch<XlaOp>(
      [&](auto primitive_type_constant) -> XlaOp {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return ConstantR0<NativeT>(builder,
                                     std::numeric_limits<NativeT>::quiet_NaN());
        }
        return builder->ReportError(InvalidArgument(
            "Invalid type for NanValue (%s).", PrimitiveType_Name(type)));
      },
      type);
}

}  // namespace xla
