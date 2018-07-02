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

#include "tensorflow/compiler/xla/client/lib/constants.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp Zero(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, Literal::Zero(type));
}

XlaOp Zeros(XlaBuilder* builder, const Shape& shape) {
  return Broadcast(Zero(builder, shape.element_type()),
                   AsInt64Slice(shape.dimensions()));
}

XlaOp ZerosLike(XlaOp prototype) {
  XlaBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return Zeros(builder, shape);
  });
}

XlaOp One(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, Literal::One(type));
}

XlaOp Epsilon(XlaBuilder* builder, PrimitiveType type) {
  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(
          builder,
          static_cast<Eigen::half>(Eigen::NumTraits<Eigen::half>::epsilon()));
    case BF16:
      return ConstantR0<bfloat16>(builder, bfloat16::epsilon());
    case F32:
      return ConstantR0<float>(builder, std::numeric_limits<float>::epsilon());
    case F64:
      return ConstantR0<double>(builder,
                                std::numeric_limits<double>::epsilon());
    default:
      return builder->ReportError(InvalidArgument(
          "Invalid type for Epsilon (%s).", PrimitiveType_Name(type).c_str()));
  }
}

XlaOp MinValue(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, Literal::MinValue(type));
}

XlaOp MinFiniteValue(XlaBuilder* builder, PrimitiveType type) {
  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(builder,
                                     Eigen::NumTraits<Eigen::half>::lowest());
    case BF16:
      return ConstantR0<bfloat16>(builder, bfloat16::lowest());
    case F32:
      return ConstantR0<float>(builder, -std::numeric_limits<float>::max());
    case F64:
      return ConstantR0<double>(builder, -std::numeric_limits<double>::max());
    default:
      return MinValue(builder, type);
  }
}

XlaOp MaxValue(XlaBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, Literal::MaxValue(type));
}

XlaOp MaxFiniteValue(XlaBuilder* builder, PrimitiveType type) {
  switch (type) {
    case F16:
      return ConstantR0<Eigen::half>(builder,
                                     Eigen::NumTraits<Eigen::half>::highest());
    case BF16:
      return ConstantR0<bfloat16>(builder, bfloat16::highest());
    case F32:
      return ConstantR0<float>(builder, std::numeric_limits<float>::max());
    case F64:
      return ConstantR0<double>(builder, std::numeric_limits<double>::max());
    default:
      return MaxValue(builder, type);
  }
}

}  // namespace xla
