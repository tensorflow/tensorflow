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

#include "tensorflow/compiler/tf2xla/lib/util.h"

#include "absl/log/log.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {

xla::XlaOp Zeros(xla::XlaBuilder* builder, const xla::Shape& shape) {
  return xla::Broadcast(
      xla::ConstantLiteral(builder,
                           xla::LiteralUtil::Zero(shape.element_type())),
      shape.dimensions());
}

xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value) {
  return xla::primitive_util::PrimitiveTypeSwitch<xla::XlaOp>(
      [&](auto primitive_type_constant) -> xla::XlaOp {
        if constexpr (xla::primitive_util::IsFloatingPointType(
                          primitive_type_constant) ||
                      xla::primitive_util::IsComplexType(
                          primitive_type_constant)) {
          using NativeT =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          return xla::ConstantR0<NativeT>(builder, static_cast<NativeT>(value));
        }
        LOG(FATAL) << "unhandled element type " << type;
      },
      type);
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64_t value) {
  xla::Literal literal = xla::primitive_util::PrimitiveTypeSwitch<xla::Literal>(
      [&](auto primitive_type_constant) -> xla::Literal {
        if constexpr (xla::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using NativeT =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          return xla::LiteralUtil::CreateR0<NativeT>(
              static_cast<NativeT>(value));
        }
        LOG(FATAL) << "unhandled element type " << type;
      },
      type);
  return xla::ConstantLiteral(builder, literal);
}

}  // namespace tensorflow
