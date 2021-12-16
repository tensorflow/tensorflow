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

#include <algorithm>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

xla::XlaOp Zeros(xla::XlaBuilder* builder, const xla::Shape& shape) {
  return xla::Broadcast(
      xla::ConstantLiteral(builder,
                           xla::LiteralUtil::Zero(shape.element_type())),
      shape.dimensions());
}

xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value) {
  switch (type) {
    case xla::F16:
      return xla::ConstantR0<xla::half>(builder, static_cast<xla::half>(value));
      break;
    case xla::BF16:
      return xla::ConstantR0<bfloat16>(builder, static_cast<bfloat16>(value));
      break;
    case xla::F32:
      return xla::ConstantR0<float>(builder, static_cast<float>(value));
      break;
    case xla::F64:
      return xla::ConstantR0<double>(builder, value);
      break;
    case xla::C64:
      return xla::ConstantR0<xla::complex64>(builder, value);
      break;
    case xla::C128:
      return xla::ConstantR0<xla::complex128>(builder, value);
      break;
    default:
      LOG(FATAL) << "unhandled element type " << type;
  }
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64_t value) {
  xla::Literal literal;
  switch (type) {
    case xla::U8:
      literal = xla::LiteralUtil::CreateR0<uint8>(value);
      break;
    case xla::U16:
      literal = xla::LiteralUtil::CreateR0<uint16>(value);
      break;
    case xla::U32:
      literal = xla::LiteralUtil::CreateR0<uint32>(value);
      break;
    case xla::U64:
      literal = xla::LiteralUtil::CreateR0<uint64>(value);
      break;
    case xla::S8:
      literal = xla::LiteralUtil::CreateR0<int8>(value);
      break;
    case xla::S16:
      literal = xla::LiteralUtil::CreateR0<int16>(value);
      break;
    case xla::S32:
      literal = xla::LiteralUtil::CreateR0<int32>(value);
      break;
    case xla::S64:
      literal = xla::LiteralUtil::CreateR0<int64_t>(value);
      break;
    case xla::F32:
      literal = xla::LiteralUtil::CreateR0<float>(value);
      break;
    case xla::F64:
      literal = xla::LiteralUtil::CreateR0<double>(value);
      break;
    case xla::C64:
      literal = xla::LiteralUtil::CreateR0<complex64>(value);
      break;
    case xla::C128:
      literal = xla::LiteralUtil::CreateR0<complex128>(value);
      break;
    case xla::PRED:
      LOG(FATAL) << "pred element type is not integral";
    case xla::BF16:
      literal =
          xla::LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(value));
      break;
    case xla::F16:
      literal =
          xla::LiteralUtil::CreateR0<xla::half>(static_cast<xla::half>(value));
      break;
    case xla::TUPLE:
      LOG(FATAL) << "tuple element type is not integral";
    case xla::OPAQUE_TYPE:
      LOG(FATAL) << "opaque element type is not integral";
    default:
      LOG(FATAL) << "unhandled element type " << type;
  }
  return xla::ConstantLiteral(builder, literal);
}

std::vector<int64_t> ConcatVectors(absl::Span<const int64_t> xs,
                                   absl::Span<const int64_t> ys) {
  std::vector<int64_t> output(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), output.begin());
  std::copy(ys.begin(), ys.end(), output.begin() + xs.size());
  return output;
}

}  // namespace tensorflow
