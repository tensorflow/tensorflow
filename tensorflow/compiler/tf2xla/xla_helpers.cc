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

// This file defines helper routines for Tla JIT compilation.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

xla::ComputationDataHandle XlaHelpers::MinValue(xla::ComputationBuilder* b,
                                                DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::LiteralUtil::MinValue(type));
}

xla::ComputationDataHandle XlaHelpers::MaxValue(xla::ComputationBuilder* b,
                                                DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::LiteralUtil::MaxValue(type));
}

xla::ComputationDataHandle XlaHelpers::Zero(xla::ComputationBuilder* b,
                                            DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::LiteralUtil::Zero(type));
}

xla::ComputationDataHandle XlaHelpers::One(xla::ComputationBuilder* b,
                                           DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::LiteralUtil::One(type));
}

xla::ComputationDataHandle XlaHelpers::IntegerLiteral(
    xla::ComputationBuilder* b, DataType data_type, int64 value) {
  xla::Literal literal;
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  switch (type) {
    case xla::U8:
      literal = *xla::LiteralUtil::CreateR0<uint8>(value);
      break;
    case xla::U32:
      literal = *xla::LiteralUtil::CreateR0<uint32>(value);
      break;
    case xla::U64:
      literal = *xla::LiteralUtil::CreateR0<uint64>(value);
      break;
    case xla::S8:
      literal = *xla::LiteralUtil::CreateR0<int8>(value);
      break;
    case xla::S32:
      literal = *xla::LiteralUtil::CreateR0<int32>(value);
      break;
    case xla::S64:
      literal = *xla::LiteralUtil::CreateR0<int64>(value);
      break;
    case xla::F32:
      literal = *xla::LiteralUtil::CreateR0<float>(value);
      break;
    case xla::F64:
      literal = *xla::LiteralUtil::CreateR0<double>(value);
      break;
    case xla::PRED:
      LOG(FATAL) << "pred element type is not integral";
    case xla::S16:
    case xla::U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case xla::F16:
      LOG(FATAL) << "f16 literals not yet implemented";
    case xla::TUPLE:
      LOG(FATAL) << "tuple element type is not integral";
    case xla::OPAQUE:
      LOG(FATAL) << "opaque element type is not integral";
    default:
      LOG(FATAL) << "unhandled element type " << type;
  }
  return b->ConstantLiteral(literal);
}

xla::ComputationDataHandle XlaHelpers::FloatLiteral(xla::ComputationBuilder* b,
                                                    DataType data_type,
                                                    double value) {
  xla::Literal literal;
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  switch (type) {
    case xla::F32:
      return b->ConstantR0<float>(static_cast<float>(value));
      break;
    case xla::F64:
      return b->ConstantR0<double>(value);
      break;
    default:
      LOG(FATAL) << "unhandled element type " << type;
  }
}

/* static */ Status XlaHelpers::ReshapeLiteral(
    const xla::Literal& input, gtl::ArraySlice<int64> dimensions,
    xla::Literal* output) {
  if (xla::ShapeUtil::IsTuple(input.shape())) {
    return errors::InvalidArgument("ReshapeLiteral does not support tuples.");
  }
  xla::Shape shape =
      xla::ShapeUtil::MakeShape(input.shape().element_type(), dimensions);
  int64 elements_before = xla::ShapeUtil::ElementsIn(input.shape());
  int64 elements_after = xla::ShapeUtil::ElementsIn(shape);
  if (elements_before != elements_after) {
    return errors::InvalidArgument(
        "Shapes before and after ReshapeLiteral have different numbers of "
        "elements.");
  }

  *output = input;
  output->mutable_shape()->Swap(&shape);
  return Status::OK();
}

}  // end namespace tensorflow
