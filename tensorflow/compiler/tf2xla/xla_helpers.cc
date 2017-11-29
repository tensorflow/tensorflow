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
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

xla::ComputationDataHandle XlaHelpers::MinValue(xla::ComputationBuilder* b,
                                                DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::MinValue(type));
}

xla::ComputationDataHandle XlaHelpers::MaxValue(xla::ComputationBuilder* b,
                                                DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::MaxValue(type));
}

xla::ComputationDataHandle XlaHelpers::Zero(xla::ComputationBuilder* b,
                                            DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::Zero(type));
}

xla::ComputationDataHandle XlaHelpers::One(xla::ComputationBuilder* b,
                                           DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::One(type));
}

xla::ComputationDataHandle XlaHelpers::IntegerLiteral(
    xla::ComputationBuilder* b, DataType data_type, int64 value) {
  xla::Literal literal;
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  switch (type) {
    case xla::U8:
      literal = *xla::Literal::CreateR0<uint8>(value);
      break;
    case xla::U32:
      literal = *xla::Literal::CreateR0<uint32>(value);
      break;
    case xla::U64:
      literal = *xla::Literal::CreateR0<uint64>(value);
      break;
    case xla::S8:
      literal = *xla::Literal::CreateR0<int8>(value);
      break;
    case xla::S32:
      literal = *xla::Literal::CreateR0<int32>(value);
      break;
    case xla::S64:
      literal = *xla::Literal::CreateR0<int64>(value);
      break;
    case xla::F32:
      literal = *xla::Literal::CreateR0<float>(value);
      break;
    case xla::F64:
      literal = *xla::Literal::CreateR0<double>(value);
      break;
    case xla::PRED:
      LOG(FATAL) << "pred element type is not integral";
    case xla::S16:
    case xla::U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case xla::F16:
      literal =
          *xla::Literal::CreateR0<xla::half>(static_cast<xla::half>(value));
      break;
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
    case xla::F16:
      return b->ConstantR0<xla::half>(static_cast<xla::half>(value));
      break;
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

template <typename T>
static Tensor MakeLinspaceTensor(const TensorShape& shape, int64 depth) {
  Tensor linspace(DataTypeToEnum<T>::v(), shape);
  auto linspace_flat = linspace.flat<T>();
  for (int64 i = 0; i < depth; ++i) {
    linspace_flat(i) = i;
  }
  return linspace;
}

Status XlaHelpers::OneHot(xla::ComputationBuilder* builder, int64 depth,
                          int axis, DataType index_type,
                          const TensorShape& indices_shape,
                          const xla::ComputationDataHandle& indices,
                          const xla::ComputationDataHandle& on_value,
                          const xla::ComputationDataHandle& off_value,
                          xla::ComputationDataHandle* one_hot) {
  const int indices_dims = indices_shape.dims();
  const int output_dims = indices_dims + 1;

  TensorShape output_shape = indices_shape;
  output_shape.InsertDim(axis, depth);

  // Build a Tensor populated with values 0, 1, 2, ... depth.
  std::vector<int64> linspace_dims(output_dims, 1);
  linspace_dims[axis] = depth;
  TensorShape linspace_shape(linspace_dims);
  Tensor linspace;
  switch (index_type) {
    case DT_UINT8:
      linspace = MakeLinspaceTensor<uint8>(linspace_shape, depth);
      break;
    case DT_INT32:
      linspace = MakeLinspaceTensor<int32>(linspace_shape, depth);
      break;
    case DT_INT64:
      linspace = MakeLinspaceTensor<int64>(linspace_shape, depth);
      break;
    default:
      return errors::InvalidArgument("Invalid argument type ",
                                     DataTypeString(index_type));
  }
  xla::Literal linspace_literal;
  TF_RETURN_IF_ERROR(HostTensorToLiteral(linspace, &linspace_literal));

  // Broadcast the linspace constant across the indices along the new axis,
  // and test equality at each position.
  std::vector<int64> broadcast_dims(indices_shape.dims());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
  xla::ComputationDataHandle one_hot_bool = builder->Eq(
      indices, builder->ConstantLiteral(linspace_literal), broadcast_dims);

  // Selects the user-provided off_value and on_value values.
  *one_hot = builder->Select(
      one_hot_bool, builder->Broadcast(on_value, output_shape.dim_sizes()),
      builder->Broadcast(off_value, output_shape.dim_sizes()));
  return Status::OK();
}

xla::ComputationDataHandle XlaHelpers::PadWithZeros(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& x,
    int count) {
  xla::ComputationDataHandle zero = builder->ConstantR1<int32>({0});
  std::vector<xla::ComputationDataHandle> xs(count + 1, zero);
  xs[0] = builder->Reshape(x, {1});
  return builder->ConcatInDim(xs, 0);
}

}  // end namespace tensorflow
