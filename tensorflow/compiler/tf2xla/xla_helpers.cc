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

// This file defines helper routines for XLA compilation.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

namespace {

Status ArgMinMax(xla::XlaBuilder* builder, XlaOpKernelContext* ctx,
                 const xla::XlaOp& input, const TensorShape& input_shape,
                 DataType input_type, DataType output_type, int axis,
                 bool is_min, xla::XlaOp* argminmax) {
  xla::XlaOp init_value;
  const xla::XlaComputation* reducer;
  if (is_min) {
    init_value = XlaHelpers::MaxValue(builder, input_type);
    reducer = ctx->GetOrCreateMin(input_type);
  } else {
    init_value = XlaHelpers::MinValue(builder, input_type);
    reducer = ctx->GetOrCreateMax(input_type);
  }

  xla::PrimitiveType xla_output_type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(output_type, &xla_output_type));

  xla::XlaOp input_max = builder->Reduce(input, init_value, *reducer,
                                         /*dimensions_to_reduce=*/{axis});
  std::vector<int64> broadcast_dims(input_shape.dims() - 1);
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
  // Compute a mask that has 1s for elements equal to the maximum.
  xla::XlaOp partial_mask = builder->ConvertElementType(
      builder->Eq(input, input_max, broadcast_dims), xla_output_type);

  // In order to make identity elements for a bitwise And, we:
  //   Left shift the 1 to the leftmost bit, yielding 0x10...0
  //   Arithmetic right shift the 1 back to the rightmost bit, yielding
  //   0xFF...F
  int32 bits_in_type =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(xla_output_type) * 8 - 1;
  xla::XlaOp shift_amount =
      XlaHelpers::IntegerLiteral(builder, output_type, bits_in_type);
  xla::XlaOp full_mask = builder->ShiftRightArithmetic(
      builder->ShiftLeft(partial_mask, shift_amount), shift_amount);

  // And with the vector [0, 1, 2, ...] to convert each 0xFF...F into its
  // index.
  xla::XlaOp iota;

  const int64 axis_size = input_shape.dim_size(axis);
  TF_RETURN_IF_ERROR(XlaHelpers::Iota(builder, output_type, axis_size, &iota));
  xla::XlaOp product =
      builder->And(full_mask, iota, /*broadcast_dimensions=*/{axis});

  // If there are multiple maximum elements, choose the one with the highest
  // index.
  xla::XlaOp output =
      builder->Reduce(product, XlaHelpers::MinValue(builder, output_type),
                      *ctx->GetOrCreateMax(output_type),
                      /*dimensions_to_reduce=*/{axis});
  *argminmax = output;
  return Status::OK();
}

}  // namespace

xla::XlaOp XlaHelpers::MinValue(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::MinValue(type));
}

xla::XlaOp XlaHelpers::MinFiniteValue(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  switch (type) {
    case xla::F16:
      return b->ConstantR0<Eigen::half>(
          Eigen::NumTraits<Eigen::half>::lowest());
    case xla::BF16:
      return b->ConstantR0<bfloat16>(bfloat16::lowest());
    case xla::F32:
      return b->ConstantR0<float>(-std::numeric_limits<float>::max());
    case xla::F64:
      return b->ConstantR0<double>(-std::numeric_limits<double>::max());
    default:
      return b->ConstantLiteral(xla::Literal::MinValue(type));
  }
}

xla::XlaOp XlaHelpers::MaxValue(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::MaxValue(type));
}

xla::XlaOp XlaHelpers::MaxFiniteValue(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  switch (type) {
    case xla::F16:
      return b->ConstantR0<Eigen::half>(
          Eigen::NumTraits<Eigen::half>::highest());
    case xla::BF16:
      return b->ConstantR0<bfloat16>(bfloat16::highest());
    case xla::F32:
      return b->ConstantR0<float>(std::numeric_limits<float>::max());
    case xla::F64:
      return b->ConstantR0<double>(std::numeric_limits<double>::max());
    default:
      return b->ConstantLiteral(xla::Literal::MaxValue(type));
  }
}

xla::XlaOp XlaHelpers::Zero(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::Zero(type));
}

xla::XlaOp XlaHelpers::One(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return b->ConstantLiteral(xla::Literal::One(type));
}

xla::XlaOp XlaHelpers::Epsilon(xla::XlaBuilder* b, DataType data_type) {
  switch (data_type) {
    case DT_HALF:
      return b->ConstantR0<Eigen::half>(
          static_cast<Eigen::half>(Eigen::NumTraits<Eigen::half>::epsilon()));
    case DT_BFLOAT16:
      return b->ConstantR0<bfloat16>(bfloat16::epsilon());
    case DT_FLOAT:
      return b->ConstantR0<float>(std::numeric_limits<float>::epsilon());
    case DT_DOUBLE:
      return b->ConstantR0<double>(std::numeric_limits<double>::epsilon());
    default:
      LOG(FATAL) << "Unsupported type in XlaHelpers::Epsilon: "
                 << DataTypeString(data_type);
  }
}

xla::XlaOp XlaHelpers::IntegerLiteral(xla::XlaBuilder* b, DataType data_type,
                                      int64 value) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return ::tensorflow::IntegerLiteral(b, type, value);
}

xla::XlaOp XlaHelpers::FloatLiteral(xla::XlaBuilder* b, DataType data_type,
                                    double value) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return ::tensorflow::FloatLiteral(b, type, value);
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

  *output = input.Clone();
  output->mutable_shape_do_not_use()->Swap(&shape);
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

Status XlaHelpers::ArgMax(xla::XlaBuilder* builder, XlaOpKernelContext* ctx,
                          const xla::XlaOp& input,
                          const TensorShape& input_shape, DataType input_type,
                          DataType output_type, int axis, xla::XlaOp* argmax) {
  return ArgMinMax(builder, ctx, input, input_shape, input_type, output_type,
                   axis, /*is_min=*/false, argmax);
}

Status XlaHelpers::ArgMin(xla::XlaBuilder* builder, XlaOpKernelContext* ctx,
                          const xla::XlaOp& input,
                          const TensorShape& input_shape, DataType input_type,
                          DataType output_type, int axis, xla::XlaOp* argmin) {
  return ArgMinMax(builder, ctx, input, input_shape, input_type, output_type,
                   axis, /*is_min=*/true, argmin);
}

Status XlaHelpers::Iota(xla::XlaBuilder* builder, DataType dtype, int64 size,
                        xla::XlaOp* iota) {
  TensorShape linspace_shape({size});
  Tensor linspace;
  switch (dtype) {
    case DT_UINT8:
      linspace = MakeLinspaceTensor<uint8>(linspace_shape, size);
      break;
    case DT_INT32:
      linspace = MakeLinspaceTensor<int32>(linspace_shape, size);
      break;
    case DT_INT64:
      linspace = MakeLinspaceTensor<int64>(linspace_shape, size);
      break;
    default:
      return errors::InvalidArgument("Invalid argument type ",
                                     DataTypeString(dtype));
  }
  xla::BorrowingLiteral linspace_literal;
  TF_RETURN_IF_ERROR(HostTensorToBorrowingLiteral(linspace, &linspace_literal));

  *iota = builder->ConstantLiteral(linspace_literal);
  return Status::OK();
}

Status XlaHelpers::OneHot(xla::XlaBuilder* builder, int64 depth, int axis,
                          DataType index_type, const TensorShape& indices_shape,
                          const xla::XlaOp& indices, const xla::XlaOp& on_value,
                          const xla::XlaOp& off_value, xla::XlaOp* one_hot) {
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

  xla::BorrowingLiteral linspace_literal;
  TF_RETURN_IF_ERROR(HostTensorToBorrowingLiteral(linspace, &linspace_literal));

  // Broadcast the linspace constant across the indices along the new axis,
  // and test equality at each position.
  std::vector<int64> broadcast_dims(indices_shape.dims());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
  xla::XlaOp one_hot_bool = builder->Eq(
      indices, builder->ConstantLiteral(linspace_literal), broadcast_dims);

  // Selects the user-provided off_value and on_value values.
  *one_hot = builder->Select(
      one_hot_bool, builder->Broadcast(on_value, output_shape.dim_sizes()),
      builder->Broadcast(off_value, output_shape.dim_sizes()));
  return Status::OK();
}

DataType XlaHelpers::SumAccumulationType(const DataType& dtype) {
  if (dtype == DT_BFLOAT16 || dtype == DT_HALF) {
    return DT_FLOAT;
  }
  return dtype;
}

xla::XlaOp XlaHelpers::ConvertElementType(xla::XlaBuilder* const builder,
                                          const xla::XlaOp& operand,
                                          const DataType new_element_type) {
  xla::PrimitiveType convert_to;
  TF_CHECK_OK(DataTypeToPrimitiveType(new_element_type, &convert_to));
  return builder->ConvertElementType(operand, convert_to);
}

}  // end namespace tensorflow
