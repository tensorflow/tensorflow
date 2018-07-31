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
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

namespace {

xla::XlaOp ArgMinMax(xla::XlaOp input, xla::PrimitiveType output_type, int axis,
                     bool is_min) {
  xla::XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(input));
    xla::XlaOp init_value;
    xla::XlaComputation reducer;
    if (is_min) {
      init_value = xla::MaxValue(builder, input_shape.element_type());
      reducer =
          xla::CreateScalarMinComputation(input_shape.element_type(), builder);
    } else {
      init_value = xla::MinValue(builder, input_shape.element_type());
      reducer =
          xla::CreateScalarMaxComputation(input_shape.element_type(), builder);
    }

    xla::XlaOp input_max = xla::Reduce(input, init_value, reducer,
                                       /*dimensions_to_reduce=*/{axis});
    std::vector<int64> broadcast_dims(xla::ShapeUtil::Rank(input_shape) - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
    std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
    // Compute a mask that has 1s for elements equal to the maximum.
    xla::XlaOp partial_mask = xla::ConvertElementType(
        xla::Eq(input, input_max, broadcast_dims), output_type);

    // In order to make identity elements for a bitwise And, we:
    //   Left shift the 1 to the leftmost bit, yielding 0x10...0
    //   Arithmetic right shift the 1 back to the rightmost bit, yielding
    //   0xFF...F
    int32 bits_in_type =
        xla::ShapeUtil::ByteSizeOfPrimitiveType(output_type) * 8 - 1;
    xla::XlaOp shift_amount =
        xla::ConstantR0WithType(builder, output_type, bits_in_type);
    xla::XlaOp full_mask = xla::ShiftRightArithmetic(
        xla::ShiftLeft(partial_mask, shift_amount), shift_amount);

    // And with the vector [0, 1, 2, ...] to convert each 0xFF...F into its
    // index.

    const int64 axis_size = xla::ShapeUtil::GetDimension(input_shape, axis);
    xla::XlaOp iota = xla::Iota(builder, output_type, axis_size);
    xla::XlaOp product =
        xla::And(full_mask, iota, /*broadcast_dimensions=*/{axis});

    // If there are multiple maximum elements, choose the one with the highest
    // index.
    return xla::Reduce(product, xla::MinValue(builder, output_type),
                       xla::CreateScalarMaxComputation(output_type, builder),
                       /*dimensions_to_reduce=*/{axis});
  });
}

}  // namespace

xla::XlaOp XlaHelpers::Zero(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return xla::ConstantLiteral(b, xla::LiteralUtil::Zero(type));
}

xla::XlaOp XlaHelpers::One(xla::XlaBuilder* b, DataType data_type) {
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(data_type, &type));
  return xla::ConstantLiteral(b, xla::LiteralUtil::One(type));
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

xla::XlaOp XlaHelpers::ArgMax(xla::XlaOp input, xla::PrimitiveType output_type,
                              int axis) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/false);
}

xla::XlaOp XlaHelpers::ArgMin(xla::XlaOp input, xla::PrimitiveType output_type,
                              int axis) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/true);
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
  xla::XlaOp one_hot_bool = xla::Eq(
      indices, xla::ConstantLiteral(builder, linspace_literal), broadcast_dims);

  // Selects the user-provided off_value and on_value values.
  *one_hot = xla::Select(one_hot_bool,
                         xla::Broadcast(on_value, output_shape.dim_sizes()),
                         xla::Broadcast(off_value, output_shape.dim_sizes()));
  return Status::OK();
}

DataType XlaHelpers::SumAccumulationType(const DataType& dtype) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
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
  return xla::ConvertElementType(operand, convert_to);
}

}  // end namespace tensorflow
