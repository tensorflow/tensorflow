/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"

namespace tensorflow {

namespace {

using errors::InvalidArgument;

absl::StatusOr<TensorShape> CalculateOutputShape(const TensorShape& lhs_shape,
                                                 const TensorShape& rhs_shape) {
  if (lhs_shape.dims() == 0) {
    return rhs_shape;
  } else if (rhs_shape.dims() == 0) {
    return lhs_shape;
  }

  std::vector<int64_t> reversed_output_shape;
  int l_dim = lhs_shape.dims() - 1;
  int r_dim = rhs_shape.dims() - 1;
  while (l_dim >= 0 || r_dim >= 0) {
    const int64_t l_dim_size = l_dim >= 0 ? lhs_shape.dim_size(l_dim) : 1;
    const int64_t r_dim_size = r_dim >= 0 ? rhs_shape.dim_size(r_dim) : 1;
    if (l_dim_size != 1 && r_dim_size != 1 && l_dim_size != r_dim_size) {
      return InvalidArgument("Cannot Add tensors of shapes: ",
                             lhs_shape.DebugString(), rhs_shape.DebugString());
    }
    reversed_output_shape.push_back(l_dim_size == 1 ? r_dim_size : l_dim_size);
    --l_dim;
    --r_dim;
  }
  absl::c_reverse(reversed_output_shape);
  TensorShape output_shape;
  TF_RETURN_IF_ERROR(
      TensorShape::BuildTensorShape(reversed_output_shape, &output_shape));
  return output_shape;
}

template <typename T>
void QuantizedAdd(const Tensor& lhs, const Tensor& rhs,
                  const Tensor& output_zero_points,
                  int output_quantization_min_val,
                  int output_quantization_max_val, int lhs_quantization_axis,
                  int rhs_quantization_axis, int output_quantizaiton_axis,
                  Tensor& output) {
  const T* lhs_data = lhs.flat<T>().data();
  const T* rhs_data = rhs.flat<T>().data();
  T* output_data = output.flat<T>().data();

  const int32* output_zero_points_data =
      output_zero_points.flat<int32>().data();

  for (int64_t output_idx = 0; output_idx < output.NumElements();
       ++output_idx) {
    int64_t output_idx_remain = output_idx;
    int64_t lhs_idx = 0;
    int64_t rhs_idx = 0;
    int64_t lhs_inner_dim_size = 1;
    int64_t rhs_inner_dim_size = 1;
    int64_t output_zero_points_idx_of_quantization_axis = 0;
    for (int output_dim = output.dims() - 1; output_dim >= 0; --output_dim) {
      const int64_t output_idx_of_dim =
          output_idx_remain % output.dim_size(output_dim);
      output_idx_remain /= output.dim_size(output_dim);
      if (output_quantizaiton_axis == output_dim) {
        output_zero_points_idx_of_quantization_axis = output_idx_of_dim;
      }

      const int lhs_dim = output_dim - (output.dims() - lhs.dims());
      if (lhs_dim >= 0) {
        const int64_t lhs_idx_of_dim =
            lhs.dim_size(lhs_dim) == 1 ? 0 : output_idx_of_dim;
        lhs_idx += lhs_idx_of_dim * lhs_inner_dim_size;
        lhs_inner_dim_size *= lhs.dim_size(lhs_dim);
      }
      const int rhs_dim = output_dim - (output.dims() - rhs.dims());
      if (rhs_dim >= 0) {
        const int64_t rhs_idx_of_dim =
            rhs.dim_size(rhs_dim) == 1 ? 0 : output_idx_of_dim;
        rhs_idx += rhs_idx_of_dim * rhs_inner_dim_size;
        rhs_inner_dim_size *= rhs.dim_size(rhs_dim);
      }
    }

    const int32_t output_zero_point =
        output_zero_points_data[output_zero_points_idx_of_quantization_axis];

    const int32_t unclamped = static_cast<int32_t>(lhs_data[lhs_idx]) +
                              static_cast<int32_t>(rhs_data[rhs_idx]) +
                              output_zero_point;
    output_data[output_idx] = static_cast<T>(std::clamp(
        unclamped, output_quantization_min_val, output_quantization_max_val));
  }
}

template <typename T>
absl::Status EvalQuantizedAdd(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    const Tensor& lhs_scales, const Tensor& lhs_zero_points,
    const Tensor& rhs_scales, const Tensor& rhs_zero_points,
    const Tensor& output_scales, const Tensor& output_zero_points,
    int output_quantization_min_val, int output_quantization_max_val,
    int lhs_quantization_axis, int rhs_quantization_axis,
    int output_quantization_axis, Tensor& output) {
  const DataType dtype = DataTypeToEnum<T>::v();

  Tensor zeros_of_output_scales_shape;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, output_scales.shape(),
                                            &zeros_of_output_scales_shape));
  zeros_of_output_scales_shape.flat<int32_t>().setZero();

  Tensor lhs_requantized;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(dtype, lhs.shape(), &lhs_requantized));
  const int lhs_requantize_output_quantization_axis =
      output_quantization_axis == -1 ? -1 : lhs_quantization_axis;
  TF_RETURN_IF_ERROR(EvalRequantize<T, T>(
      context, lhs, lhs_scales, lhs_zero_points, output_scales,
      /*output_zero_points=*/zeros_of_output_scales_shape,
      lhs_quantization_axis, lhs_requantize_output_quantization_axis,
      /*quantization_min_val=*/std::numeric_limits<T>::min(),
      /*quantization_max_val=*/std::numeric_limits<T>::max(), lhs_requantized));

  Tensor rhs_requantized;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(dtype, rhs.shape(), &rhs_requantized));
  TF_RETURN_IF_ERROR(EvalRequantize<T, T>(
      context, rhs, rhs_scales, rhs_zero_points, output_scales,
      /*output_zero_points=*/zeros_of_output_scales_shape,
      rhs_quantization_axis, output_quantization_axis,
      /*quantization_min_val=*/std::numeric_limits<T>::min(),
      /*quantization_max_val=*/std::numeric_limits<T>::max(), rhs_requantized));

  QuantizedAdd<T>(lhs_requantized, rhs_requantized, output_zero_points,
                  output_quantization_min_val, output_quantization_max_val,
                  lhs_quantization_axis, rhs_quantization_axis,
                  output_quantization_axis, output);

  return absl::OkStatus();
}

}  // namespace

template <typename T>
class UniformQuantizedAddOp : public OpKernel {
 public:
  explicit UniformQuantizedAddOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, (std::is_same<T, qint32>()),
                InvalidArgument("Unsupported operand type."));

    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_min_val",
                                             &output_quantization_min_val_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_max_val",
                                             &output_quantization_max_val_));

    OP_REQUIRES_OK(context, context->GetAttr("lhs_quantization_axis",
                                             &lhs_quantization_axis_));
    OP_REQUIRES_OK(context, context->GetAttr("rhs_quantization_axis",
                                             &rhs_quantization_axis_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_axis",
                                             &output_quantization_axis_));

    OP_REQUIRES(
        context,
        (lhs_quantization_axis_ >= -1 && rhs_quantization_axis_ >= -1 &&
         output_quantization_axis_ >= -1),
        InvalidArgument("lhs, rhs and output quantization_axis must be -1 or "
                        "within [0, dims)"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& lhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& lhs_scales = context->input(2);
    const Tensor& lhs_zero_points = context->input(3);
    const Tensor& rhs_scales = context->input(4);
    const Tensor& rhs_zero_points = context->input(5);
    const Tensor& output_scales = context->input(6);
    const Tensor& output_zero_points = context->input(7);

    OP_REQUIRES_OK(
        context, QuantizationAxisAndShapeValid(lhs.shape(), lhs_scales.shape(),
                                               lhs_zero_points.shape(),
                                               lhs_quantization_axis_));
    OP_REQUIRES_OK(
        context, QuantizationAxisAndShapeValid(rhs.shape(), rhs_scales.shape(),
                                               rhs_zero_points.shape(),
                                               rhs_quantization_axis_));

    auto output_shape_status = CalculateOutputShape(lhs.shape(), rhs.shape());
    OP_REQUIRES_OK(context, output_shape_status.status());

    const auto& output_shape = output_shape_status.value();
    OP_REQUIRES_OK(context,
                   QuantizationAxisAndShapeValid(
                       output_shape, output_scales.shape(),
                       output_zero_points.shape(), output_quantization_axis_));

    OP_REQUIRES(
        context,
        (!(lhs_quantization_axis_ >= 0 && output_quantization_axis_ >= 0) ||
         (lhs.dims() - lhs_quantization_axis_ ==
          output_shape.dims() - output_quantization_axis_)),
        InvalidArgument("If lhs and output is both per-axis quantized, the "
                        "quantization axis must match."));
    OP_REQUIRES(
        context,
        (!(rhs_quantization_axis_ >= 0 && output_quantization_axis_ >= 0) ||
         (rhs.dims() - rhs_quantization_axis_ ==
          output_shape.dims() - output_quantization_axis_)),
        InvalidArgument("If rhs and output is both per-axis quantized, the "
                        "quantization axis must match."));

    OP_REQUIRES(context,
                (AllElementsPositive<float>(lhs_scales) &&
                 AllElementsPositive<float>(rhs_scales) &&
                 AllElementsPositive<float>(output_scales)),
                InvalidArgument(
                    "lhs/rhs/output scales elements must be all positive."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    OP_REQUIRES_OK(
        context, EvalQuantizedAdd<T>(
                     context, lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales,
                     rhs_zero_points, output_scales, output_zero_points,
                     output_quantization_min_val_, output_quantization_max_val_,
                     lhs_quantization_axis_, rhs_quantization_axis_,
                     output_quantization_axis_, *output));
  }

 private:
  int lhs_quantization_axis_;
  int rhs_quantization_axis_;
  int output_quantization_axis_;

  int output_quantization_min_val_;
  int output_quantization_max_val_;
};

REGISTER_KERNEL_BUILDER(
    Name("UniformQuantizedAdd").Device(DEVICE_CPU).TypeConstraint<qint32>("T"),
    UniformQuantizedAddOp<qint32>);

}  // namespace tensorflow
