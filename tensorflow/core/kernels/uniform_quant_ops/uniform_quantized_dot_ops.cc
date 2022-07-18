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
#include <cmath>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

inline bool IsFinite(float x) { return !std::isinf(x) && !std::isnan(x); }

// Returns if all elements in given tensors are positive.
template <typename T>
bool AllElementsPositive(const Tensor& tensor) {
  auto* tensor_data = tensor.flat<T>().data();
  return std::all_of(tensor_data, tensor_data + tensor.NumElements(),
                     [](T v) { return v > 0; });
}

// Given lhs and rhs shapes, returns if the shapes are valid for 2D X 2D dot.
Status DotInputShapeValid(const TensorShape& lhs_shape,
                          const TensorShape& rhs_shape) {
  if (lhs_shape.dims() != 2) {
    return InvalidArgument("lhs rank must be 2, but given lhs shape ",
                           lhs_shape.DebugString());
  }
  if (rhs_shape.dims() != 2) {
    return InvalidArgument("rhs rank must be 2, but given rhs shape ",
                           rhs_shape.DebugString());
  }
  if (lhs_shape.dim_size(1) != rhs_shape.dim_size(0)) {
    return InvalidArgument(
        "lhs.dim_size(1) and rhs.dim_size(0) must be equal, but given lhs "
        "shape ",
        lhs_shape.DebugString(), " and rhs shape ", rhs_shape.DebugString());
  }
  return OkStatus();
}

// Given data tensor's shape and quantization params, returns if the shapes are
// valid.
Status QuantizationAxisAndShapeValid(const TensorShape& data_shape,
                                     const TensorShape& scales_shape,
                                     const TensorShape& zero_points_shape,
                                     int quantization_axis) {
  if (!scales_shape.IsSameSize(zero_points_shape)) {
    return InvalidArgument(
        "scales and zero_points shape must be same, but given scales shape ",
        scales_shape.DebugString(), " and zero_points shape ",
        zero_points_shape.DebugString());
  }
  if (quantization_axis < -1 || quantization_axis >= data_shape.dims()) {
    return InvalidArgument(
        "quantization_axis must be -1 or in range [0, input.rank), but given ",
        quantization_axis);
  }

  if (quantization_axis == -1) {
    if (scales_shape.dims() != 0) {
      return InvalidArgument(
          "If quantization_axis is -1, scales and zero_points must be scalar "
          "tensors, but given scales shape ",
          scales_shape.DebugString(), " and zero_points shape ",
          zero_points_shape.DebugString());
    }
  } else {
    if (!(scales_shape.dims() == 1 &&
          scales_shape.dim_size(0) == data_shape.dim_size(quantization_axis))) {
      return InvalidArgument(
          "If quantization_axis is not -1, scales and zero_points must be a "
          "tensor of rank 1 and the size must be equal to the "
          "input.dim_size(quantization_axis), but given quantization_axis ",
          quantization_axis, ", scales shape ", scales_shape.DebugString(),
          " and zero_points shape ", zero_points_shape.DebugString());
    }
  }
  return OkStatus();
}

// Quantize input_val using given inv_scale and zero_point, using the formula:
// quantized_val = input_val * inv_scale + zero_point
//
// The caller is reponsible for the validity of the inv_scale (Avoid precision
// loss from taking inverse, and ensure that inv_scale is a finite number.)
template <typename Tin, typename Tout>
void AffineQuantize(const Tin& input_tensor, float inv_scale, int32 zero_point,
                    int32_t quantization_min_val, int32_t quantization_max_val,
                    Tout& quantized_tensor) {
  quantized_tensor = ((input_tensor.template cast<float>() * inv_scale + 0.5f)
                          .floor()
                          .template cast<int32_t>() +
                      zero_point)
                         .cwiseMin(quantization_max_val)
                         .cwiseMax(quantization_min_val)
                         .template cast<typename Tout::Scalar>();
}

// Given a portion of input float tensor, quantizes the data and writes output
// to the corresponding portion in quantized_tensor. The quantization scale and
// zero_point is calculated using the input data min and max.
// This function is used for dynamic range quantization in hybrid (float x qint)
// kernels.
//
// This function behavior aligns with TFLite AsymmetricQuantize() to achieve
// feature parity with TFLite which is required since supporting mobile
// executions is the one of the major use cases. The behavior is same except for
// following difference:
// TFLite AsymmetricQuantize() uses
// round(input / scale + zero_point),
// while AffineQuantize() uses
// floor(input_val * (1./scale) + 0.5) + zero_point
void AsymmetricQuantize(const Tensor& tensor, int apply_offset, int apply_size,
                        int32_t quantization_min_val,
                        int32_t quantization_max_val, float& scale,
                        int32& zero_point, Tensor& quantized_tensor) {
  Eigen::DSizes<Eigen::Index, 1> apply_offset_array{apply_offset};
  Eigen::DSizes<Eigen::Index, 1> apply_offset_size{apply_size};

  auto tensor_slice =
      tensor.flat<float>().slice(apply_offset_array, apply_offset_size);
  auto quantized_tensor_slice = quantized_tensor.flat<qint8>().slice(
      apply_offset_array, apply_offset_size);

  Eigen::Tensor<float, 0, Eigen::RowMajor> tensor_slice_min =
      tensor_slice.minimum();
  Eigen::Tensor<float, 0, Eigen::RowMajor> tensor_slice_max =
      tensor_slice.maximum();
  const double rmin = static_cast<double>(std::min(0.0f, tensor_slice_min()));
  const double rmax = static_cast<double>(std::max(0.0f, tensor_slice_max()));
  const double qmin_double = quantization_min_val;
  const double qmax_double = quantization_max_val;

  float inv_scale = 0;
  scale = (rmax - rmin) / (qmax_double - qmin_double);
  if (rmax - rmin != 0) {
    // Re-calculate the inverse instead of using (1./scale), to avoid loss of
    // precision.
    inv_scale = (qmax_double - qmin_double) / (rmax - rmin);
  }
  if (scale == 0 || !IsFinite(inv_scale)) {
    quantized_tensor_slice.setZero();
    scale = 1.0;
    zero_point = 0;
    return;
  }

  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);
  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  int8_t nudged_zero_point = 0;
  if (zero_point_double <= qmin_double) {
    nudged_zero_point = quantization_min_val;
  } else if (zero_point_double >= qmax_double) {
    nudged_zero_point = quantization_max_val;
  } else {
    nudged_zero_point = static_cast<int8_t>(round(zero_point_double));
  }
  zero_point = nudged_zero_point;

  AffineQuantize(tensor_slice, inv_scale, zero_point, quantization_min_val,
                 quantization_max_val, quantized_tensor_slice);
}

// Performs dot(lhs, rhs) and writes output to output. Assumes that output is
// already allocated with correct size.
//
// Given
// int32_t acc_f(Tlhs lhs_val, Trhs rhs_val, int64_t batch_idx,
//     int64_t output_channel_idx)
// and
// int32_t output_f(int32_t acc_val, int64_t batch_idx,
//     int64_t output_channel_idx)
// for each output element, accumulate value using acc_f along the contracting
// dimension and writes the accumulated value using output_f.
template <typename Tlhs, typename Trhs, typename Tout, typename AccF,
          typename OutputF>
void DotWithAccFunctionAndOutputFunction(const Tensor& lhs, const Tensor& rhs,
                                         Tensor& output, const AccF& acc_f,
                                         const OutputF& output_f) {
  const int64_t batches = output.dim_size(0);
  const int64_t output_depth = output.dim_size(1);
  const int64_t accum_depth = rhs.dim_size(0);

  const Tlhs* lhs_data = lhs.flat<Tlhs>().data();
  const Trhs* rhs_data = rhs.flat<Trhs>().data();
  Tout* output_data = output.flat<Tout>().data();

  for (int64_t b = 0; b < batches; ++b) {
    for (int64_t out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int64_t d = 0; d < accum_depth; ++d) {
        acc += acc_f(lhs_data[b * accum_depth + d],
                     rhs_data[d * output_depth + out_c], b, out_c);
      }
      output_data[b * output_depth + out_c] = output_f(acc, b, out_c);
    }
  }
}

// Performs dot on per-batch (dimension 0) quantized lhs and per-tensor
// quantized rhs.
template <typename Tlhs, typename Trhs>
void EvalLhsPerBatchAndRhsPerTensorQuantizedDot(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    const Tensor& lhs_scales, const Tensor& lhs_zero_points, float rhs_scale,
    int32_t rhs_zero_point, Tensor& output) {
  const float* lhs_scales_data = lhs_scales.flat<float>().data();
  const int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();

  DotWithAccFunctionAndOutputFunction<Tlhs, Trhs, float>(
      lhs, rhs, output,
      [lhs_zero_points_data, rhs_zero_point](Tlhs lhs_val, Trhs rhs_val,
                                             int64_t b, int64_t out_c) {
        return (static_cast<int32_t>(lhs_val) - lhs_zero_points_data[b]) *
               (static_cast<int32_t>(rhs_val) - rhs_zero_point);
      },
      [lhs_scales_data, rhs_scale](int32_t acc, int64_t b, int64_t out_c) {
        return acc * lhs_scales_data[b] * rhs_scale;
      });
}

// Performs dot on per-batch (dimension 0) quantized lhs and per-channel
// (dimension 1) quantized rhs.
template <typename Tlhs, typename Trhs>
void EvalLhsPerBatchAndRhsPerChannelQuantizedDot(
    const Tensor& lhs, const Tensor& rhs, const Tensor& lhs_scales,
    const Tensor& lhs_zero_points, const Tensor& rhs_scales,
    const Tensor& rhs_zero_points, Tensor& output) {
  const float* lhs_scales_data = lhs_scales.flat<float>().data();
  const int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();
  const float* rhs_scales_data = rhs_scales.flat<float>().data();
  const int32_t* rhs_zero_points_data = rhs_zero_points.flat<int32_t>().data();

  DotWithAccFunctionAndOutputFunction<Tlhs, Trhs, float>(
      lhs, rhs, output,
      [lhs_zero_points_data, rhs_zero_points_data](Tlhs lhs_val, Trhs rhs_val,
                                                   int64_t b, int64_t out_c) {
        return (static_cast<int32_t>(lhs_val) - lhs_zero_points_data[b]) *
               (static_cast<int32_t>(rhs_val) - rhs_zero_points_data[out_c]);
      },
      [lhs_scales_data, rhs_scales_data](int32_t acc, int64_t b,
                                         int64_t out_c) {
        return acc * lhs_scales_data[b] * rhs_scales_data[out_c];
      });
}

// Given float lhs and quantized rhs, performs per-batch dynamic range
// quantization on lhs, and then performs quantized dot on lhs and rhs. Assumes
// that output is already allocated with correct size.
// For more details on lhs quantization policy, refer to the comment of class
// UniformQuantizedDotHybridOp below.
template <typename Trhs>
Status EvalHybridDot(OpKernelContext* context, const Tensor& lhs,
                     const Tensor& rhs, const Tensor& rhs_scales,
                     const Tensor& rhs_zero_points, Tensor& output) {
  const int64_t batches = lhs.dim_size(0);
  const int64_t accum_depth = lhs.dim_size(1);

  Tensor lhs_quantized;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DT_QINT8, lhs.shape(), &lhs_quantized));
  Tensor lhs_scales;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_FLOAT, {batches}, &lhs_scales));
  Tensor lhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DT_INT32, {batches}, &lhs_zero_points));
  float* lhs_scales_data = lhs_scales.flat<float>().data();
  int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();

  for (int64_t b = 0; b < batches; ++b) {
    AsymmetricQuantize(lhs, b * accum_depth, accum_depth,
                       /*quantization_min_val=*/-128,
                       /*quantization_max_val=*/127, lhs_scales_data[b],
                       lhs_zero_points_data[b], lhs_quantized);
  }
  if (rhs_scales.dims() != 0) {
    EvalLhsPerBatchAndRhsPerChannelQuantizedDot<qint8, Trhs>(
        lhs_quantized, rhs, lhs_scales, lhs_zero_points, rhs_scales,
        rhs_zero_points, output);
  } else {
    EvalLhsPerBatchAndRhsPerTensorQuantizedDot<qint8, Trhs>(
        context, lhs_quantized, rhs, lhs_scales, lhs_zero_points,
        rhs_scales.scalar<float>()(), rhs_zero_points.scalar<int32_t>()(),
        output);
  }
  return Status::OK();
}

}  // namespace

// Given float lhs and quantized rhs, internally performs per-batch dynamic
// range quantization on lhs, and then performs quantized dot on lhs and rhs.
// This kernel internally quantizes lhs with following conditions. This aligns
// with the TFLite Hybrid FullyConnected kernel behavior (per-batch dynamic
// range quantization for float input), thus achieves the feature parity with
// TFLite which is required since supporting mobile executions is the one of the
// major use cases.
//
// - lhs_quantization_min = -128 (narrow_range = false)
// - lhs_quantization_max = 127
// - lhs_quantization_axis = 0 (per-batch quantization)
// - lhs_asymmetric_quantize = true
template <typename Tlhs, typename Trhs, typename Tout>
class UniformQuantizedDotHybridOp : public OpKernel {
 public:
  explicit UniformQuantizedDotHybridOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, (std::is_same<Tlhs, float>()),
                InvalidArgument("Unsupported lhs type."));
    OP_REQUIRES(context, (std::is_same<Trhs, qint8>()),
                InvalidArgument("Unsupported rhs type."));
    OP_REQUIRES(context, (std::is_same<Tout, float>()),
                InvalidArgument("Unsupported output type."));

    int rhs_quantization_axis;
    OP_REQUIRES_OK(context, context->GetAttr("rhs_quantization_axis",
                                             &rhs_quantization_axis));
    OP_REQUIRES(context,
                (rhs_quantization_axis == 1 || rhs_quantization_axis == -1),
                InvalidArgument("rhs_quantization_axis Attr must be 1 "
                                "(per-channel) or -1 (per-tensor)."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& lhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& rhs_scales = context->input(2);
    const Tensor& rhs_zero_points = context->input(3);

    // Check lhs and rhs shapes.
    OP_REQUIRES_OK(context, DotInputShapeValid(lhs.shape(), rhs.shape()));
    // Check rhs scales/zero_points shapes.
    OP_REQUIRES_OK(context,
                   QuantizationAxisAndShapeValid(
                       rhs.shape(), rhs_scales.shape(), rhs_zero_points.shape(),
                       /*quantization_axis=*/rhs_scales.dims() == 0 ? -1 : 1));
    OP_REQUIRES(context, AllElementsPositive<float>(rhs_scales),
                InvalidArgument("rhs scales elements must be all positive."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({lhs.dim_size(0), rhs.dim_size(1)}), &output));
    OP_REQUIRES_OK(context, EvalHybridDot<Trhs>(context, lhs, rhs, rhs_scales,
                                                rhs_zero_points, *output));
  }
};

REGISTER_KERNEL_BUILDER(Name("UniformQuantizedDotHybrid")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Tlhs")
                            .TypeConstraint<qint8>("Trhs")
                            .TypeConstraint<float>("Tout"),
                        UniformQuantizedDotHybridOp<float, qint8, float>);

}  // namespace tensorflow
