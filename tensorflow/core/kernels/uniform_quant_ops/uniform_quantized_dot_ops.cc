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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

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

// Performs dot on per-tensor quantized lhs and per-tensor quantized rhs.
template <typename Tin, typename Tout>
Status EvalLhsPerTensorAndRhsPerTensorQuantizedDot(
    const Tensor& lhs, const Tensor& rhs, float lhs_scale,
    int32_t lhs_zero_point, float rhs_scale, int32_t rhs_zero_point,
    float output_scale, int32_t output_zero_point,
    int output_quantization_min_val, int output_quantization_max_val,
    Tensor& output) {
  const double effective_multiplier =
      static_cast<double>(lhs_scale) * rhs_scale / output_scale;
  int32_t effective_quantized_multiplier;
  int effective_shift;
  TF_RETURN_IF_ERROR(QuantizeMultiplier(
      effective_multiplier, effective_quantized_multiplier, effective_shift));

  DotWithAccFunctionAndOutputFunction<Tin, Tin, Tout>(
      lhs, rhs, output,
      [lhs_zero_point, rhs_zero_point](Tin lhs_val, Tin rhs_val, int64_t b,
                                       int64_t out_c) {
        return static_cast<Tout>(
            (static_cast<int32_t>(lhs_val) - lhs_zero_point) *
            (static_cast<int32_t>(rhs_val) - rhs_zero_point));
      },
      [effective_quantized_multiplier, effective_shift, output_zero_point,
       output_quantization_min_val,
       output_quantization_max_val](int32_t acc, int64_t b, int64_t out_c) {
        return AffineRequantizeWithQuantizedMultiplierAndShift<int32_t, Tout>(
            acc, effective_quantized_multiplier, effective_shift,
            /*input_zero_point=*/0, output_zero_point,
            output_quantization_min_val, output_quantization_max_val);
      });
  return OkStatus();
}

// Performs dot on per-tensor quantized lhs and per-channel (dimension 1)
// quantized rhs.
template <typename Tin, typename Tout>
Status EvalLhsPerTensorAndRhsPerChannelQuantizedDot(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    float lhs_scale, int32_t lhs_zero_point, const Tensor& rhs_scales,
    const Tensor& rhs_zero_points, const Tensor& output_scales,
    const Tensor& output_zero_points, int output_quantization_min_val,
    int output_quantization_max_val, Tensor& output) {
  const int output_depth = output.dim_size(1);
  const float* rhs_scales_data = rhs_scales.flat<float>().data();
  const int32_t* rhs_zero_points_data = rhs_zero_points.flat<int32_t>().data();

  Tensor effective_quantized_multipliers;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, rhs_scales.shape(),
                                            &effective_quantized_multipliers));
  Tensor effective_shifts;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DT_INT32, rhs_scales.shape(), &effective_shifts));
  int32_t* effective_quantized_multipliers_data =
      effective_quantized_multipliers.flat<int32_t>().data();
  int32_t* effective_shifts_data = effective_shifts.flat<int32_t>().data();

  const bool is_output_scales_scalar = output_scales.dims() == 0;

  if (!is_output_scales_scalar) {
    const float* output_scales_data = output_scales.flat<float>().data();
    for (int64_t out_c = 0; out_c < output_depth; ++out_c) {
      const double effective_multiplier = static_cast<double>(lhs_scale) *
                                          rhs_scales_data[out_c] /
                                          output_scales_data[out_c];
      TF_RETURN_IF_ERROR(QuantizeMultiplier(
          effective_multiplier, effective_quantized_multipliers_data[out_c],
          effective_shifts_data[out_c]));
    }
  } else {
    const float output_scale = output_scales.scalar<float>()();
    for (int64_t out_c = 0; out_c < output_depth; ++out_c) {
      const double effective_multiplier = static_cast<double>(lhs_scale) *
                                          rhs_scales_data[out_c] / output_scale;
      TF_RETURN_IF_ERROR(QuantizeMultiplier(
          effective_multiplier, effective_quantized_multipliers_data[out_c],
          effective_shifts_data[out_c]));
    }
  }

  const int32_t* output_zero_points_data =
      output_zero_points.flat<int32_t>().data();
  DotWithAccFunctionAndOutputFunction<Tin, Tin, Tout>(
      lhs, rhs, output,
      [lhs_zero_point, rhs_zero_points_data](Tin lhs_val, Tin rhs_val,
                                             int64_t b, int64_t out_c) {
        return (static_cast<int32_t>(lhs_val) - lhs_zero_point) *
               (static_cast<int32_t>(rhs_val) - rhs_zero_points_data[out_c]);
      },
      [effective_quantized_multipliers_data, effective_shifts_data,
       output_zero_points_data, output_quantization_min_val,
       output_quantization_max_val,
       is_output_scales_scalar](int32_t acc, int64_t b, int64_t out_c) {
        return AffineRequantizeWithQuantizedMultiplierAndShift<int32_t, Tout>(
            acc, effective_quantized_multipliers_data[out_c],
            effective_shifts_data[out_c],
            /*input_zero_point=*/0,
            output_zero_points_data[is_output_scales_scalar ? 0 : out_c],
            output_quantization_min_val, output_quantization_max_val);
      });
  return OkStatus();
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

// Given quantized lhs and quantized rhs, performs quantized dot on lhs and rhs,
// and produce quantized output. Assumes that output is already allocated with
// correct size.
template <typename Tin, typename Tout>
Status EvalQuantizedDot(OpKernelContext* context, const Tensor& lhs,
                        const Tensor& rhs, const Tensor& lhs_scales,
                        const Tensor& lhs_zero_points, const Tensor& rhs_scales,
                        const Tensor& rhs_zero_points,
                        const Tensor& output_scales,
                        const Tensor& output_zero_points,
                        int output_quantization_min_val,
                        int output_quantization_max_val, Tensor& output) {
  const float lhs_scale = lhs_scales.scalar<float>()();
  const int32_t lhs_zero_point = lhs_zero_points.scalar<int32_t>()();
  if (rhs_scales.dims() != 0) {
    return EvalLhsPerTensorAndRhsPerChannelQuantizedDot<Tin, Tout>(
        context, lhs, rhs, lhs_scale, lhs_zero_point, rhs_scales,
        rhs_zero_points, output_scales, output_zero_points,
        output_quantization_min_val, output_quantization_max_val, output);
  } else {
    const float rhs_scale = rhs_scales.scalar<float>()();
    const int32_t rhs_zero_point = rhs_zero_points.scalar<int32_t>()();
    const float output_scale = output_scales.scalar<float>()();
    const int32_t output_zero_point = output_zero_points.scalar<int32_t>()();
    return EvalLhsPerTensorAndRhsPerTensorQuantizedDot<Tin, Tout>(
        lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point,
        output_scale, output_zero_point, output_quantization_min_val,
        output_quantization_max_val, output);
  }
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

  auto lhs_tensor = lhs.template tensor<float, 2>();
  auto lhs_quantized_tensor = lhs_quantized.template tensor<qint8, 2>();
  for (int64_t b = 0; b < batches; ++b) {
    TF_RETURN_IF_ERROR(AsymmetricQuantize(
        lhs_tensor.template chip<0>(b),
        /*quantization_min_val=*/-128,
        /*quantization_max_val=*/127, lhs_scales_data[b],
        lhs_zero_points_data[b], lhs_quantized_tensor.template chip<0>(b)));
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
  return OkStatus();
}

}  // namespace

// Given quantized lhs and quantized rhs, performs quantized dot on lhs and rhs,
// and produce quantized output.
template <typename Tin, typename Tout>
class UniformQuantizedDotOp : public OpKernel {
 public:
  explicit UniformQuantizedDotOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, (std::is_same<Tin, qint8>()),
                InvalidArgument("Unsupported lhs/rhs type."));
    OP_REQUIRES(context, (std::is_same<Tout, qint32>()),
                InvalidArgument("Unsupported output type."));

    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_min_val",
                                             &output_quantization_min_val_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_max_val",
                                             &output_quantization_max_val_));

    int lhs_quantization_axis;
    OP_REQUIRES_OK(context, context->GetAttr("lhs_quantization_axis",
                                             &lhs_quantization_axis));
    OP_REQUIRES(
        context, (lhs_quantization_axis == -1),
        InvalidArgument("lhs_quantization_axis Attr must be -1 (per-tensor)."));

    int rhs_quantization_axis;
    OP_REQUIRES_OK(context, context->GetAttr("rhs_quantization_axis",
                                             &rhs_quantization_axis));
    OP_REQUIRES(context,
                (rhs_quantization_axis == 1 || rhs_quantization_axis == -1),
                InvalidArgument("rhs_quantization_axis Attr must be 1 "
                                "(per-channel) or -1 (per-tensor)."));

    int output_quantization_axis;
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_axis",
                                             &output_quantization_axis));
    OP_REQUIRES(
        context,
        (output_quantization_axis == 1 || output_quantization_axis == -1),
        InvalidArgument("output_quantization_axis Attr must be 1 "
                        "(per-channel) or -1 (per-tensor)."));
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

    OP_REQUIRES(context, (AllElementsPositive<float>(lhs_scales)),
                InvalidArgument("lhs scales elements must be all positive."));
    OP_REQUIRES(context, (AllElementsPositive<float>(rhs_scales)),
                InvalidArgument("rhs scales elements must be all positive."));
    OP_REQUIRES(
        context, (AllElementsPositive<float>(output_scales)),
        InvalidArgument("output scales elements must be all positive."));

    // Check lhs and rhs shapes.
    OP_REQUIRES_OK(context, DotInputShapeValid(lhs.shape(), rhs.shape()));
    // Check lhs scales/zero_points shapes.
    OP_REQUIRES(
        context,
        (lhs_scales.IsSameSize(lhs_zero_points) && lhs_scales.dims() == 0),
        InvalidArgument(
            "lhs scales/zero_points must be all scalar tensors. Given: ",
            lhs_scales.shape().DebugString(),
            lhs_zero_points.shape().DebugString()));

    // Check rhs scales/zero_points shapes.
    OP_REQUIRES_OK(context,
                   QuantizationAxisAndShapeValid(
                       rhs.shape(), rhs_scales.shape(), rhs_zero_points.shape(),
                       /*quantization_axis=*/rhs_scales.dims() == 0 ? -1 : 1));
    TensorShape output_shape({lhs.dim_size(0), rhs.dim_size(1)});
    // Check output scales/zero_points shapes.
    OP_REQUIRES_OK(
        context,
        QuantizationAxisAndShapeValid(
            output_shape, output_scales.shape(), output_zero_points.shape(),
            /*quantization_axis=*/output_scales.dims() == 0 ? -1 : 1));

    OP_REQUIRES(
        context, (rhs_scales.dims() > 0 || output_scales.dims() == 0),
        InvalidArgument(
            "If rhs is per-tensor quantized, output must be also per-tensor "
            "quantized. Given rhs scales and zero_points of shape ",
            rhs_scales.shape().DebugString(),
            " but given output scales and zero_points of shape ",
            output_scales.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({lhs.dim_size(0), rhs.dim_size(1)}), &output));

    OP_REQUIRES_OK(
        context, EvalQuantizedDot<Tin, Tout>(
                     context, lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales,
                     rhs_zero_points, output_scales, output_zero_points,
                     output_quantization_min_val_, output_quantization_max_val_,
                     *output));
  }

 private:
  int output_quantization_min_val_;
  int output_quantization_max_val_;
};

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

REGISTER_KERNEL_BUILDER(Name("UniformQuantizedDot")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tin")
                            .TypeConstraint<qint32>("Tout"),
                        UniformQuantizedDotOp<qint8, qint32>);

REGISTER_KERNEL_BUILDER(Name("UniformQuantizedDotHybrid")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Tlhs")
                            .TypeConstraint<qint8>("Trhs")
                            .TypeConstraint<float>("Tout"),
                        UniformQuantizedDotHybridOp<float, qint8, float>);

}  // namespace tensorflow
