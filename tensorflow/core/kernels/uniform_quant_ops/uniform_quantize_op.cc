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
#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "xla/tsl/platform/errors.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

// Compute validates finite inputs and positive normal scales. Add the zero
// point and clamp in double before narrowing, including when the float product
// overflows. Keep the existing float rounding when the reciprocal is normal.
template <typename ConstTensorTin, typename TensorTout>
void SaturatingAffineQuantize(const ConstTensorTin& input_tensor, float scale,
                              int32_t zero_point, int32_t quantization_min_val,
                              int32_t quantization_max_val,
                              TensorTout quantized_tensor) {
  const auto saturate = [&](const auto& rounded) {
    quantized_tensor =
        (rounded.template cast<double>() + static_cast<double>(zero_point))
            .cwiseMin(static_cast<double>(quantization_max_val))
            .cwiseMax(static_cast<double>(quantization_min_val))
            .template cast<int32_t>()
            .template cast<typename TensorTout::Scalar>();
  };
  const float inv_scale = 1.0f / scale;
  if (std::isnormal(inv_scale)) {
    saturate((input_tensor.template cast<float>() * inv_scale + 0.5f).floor());
  } else {
    // A large normal scale can have a subnormal reciprocal. Divide in double
    // so flushing float subnormals does not discard the entire input value.
    saturate(
        (input_tensor.template cast<double>() / static_cast<double>(scale) +
         0.5)
            .floor());
  }
}

template <typename Tin, typename Tout>
void EvalPerTensorQuantize(const Tensor& input, float scale, int32_t zero_point,
                           int32_t quantization_min_val,
                           int32_t quantization_max_val, Tensor& output) {
  SaturatingAffineQuantize(input.flat<Tin>(), scale, zero_point,
                           quantization_min_val, quantization_max_val,
                           output.flat<Tout>());
}

template <typename Tin, typename Tout>
void EvalPerChannelQuantize(const Tensor& input, const Tensor& scales,
                            const Tensor& zero_points, int quantization_axis,
                            int32_t quantization_min_val,
                            int32_t quantization_max_val, Tensor& output) {
  DCHECK(input.IsSameSize(output));
  const auto scales_data = scales.flat<float>();
  const auto zero_points_data = zero_points.flat<int32_t>();

  auto input_tensor =
      input.template flat_inner_outer_dims<Tin, 3>(quantization_axis - 1);
  auto output_tensor =
      output.template flat_inner_outer_dims<Tout, 3>(quantization_axis - 1);

  for (int64_t i = 0; i < scales_data.size(); ++i) {
    SaturatingAffineQuantize(input_tensor.template chip<1>(i), scales_data(i),
                             zero_points_data(i), quantization_min_val,
                             quantization_max_val,
                             output_tensor.template chip<1>(i));
  }
}

template <typename Tin, typename Tout>
void EvalQuantize(const Tensor& input, const Tensor& scales,
                  const Tensor& zero_points, int quantization_axis,
                  int32_t quantization_min_val, int32_t quantization_max_val,
                  Tensor& output) {
  if (quantization_axis >= 0) {
    EvalPerChannelQuantize<Tin, Tout>(input, scales, zero_points,
                                      quantization_axis, quantization_min_val,
                                      quantization_max_val, output);
  } else {
    EvalPerTensorQuantize<Tin, Tout>(
        input, scales.scalar<float>()(), zero_points.scalar<int32_t>()(),
        quantization_min_val, quantization_max_val, output);
  }
}

}  // namespace

class UniformQuantizeOp : public OpKernel {
 public:
  explicit UniformQuantizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Tin", &tin_));
    OP_REQUIRES(context, tin_ == DataType::DT_FLOAT,
                absl::InvalidArgumentError("Unsupported input type."));
    OP_REQUIRES_OK(context, context->GetAttr("Tout", &tout_));
    OP_REQUIRES(context,
                tout_ == DataType::DT_QINT8 || tout_ == DataType::DT_QUINT8 ||
                    tout_ == DataType::DT_QINT32,
                absl::InvalidArgumentError("Unsupported output type."));

    OP_REQUIRES_OK(context, context->GetAttr("quantization_min_val",
                                             &quantization_min_val_));
    OP_REQUIRES_OK(context, context->GetAttr("quantization_max_val",
                                             &quantization_max_val_));

    const int32_t output_min =
        tout_ == DT_QINT8
            ? -128
            : (tout_ == DT_QUINT8 ? 0 : std::numeric_limits<int32_t>::min());
    const int32_t output_max =
        tout_ == DT_QINT8
            ? 127
            : (tout_ == DT_QUINT8 ? 255 : std::numeric_limits<int32_t>::max());
    OP_REQUIRES(context, quantization_min_val_ <= quantization_max_val_,
                InvalidArgument("quantization bounds must be ordered."));
    OP_REQUIRES(context, quantization_min_val_ >= output_min,
                InvalidArgument("quantization_min_val must fit Tout."));
    OP_REQUIRES(context, quantization_max_val_ <= output_max,
                InvalidArgument("quantization_max_val must fit Tout."));

    OP_REQUIRES_OK(context,
                   context->GetAttr("quantization_axis", &quantization_axis_));
    OP_REQUIRES(
        context, (quantization_axis_ >= -1),
        absl::InvalidArgumentError(absl::StrCat(
            "quantization_axis must be >= -1, given: ", quantization_axis_)));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& scales = context->input(1);
    const Tensor& zero_points = context->input(2);

    OP_REQUIRES(context, scales.NumElements() == zero_points.NumElements(),
                InvalidArgument("scales and zero_points sizes must match."));
    if (quantization_axis_ == -1) {
      OP_REQUIRES(context, scales.dims() == 0,
                  InvalidArgument("scales must be scalar."));
      OP_REQUIRES(context, zero_points.dims() == 0,
                  InvalidArgument("zero_points must be scalar."));
    }
    OP_REQUIRES_OK(context, (QuantizationAxisAndShapeValid(
                                input.shape(), scales.shape(),
                                zero_points.shape(), quantization_axis_)));

    // A positive normal scale has a finite reciprocal. Subnormal scales are
    // rejected because their reciprocals can overflow or they can flush to
    // zero.
    const auto scales_data = scales.flat<float>();
    for (int64_t i = 0; i < scales_data.size(); ++i) {
      const float scale = scales_data(i);
      OP_REQUIRES(context, scale > 0.0f && std::isnormal(scale),
                  errors::InvalidArgument(
                      "scales must be positive, finite, and not subnormal."));
    }

    const auto input_data = input.flat<float>();
    for (int64_t i = 0; i < input_data.size(); ++i) {
      OP_REQUIRES(context, std::isfinite(input_data(i)),
                  InvalidArgument("input values must be finite."));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (tout_ == DataType::DT_QINT8) {
      EvalQuantize<float, qint8>(input, scales, zero_points, quantization_axis_,
                                 quantization_min_val_, quantization_max_val_,
                                 *output);
    } else if (tout_ == DataType::DT_QUINT8) {
      EvalQuantize<float, quint8>(input, scales, zero_points,
                                  quantization_axis_, quantization_min_val_,
                                  quantization_max_val_, *output);
    } else {
      EvalQuantize<float, qint32>(input, scales, zero_points,
                                  quantization_axis_, quantization_min_val_,
                                  quantization_max_val_, *output);
    }
  }

 private:
  DataType tin_, tout_;
  int quantization_axis_;
  int32_t quantization_min_val_;
  int32_t quantization_max_val_;
};

REGISTER_KERNEL_BUILDER(Name("UniformQuantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Tin")
                            .TypeConstraint("Tout",
                                            {DT_QINT8, DT_QUINT8, DT_QINT32}),
                        UniformQuantizeOp);

}  // namespace tensorflow
