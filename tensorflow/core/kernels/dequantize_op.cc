/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"

namespace {
enum {
  QUANTIZE_MODE_MIN_COMBINED,
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
T Cast(float v) {
  return v;
}

template <>
bfloat16 Cast<bfloat16>(float v) {
  return bfloat16(v);
}

template <typename Device, typename T, typename S>
class DequantizeOp : public OpKernel {
 public:
  explicit DequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(
        ctx,
        (ctx->output_type(0) == DT_FLOAT || ctx->output_type(0) == DT_BFLOAT16),
        errors::InvalidArgument("Output type must be bfloat16 or float,"
                                " is '" +
                                DataTypeString(ctx->output_type(0)) + "'"));

    need_cast_ = true;
    if (ctx->output_type(0) == DT_FLOAT) {
      need_cast_ = false;
      OP_REQUIRES(ctx,
                  (mode_string == "MIN_COMBINED" ||
                   mode_string == "MIN_FIRST" || mode_string == "SCALED"),
                  errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                          " 'MIN_FIRST', or 'SCALED', is '" +
                                          mode_string + "'"));
    } else {
      OP_REQUIRES(
          ctx, (mode_string == "MIN_COMBINED"),
          errors::InvalidArgument("When output type is bfloat16, Mode"
                                  " string must be 'MIN_COMBINED', is '" +
                                  mode_string + "'"));
    }

    if (mode_string == "MIN_COMBINED") {
      mode_ = QUANTIZE_MODE_MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& input_min_tensor = ctx->input(1);
    const Tensor& input_max_tensor = ctx->input(2);

    OP_REQUIRES(
        ctx, axis_ < input.dims(),
        errors::InvalidArgument("Axis must be less than input dimension(",
                                input.dims(), "), got ", axis_));

    int num_slices = 1;
    if (axis_ > -1) {
      num_slices = input.dim_size(axis_);
    }
    OP_REQUIRES(ctx, input_min_tensor.NumElements() == num_slices,
                errors::InvalidArgument(
                    "input_min_tensor must have as many elements as input on "
                    "the dequantization axis (",
                    axis_, "), got ", input_min_tensor.NumElements(),
                    ", expected ", num_slices));
    OP_REQUIRES(ctx, input_max_tensor.NumElements() == num_slices,
                errors::InvalidArgument(
                    "input_max_tensor must have as many elements as input on "
                    "the dequantization axis (",
                    axis_, "), got ", input_max_tensor.NumElements(),
                    ", expected ", num_slices));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    Tensor float_output =
        need_cast_ ? tensorflow::Tensor(DT_FLOAT, input.shape()) : *output;
    if (num_slices == 1) {
      const float min_range = input_min_tensor.flat<float>()(0);
      const float max_range = input_max_tensor.flat<float>()(0);
      DequantizeTensor(ctx, input, min_range, max_range, &float_output);
    } else {
      OP_REQUIRES(ctx, mode_ != QUANTIZE_MODE_MIN_FIRST,
                  errors::Unimplemented("MIN_FIRST mode is not implemented for "
                                        "Dequantize with axis != -1."));

      int64_t pre_dim = 1, post_dim = 1;
      for (int i = 0; i < axis_; ++i) {
        pre_dim *= float_output.dim_size(i);
      }
      for (int i = axis_ + 1; i < float_output.dims(); ++i) {
        post_dim *= float_output.dim_size(i);
      }
      auto input_tensor = input.template bit_casted_shaped<T, 3>(
          {pre_dim, num_slices, post_dim});
      auto output_tensor =
          float_output.flat_inner_outer_dims<float, 3>(axis_ - 1);
      auto min_ranges = input_min_tensor.vec<float>();
      auto max_ranges = input_max_tensor.vec<float>();
      for (int i = 0; i < num_slices; ++i) {
        DequantizeSlice(ctx->eigen_device<Device>(), ctx,
                        input_tensor.template chip<1>(i), min_ranges(i),
                        max_ranges(i), output_tensor.template chip<1>(i));
      }
    }
    if (need_cast_) {
      S* out_ptr = output->flat<S>().data();
      float* in_ptr = float_output.flat<float>().data();
      for (int64_t i = 0; i < float_output.NumElements(); ++i) {
        out_ptr[i] = static_cast<S>(in_ptr[i]);
      }
    }
  }

  void DequantizeTensor(OpKernelContext* ctx, const Tensor& input,
                        const float min_range, const float max_range,
                        Tensor* output) {
    const float half_range =
        !std::numeric_limits<T>::is_signed
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;

    if (mode_ == QUANTIZE_MODE_MIN_COMBINED) {
      const float scale_factor =
          (max_range - min_range) /
          (static_cast<float>(std::numeric_limits<T>::max()) -
           std::numeric_limits<T>::min());

      const auto& input_tensor = input.flat<T>();
      output->flat<float>() =
          ((input_tensor.template cast<float>() + half_range) * scale_factor) +
          min_range;

    } else if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      if (meta::IsSupportedAndEnabled() && std::is_same<T, quint8>()) {
        auto input_ui8_array = input.flat<quint8>();
        meta::Dequantize(ctx, input_ui8_array.data(), input_ui8_array.size(),
                         min_range, max_range, output->flat<float>().data());
      } else {
        QuantizedTensorToFloatInPlaceUsingEigen<T>(
            ctx->template eigen_device<Device>(), input, min_range, max_range,
            output);
      }
    } else if (mode_ == QUANTIZE_MODE_SCALED) {
      const int min_output_value =
          std::numeric_limits<T>::min() + (narrow_range_ ? 1 : 0);
      const float scale_factor =
          std::numeric_limits<T>::min() == 0
              ? (max_range / std::numeric_limits<T>::max())
              : std::max(min_range / min_output_value,
                         max_range / std::numeric_limits<T>::max());
      const auto& input_tensor = input.flat<T>();
      output->flat<float>() =
          input_tensor.template cast<int>().template cast<float>() *
          scale_factor;
    }
  }

  template <typename ConstVec, typename Vec>
  void DequantizeSlice(const Device& d, OpKernelContext* ctx,
                       const ConstVec& input, float min_range, float max_range,
                       Vec output) {
    // TODO(pauldonnelly): Factor out the similar calculations in quantize,
    //   dequantize and quantize_and_dequantize ops.
    const float half_range =
        !std::numeric_limits<T>::is_signed
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;

    if (mode_ == QUANTIZE_MODE_MIN_COMBINED) {
      const float scale_factor =
          (max_range - min_range) /
          (static_cast<float>(std::numeric_limits<T>::max()) -
           std::numeric_limits<T>::min());

      output.device(d) =
          ((input.template cast<float>() + half_range) * scale_factor) +
          min_range;
    } else if (mode_ == QUANTIZE_MODE_SCALED) {
      const int min_output_value =
          std::numeric_limits<T>::min() + (narrow_range_ ? 1 : 0);
      const float scale_factor =
          std::numeric_limits<T>::min() == 0
              ? (max_range / std::numeric_limits<T>::max())
              : std::max(min_range / min_output_value,
                         max_range / std::numeric_limits<T>::max());
      output.device(d) = input.template cast<float>() * scale_factor;
    }
  }

 private:
  int mode_;
  int axis_;
  bool narrow_range_;
  bool need_cast_;
};

REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, quint8, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, qint8, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint16>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, quint16, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint16>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, qint16, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, qint32, float>);

REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, quint8, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, qint8, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint16>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, quint16, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint16>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, qint16, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, qint32, bfloat16>);
}  // namespace tensorflow
