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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace {
enum {
  QUANTIZE_MODE_MIN_COMBINED,
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class DequantizeOp : public OpKernel {
 public:
  explicit DequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    half_range_ = !std::is_signed<T>::value
                      ? 0.0f
                      : (static_cast<float>(std::numeric_limits<T>::max()) -
                         std::numeric_limits<T>::min() + 1) /
                            2.0f;
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST" ||
                 mode_string == "SCALED"),
                errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                        " 'MIN_FIRST', or 'SCALED', is '" +
                                        mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QUANTIZE_MODE_MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const float min_range = ctx->input(1).flat<float>()(0);
    const float max_range = ctx->input(2).flat<float>()(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (mode_ == QUANTIZE_MODE_MIN_COMBINED) {
      const float scale_factor =
          (max_range - min_range) /
          (static_cast<float>(std::numeric_limits<T>::max()) -
           std::numeric_limits<T>::min());

      float* out_ptr = output->flat<float>().data();
      const T* in_ptr = input.flat<T>().data();

      const int64 num_elements = input.NumElements();
      for (int i = 0; i < num_elements; ++i) {
        out_ptr[i] =
            ((static_cast<int>(in_ptr[i]) + half_range_) * scale_factor) +
            min_range;
      }
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
      const float scale_factor =
          std::numeric_limits<T>::min() == 0
              ? (max_range / std::numeric_limits<T>::max())
              : std::max(min_range / std::numeric_limits<T>::min(),
                         max_range / std::numeric_limits<T>::max());
      float* out_ptr = output->flat<float>().data();
      const T* in_ptr = input.flat<T>().data();
      const int64 num_elements = input.NumElements();
      for (int64 i = 0; i < num_elements; ++i) {
        out_ptr[i] = static_cast<int>(in_ptr[i]) * scale_factor;
      }
    }
  }

 private:
  float half_range_;
  int mode_;
};

REGISTER_KERNEL_BUILDER(
    Name("Dequantize").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    DequantizeOp<CPUDevice, quint8>);
REGISTER_KERNEL_BUILDER(
    Name("Dequantize").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    DequantizeOp<CPUDevice, qint8>);
REGISTER_KERNEL_BUILDER(
    Name("Dequantize").Device(DEVICE_CPU).TypeConstraint<quint16>("T"),
    DequantizeOp<CPUDevice, quint16>);
REGISTER_KERNEL_BUILDER(
    Name("Dequantize").Device(DEVICE_CPU).TypeConstraint<qint16>("T"),
    DequantizeOp<CPUDevice, qint16>);

REGISTER_KERNEL_BUILDER(
    Name("Dequantize").Device(DEVICE_CPU).TypeConstraint<qint32>("T"),
    DequantizeOp<CPUDevice, qint32>);

}  // namespace tensorflow
