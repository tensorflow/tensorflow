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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include <math.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <class T1, class T2>
class QuantizeDownAndShrinkRangeOp : public OpKernel {
 public:
  explicit QuantizeDownAndShrinkRangeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const float input_min_float = ctx->input(1).flat<float>()(0);
    const float input_max_float = ctx->input(2).flat<float>()(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_min));
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &output_max));

    // See QuantizationRangeOp as well, which has a copy of this logic.
    auto input_array = input.flat<T1>();
    const int32_t input_lowest_quantized =
        static_cast<int32>(Eigen::NumTraits<T1>::lowest());
    const int32_t input_highest_quantized =
        static_cast<int32>(Eigen::NumTraits<T1>::highest());
    T1 actual_min_quantized = input_highest_quantized;
    T1 actual_max_quantized = input_lowest_quantized;
    for (int i = 0; i < input_array.size(); ++i) {
      const T1 value = input_array(i);
      actual_min_quantized = std::min(actual_min_quantized, value);
      actual_max_quantized = std::max(actual_max_quantized, value);
    }
    // We want to make sure that the minimum is no larger than zero, so that the
    // convolution operation can run efficiently.
    const float actual_min_float =
        std::min(0.0f, QuantizedToFloat(actual_min_quantized, input_min_float,
                                        input_max_float));
    const float actual_max_float = QuantizedToFloat(
        actual_max_quantized, input_min_float, input_max_float);

#if 0
    // This is the reference, non-eigen implementation:
    auto output_array = output->flat<T2>();
    RequantizeManyInNewRange<T1, T2>(input_array.data(), input_array.size(),
                                     input_min_float, input_max_float,
                                     actual_min_float, actual_max_float,
                                     output_array.data());
#endif

    if (input_array.size() > 0) {
      if (meta::IsSupportedAndEnabled() && std::is_same<T1, qint32>() &&
          std::is_same<T2, quint8>()) {
        auto input_i32_array = input.flat<qint32>();
        meta::Requantize(ctx, input_i32_array.data(), input_i32_array.size(),
                         input_min_float, input_max_float, actual_min_float,
                         actual_max_float, output->flat<quint8>().data());
      } else {
        RequantizeManyInNewRangeUsingEigen<T1, T2>(
            ctx->eigen_device<CPUDevice>(), input, input_min_float,
            input_max_float, actual_min_float, actual_max_float, output);
      }
    }

    output_min->flat<float>().setConstant(actual_min_float);
    output_max->flat<float>().setConstant(actual_max_float);
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantizeDownAndShrinkRange")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput")
                            .TypeConstraint<quint8>("out_type"),
                        QuantizeDownAndShrinkRangeOp<qint32, quint8>);

}  // namespace tensorflow
