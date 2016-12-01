/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <class T1>
class RequantizationRangeOp : public OpKernel {
 public:
  explicit RequantizationRangeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const float input_min_float = ctx->input(1).flat<float>()(0);
    const float input_max_float = ctx->input(2).flat<float>()(0);
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output_min));
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_max));

    // See the deprecated QuantizeDownAndShrinkRangeOp as well, which has a copy
    // of this logic.
    auto input_array = input.flat<T1>();
    const int32 input_lowest_quantized =
        static_cast<int32>(Eigen::NumTraits<T1>::lowest());
    const int32 input_highest_quantized =
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

    output_min->flat<float>().setConstant(actual_min_float);
    output_max->flat<float>().setConstant(actual_max_float);
  }
};

REGISTER_KERNEL_BUILDER(Name("RequantizationRange")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput"),
                        RequantizationRangeOp<qint32>);

}  // namespace tensorflow
