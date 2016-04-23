/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Implements a quantized eight-bit version of the bias addition operation.

#include "tensorflow/contrib/quantization/kernels/quantization_utils.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

template <class T1, class T2, class T3>
class QuantizedBiasAddOp : public OpKernel {
 public:
  explicit QuantizedBiasAddOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);
    const float input_min = context->input(2).flat<float>()(0);
    const float input_max = context->input(3).flat<float>()(0);
    const float bias_min = context->input(4).flat<float>()(0);
    const float bias_max = context->input(5).flat<float>()(0);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));
    const auto last_dim = input.shape().dims() - 1;
    OP_REQUIRES(
        context, bias.shape().dim_size(0) == input.shape().dim_size(last_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    const auto& input_flat = input.flat<T1>();
    const int64 input_element_count = input.NumElements();
    const auto& bias_flat = bias.flat<T2>();
    const int64 bias_element_count = bias.NumElements();
    auto output_flat = output->flat<T3>();
    const size_t how_many_iterations =
        (input_element_count / bias_element_count);

    // We need to have a good range to add our two arguments together in. This
    // is surprisingly tricky, since it has to satisfy a few different needs:
    //  - Must be symmetrical around zero, so that 0 + 0 = 0.
    //  - Must hold the largest of the argument ranges.
    //  - Should have enough range that the bits of the lowest and highest
    //    arguments overlap if possible without the lower getting truncated.
    //  - Should have some headroom so that there's no overflow.
    //  - Needs to be signed.
    // This leads us to use a scheme where we (assuming the inputs are eight bit
    // and the output is 32-bit) use the bottom 32 - 17 = 15 bits to store the
    // accumulated results. This gives us all the properties we need.
    const float total_max =
        std::max(input_max,
                 std::max(-input_min, std::max(bias_max, -bias_min))) *
        (1 << 17);
    const float total_min = -total_max;

    // To do addition properly, we need to compensate for a possibly unbalanced
    // zero point in the total representation. The quantized value that
    // represents the real number zero needs to be subtracted before addition to
    // make sure that the identity of zero + zero = zero holds.
    const T3 zero_in_total_space =
        FloatToQuantized<T3>(0.0f, total_min, total_max);

    // This is a reference implementation of the bias addition for quantized
    // buffers, designed to provide a clear specification for the result we
    // want. We'll want to specialize this for particular hardware, and
    // probably even fuse it with matrix multiplications in a lot of cases. It's
    // important to show the clamping behavior we want in particular.
    for (size_t iteration = 0; iteration < how_many_iterations; ++iteration) {
      const size_t offset = iteration * bias_element_count;
      for (int c = 0; c < bias_element_count; ++c) {
        const int index = (offset + c);
        // The two numbers we're going to add can each be in very different
        // ranges (e.g. the quantized value '127' may represent very different
        // real numbers in both) so we need to convert them to a common range
        // before we sum them.
        const T1 input_value = input_flat(index);
        const T3 input_in_total_space = RequantizeInNewRange<T1, T3>(
            input_value, input_min, input_max, total_min, total_max);
        const T2 bias_value = bias_flat(c);
        const T3 bias_in_total_space = RequantizeInNewRange<T2, T3>(
            bias_value, bias_min, bias_max, total_min, total_max);
        const T3 total_pre = input_in_total_space + bias_in_total_space;
        // As noted above, we need to compensate for the offset of the actual
        // zero point in the space we're operating in.
        const T3 total = total_pre + zero_in_total_space;
        output_flat(index) = total;
      }
    }

    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = total_min;

    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = total_max;
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantizedBiasAdd")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<quint8>("T2")
                            .TypeConstraint<qint32>("out_type"),
                        QuantizedBiasAddOp<quint8, quint8, qint32>);
REGISTER_KERNEL_BUILDER(Name("QuantizedBiasAdd")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T1")
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint<qint32>("out_type"),
                        QuantizedBiasAddOp<qint8, qint8, qint32>);
}  // namespace tensorflow
