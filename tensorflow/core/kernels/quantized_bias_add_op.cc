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

// Implements a quantized eight-bit version of the bias addition operation.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

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
    OP_REQUIRES(context, bias.NumElements() > 0,
                errors::InvalidArgument("Must provide at least 1 bias"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    float total_min;
    float total_max;

    if (meta::IsSupportedAndEnabled() && std::is_same<T1, quint8>() &&
        std::is_same<T2, quint8>() && std::is_same<T3, qint32>()) {
      auto input_ui8_array = input.flat<quint8>();
      auto bias_ui8_array = bias.flat<quint8>();
      GetOutputMinAndMaxForQuantizedAdd(input_min, input_max, bias_min,
                                        bias_max, &total_min, &total_max);
      meta::QuantizedBiasAdd(context, input_ui8_array.data(),
                             input_ui8_array.size(), bias_ui8_array.data(),
                             bias_ui8_array.size(), input_min, input_max,
                             bias_min, bias_max, total_min, total_max,
                             output->flat<qint32>().data());
    } else {
      QuantizedAddUsingEigen<T1, T2, T3>(
          context->template eigen_device<CPUDevice>(), input, input_min,
          input_max, bias, bias_min, bias_max, output, &total_min, &total_max);
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
