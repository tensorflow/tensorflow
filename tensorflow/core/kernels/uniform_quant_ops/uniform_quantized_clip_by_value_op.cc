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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

namespace {

using errors::InvalidArgument;

template <typename T>
void EvalQuantizedClipByValue(const Tensor& operand, const Tensor& min,
                              const Tensor& max, int quantization_axis,
                              Tensor& output) {
  if (quantization_axis >= 0) {
    // It is guaranteed by QuantizationAxisAndShapeValid() that
    // quantization_axis lies in region [0, operand.dims) if quantization_axis
    // >= 0.
    auto operand_tensor =
        operand.template flat_inner_outer_dims<T, 3>(quantization_axis - 1);
    auto output_tensor =
        output.template flat_inner_outer_dims<T, 3>(quantization_axis - 1);
    auto min_tensor = min.flat<T>();
    auto max_tensor = max.flat<T>();

    const int64_t quantization_dim_size = operand.dim_size(quantization_axis);
    for (int i = 0; i < quantization_dim_size; ++i) {
      output_tensor.template chip<1>(i) = operand_tensor.template chip<1>(i)
                                              .cwiseMax(min_tensor(i))
                                              .cwiseMin(max_tensor(i));
    }
  } else {
    output.flat<T>() = operand.flat<T>()
                           .cwiseMax(min.scalar<T>()())
                           .cwiseMin(max.scalar<T>()());
  }
}

}  // namespace

template <typename T>
class UniformQuantizedClipByValueOp : public OpKernel {
 public:
  explicit UniformQuantizedClipByValueOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, (std::is_same<T, qint32>()),
                InvalidArgument("Unsupported operand type."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("quantization_axis", &quantization_axis_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& operand = context->input(0);
    const Tensor& min = context->input(1);
    const Tensor& max = context->input(2);
    const Tensor& scales = context->input(3);
    const Tensor& zero_points = context->input(4);

    OP_REQUIRES_OK(context, QuantizationAxisAndShapeValid(
                                operand.shape(), scales.shape(),
                                zero_points.shape(), quantization_axis_));
    OP_REQUIRES(context, (min.IsSameSize(scales)),
                InvalidArgument("Input min shape must be same as "
                                "scales/zero_points. Given min of shape ",
                                min.shape().DebugString(),
                                " and scales/zero_points of shape ",
                                scales.shape().DebugString()));
    OP_REQUIRES(context, (max.IsSameSize(scales)),
                InvalidArgument("Input max shape must be same as "
                                "scales/zero_points. Given max of shape ",
                                max.shape().DebugString(),
                                " and scales/zero_points of shape ",
                                scales.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, operand.shape(), &output));
    EvalQuantizedClipByValue<T>(operand, min, max, quantization_axis_, *output);
  }

 private:
  int quantization_axis_;
};

REGISTER_KERNEL_BUILDER(Name("UniformQuantizedClipByValue")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T"),
                        UniformQuantizedClipByValueOp<qint32>);

}  // namespace tensorflow
