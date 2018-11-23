/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

using errors::InvalidArgument;

template <typename T>
class RaggedRangeOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    const Tensor& starts_in = context->input(0);
    const Tensor& limits_in = context->input(1);
    const Tensor& deltas_in = context->input(2);

    // Check input tensor shapes.
    OP_REQUIRES(context, starts_in.shape().dims() <= 1,
                InvalidArgument("starts must be a scalar or vector"));
    OP_REQUIRES(context, limits_in.shape().dims() <= 1,
                InvalidArgument("limits must be a scalar or vector"));
    OP_REQUIRES(context, deltas_in.shape().dims() <= 1,
                InvalidArgument("deltas must be a scalar or vector"));

    // Determine which tensors we need to broadcast.
    bool broadcast_starts = starts_in.shape().dims() == 0;
    bool broadcast_limits = limits_in.shape().dims() == 0;
    bool broadcast_deltas = deltas_in.shape().dims() == 0;

    // nrows (number of output rows) is the size of the non-broadcast inputs,
    // or 1 if all inputs are scalars.
    std::vector<int> in_sizes;
    if (!broadcast_starts) in_sizes.push_back(starts_in.shape().dim_size(0));
    if (!broadcast_limits) in_sizes.push_back(limits_in.shape().dim_size(0));
    if (!broadcast_deltas) in_sizes.push_back(deltas_in.shape().dim_size(0));
    for (int i = 1; i < in_sizes.size(); ++i) {
      OP_REQUIRES(context, in_sizes[i] == in_sizes[i - 1],
                  InvalidArgument("starts, limits, and deltas must have the "
                                  "same shape"));
    }
    int64 nrows = in_sizes.empty() ? 1 : in_sizes[0];

    const auto& starts = starts_in.flat<T>();
    const auto& limits = limits_in.flat<T>();
    const auto& deltas = deltas_in.flat<T>();

    // Construct the rt_nested_splits tensor.
    Tensor* rt_nested_splits_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({nrows + 1}),
                                            &rt_nested_splits_out));
    auto rt_nested_splits = rt_nested_splits_out->flat<int64>();
    rt_nested_splits(0) = 0;
    for (int row = 0; row < nrows; ++row) {
      T start = broadcast_starts ? starts(0) : starts(row);
      T limit = broadcast_limits ? limits(0) : limits(row);
      T delta = broadcast_deltas ? deltas(0) : deltas(row);
      OP_REQUIRES(context, delta != 0, InvalidArgument("Requires delta != 0"));
      rt_nested_splits(row + 1) =
          rt_nested_splits(row) + RangeSize(start, limit, delta);
    }
    int64 nvals = rt_nested_splits(nrows);

    // Construct the rt_dense_values tensor.
    Tensor* rt_dense_values_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({nvals}),
                                                     &rt_dense_values_out));
    auto rt_dense_values = rt_dense_values_out->flat<T>();
    int value_index = 0;
    for (int row = 0; row < nrows; ++row) {
      int64 row_size = rt_nested_splits(row + 1) - rt_nested_splits(row);
      T value = broadcast_starts ? starts(0) : starts(row);
      T delta = broadcast_deltas ? deltas(0) : deltas(row);
      for (int64 i = 0; i < row_size; ++i) {
        rt_dense_values(value_index++) = T(value);
        value += delta;
      }
    }
  }

 private:
  // Returns the number of elements in the specified range.
  int64 RangeSize(T start, T limit, T delta) {
    if (((delta > 0) && (limit < start)) || ((delta < 0) && (limit > start))) {
      return 0;
    }
    // The following is copied from tensorflow::RangeOp::Compute().
    return (std::is_integral<T>::value
                ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                   std::abs(delta))
                : std::ceil(std::abs((limit - start) / delta)));
  }
};

#define REGISTER_CPU_KERNEL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("RaggedRange").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      RaggedRangeOp<TYPE>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
TF_CALL_int32(REGISTER_CPU_KERNEL);
TF_CALL_int64(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
