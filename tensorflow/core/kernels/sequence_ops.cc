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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

int32 GetValue(int32 v) { return v; }

template <typename T>
class RangeOp : public OpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& start_in = context->input(0);
    const Tensor& limit_in = context->input(1);
    const Tensor& delta_in = context->input(2);
    OP_REQUIRES(context, IsLegacyScalar(start_in.shape()),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().DebugString()));
    OP_REQUIRES(context, IsLegacyScalar(limit_in.shape()),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in.shape().DebugString()));
    OP_REQUIRES(context, IsLegacyScalar(delta_in.shape()),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in.shape().DebugString()));
    const int32 start = GetValue(start_in.scalar<T>()());
    const int32 limit = GetValue(limit_in.scalar<T>()());
    OP_REQUIRES(context, start <= limit,
                errors::InvalidArgument("Requires start <= limit: ", start, "/",
                                        limit));
    const int32 delta = GetValue(delta_in.scalar<T>()());
    OP_REQUIRES(context, delta > 0,
                errors::InvalidArgument("Requires delta > 0: ", delta));
    int32 size = (limit - start + delta - 1) / delta;
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({size}), &out));
    auto flat = out->flat<T>();
    int32 val = start;
    for (int32 i = 0; i < size; ++i) {
      flat(i) = T(val);
      val += delta;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Range")
                            .Device(DEVICE_CPU)
                            .HostMemory("start")
                            .HostMemory("limit")
                            .HostMemory("delta")
                            .HostMemory("output"),
                        RangeOp<int32>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Range")
                            .Device(DEVICE_GPU)
                            .HostMemory("start")
                            .HostMemory("limit")
                            .HostMemory("delta")
                            .HostMemory("output"),
                        RangeOp<int32>);
#endif  // GOOGLE_CUDA

template <typename T>
class LinSpaceOp : public OpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& start_in = context->input(0);
    const Tensor& stop_in = context->input(1);
    const Tensor& num_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(start_in.shape()),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stop_in.shape()),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_in.shape()),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in.shape().DebugString()));
    const T start = start_in.scalar<T>()();
    const T stop = stop_in.scalar<T>()();
    const int32 num = num_in.scalar<int32>()();
    OP_REQUIRES(context, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num}), &out));
    auto flat = out->flat<T>();
    if (num == 1) {
      flat(0) = start;
    } else {
      const T step = (stop - start) / (num - 1);
      for (int32 i = 0; i < num; ++i) flat(i) = start + step * i;
    }
  }
};

#define REGISTER_KERNEL(DEV, T)                       \
  REGISTER_KERNEL_BUILDER(Name("LinSpace")            \
                              .Device(DEV)            \
                              .TypeConstraint<T>("T") \
                              .HostMemory("start")    \
                              .HostMemory("stop")     \
                              .HostMemory("num")      \
                              .HostMemory("output"),  \
                          LinSpaceOp<T>);
#define REGISTER_CPU_KERNEL(T) REGISTER_KERNEL(DEVICE_CPU, T)
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);

// NOTE(touts): We register the op on GPU but it still runs on CPU
// because its inputs and outputs are tagged as HostMemory.
#define REGISTER_GPU_KERNEL(T) REGISTER_KERNEL(DEVICE_GPU, T)
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);

}  // namespace tensorflow
