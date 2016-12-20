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

#include <cmath>

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
    const T start = start_in.scalar<T>()();
    const T limit = limit_in.scalar<T>()();
    const T delta = delta_in.scalar<T>()();
    OP_REQUIRES(context, delta != 0,
                errors::InvalidArgument("Requires delta != 0: ", delta));
    if (delta > 0) {
      OP_REQUIRES(
          context, start <= limit,
          errors::InvalidArgument("Requires start <= limit when delta > 0: ",
                                  start, "/", limit));
    } else {
      OP_REQUIRES(
          context, start >= limit,
          errors::InvalidArgument("Requires start >= limit when delta < 0: ",
                                  start, "/", limit));
    }
    int64 size = (std::is_integral<T>::value
                      ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                         std::abs(delta))
                      : std::ceil(std::abs((limit - start) / delta)));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({size}), &out));
    auto flat = out->flat<T>();
    T val = start;
    for (int64 i = 0; i < size; ++i) {
      flat(i) = T(val);
      val += delta;
    }
  }
};

#define REGISTER_KERNEL(DEV, TYPE)                           \
  REGISTER_KERNEL_BUILDER(Name("Range")                      \
                              .Device(DEV)                   \
                              .HostMemory("start")           \
                              .HostMemory("limit")           \
                              .HostMemory("delta")           \
                              .HostMemory("output")          \
                              .TypeConstraint<TYPE>("Tidx"), \
                          RangeOp<TYPE>);

#define REGISTER_CPU_KERNEL(T) REGISTER_KERNEL(DEVICE_CPU, T)
#define REGISTER_GPU_KERNEL(T) REGISTER_KERNEL(DEVICE_GPU, T)
#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(T) REGISTER_KERNEL(DEVICE_SYCL, T)
TF_CALL_float(REGISTER_SYCL_KERNEL);
TF_CALL_int32(REGISTER_SYCL_KERNEL);
TF_CALL_int64(REGISTER_SYCL_KERNEL);
#endif // TENSORFLOW_USE_SYCL

TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
TF_CALL_int32(REGISTER_CPU_KERNEL);
TF_CALL_int64(REGISTER_CPU_KERNEL);

#if GOOGLE_CUDA

TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);

#endif  // GOOGLE_CUDA

#undef REGISTER_KERNEL
#undef REGISTER_CPU_KERNEL
#undef REGISTER_GPU_KERNEL

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

#define REGISTER_KERNEL(DEV, T)                              \
  REGISTER_KERNEL_BUILDER(Name("LinSpace")                   \
                              .Device(DEV)                   \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<int32>("Tidx") \
                              .HostMemory("start")           \
                              .HostMemory("stop")            \
                              .HostMemory("num")             \
                              .HostMemory("output"),         \
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
