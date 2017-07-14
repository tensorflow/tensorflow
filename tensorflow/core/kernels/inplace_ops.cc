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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SyclDevice;
#endif // TENSORFLOW_USE_SYCL

namespace functor {

template <typename Device, typename T>
Status DoParallelConcatUpdate(const Device& d, const Tensor& value,
                              int32 loc, Tensor* output) {
  auto Tvalue = value.flat_outer_dims<T>();
  auto Toutput = output->flat_outer_dims<T>();
  auto nrows = Toutput.dimension(0);
  auto r = (loc % nrows + nrows) % nrows;  // Guard index range.
  Toutput.template chip<0>(r).device(d) = Tvalue.template chip<0>(0);
  return Status::OK();
}

template <>
Status DoParallelConcat(const CPUDevice& d, const Tensor& value, int32 loc,
                        Tensor* output) {
  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                  \
  case DataTypeToEnum<type>::value: \
    return DoParallelConcatUpdate<CPUDevice, type>(d, value, loc, output);
    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_string(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ", value.dtype());
  }
}

#ifdef TENSORFLOW_USE_SYCL
template <>
Status DoParallelConcat(const SyclDevice& d, const Tensor& value, int32 loc,
                        Tensor* output) {
  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                  \
  case DataTypeToEnum<type>::value: \
    return DoParallelConcatUpdate<SyclDevice, type>(d, value, loc, output);
    TF_CALL_GPU_NUMBER_TYPES_NO_HALF(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ", value.dtype());
  }
}
#endif // TENSORFLOW_USE_SYCL

}  // end namespace functor

namespace {

template <typename Device>
class ParallelConcatUpdate : public OpKernel {
 public:
  explicit ParallelConcatUpdate(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto value = ctx->input(0);
    auto update = ctx->input(1);

    OP_REQUIRES(
        ctx, value.dims() == update.dims(),
        errors::InvalidArgument("value and update shape doesn't match: ",
                                value.shape().DebugString(), " vs. ",
                                update.shape().DebugString()));
    for (int i = 1; i < value.dims(); ++i) {
      OP_REQUIRES(
          ctx, value.dim_size(i) == update.dim_size(i),
          errors::InvalidArgument("value and update shape doesn't match ",
                                  value.shape().DebugString(), " vs. ",
                                  update.shape().DebugString()));
    }
    OP_REQUIRES(ctx, 1 == update.dim_size(0),
                errors::InvalidArgument("update shape doesn't match: ",
                                        update.shape().DebugString()));

    Tensor output = value;  // This creates an alias intentionally.
    const auto& d = ctx->eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, ::tensorflow::functor::DoParallelConcat(d, update, loc_, &output));
    ctx->set_output(0, output);
  }

 private:
  int32 loc_;
};

template <typename Device, typename T>
class ParallelConcatStart : public OpKernel {
 public:
  explicit ParallelConcatStart(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out = nullptr;
    // We do not know whether the output will be used on GPU. Setting it to be
    // gpu-compatible for now.
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &out, attr));
  }

 private:
  TensorShape shape_;
};

class FailureKernel : public OpKernel {
 public:
  explicit FailureKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   errors::Internal("Found instance of parallel_stack which "
                                    "could not be properly replaced."));
  }

  void Compute(OpKernelContext*) override {}
};

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<CPUDevice>);
TF_CALL_POD_STRING_TYPES(REGISTER)
#undef REGISTER

#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#define REGISTER_PARALLEL_CONCAT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_POD_STRING_TYPES(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_SYCL)            \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<SyclDevice, type>);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#define REGISTER_PARALLEL_CONCAT(type)                                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ParallelConcat").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_SYCL)        \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<SyclDevice>);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER)
#undef REGISTER

// Register versions that operate on int32 data on the CPU even though the op
// has been placed on the SYCL

REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")
                            .Device(DEVICE_SYCL)
                            .HostMemory("value")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ParallelConcatUpdate<CPUDevice>);
#endif // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#define REGISTER_PARALLEL_CONCAT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<GPUDevice>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER)
#undef REGISTER

// Register versions that operate on int32 data on the CPU even though the op
// has been placed on the GPU

REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ParallelConcatUpdate<CPUDevice>);
#endif

}  // end namespace
}  // end namespace tensorflow
