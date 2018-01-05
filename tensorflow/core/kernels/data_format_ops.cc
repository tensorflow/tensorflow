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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/data_format_ops.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class DataFormatDimMapOp : public OpKernel {
 public:
  explicit DataFormatDimMapOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(
        context, src_format == "NHWC",
        errors::InvalidArgument(strings::StrCat(
            "Current implementation doesn't support source data format ",
            src_format)));
    OP_REQUIRES(context, dst_format == "NCHW",
                errors::InvalidArgument(strings::StrCat(
                    "Current implementation doesn't support dst data format ",
                    dst_format)));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    functor::DataFormatDimMap<Device, T>()(context->eigen_device<Device>(),
                                           input.flat<T>(), output->flat<T>());
  }
};

template <typename Device, typename T>
class DataFormatVecPermuteOp : public OpKernel {
 public:
  explicit DataFormatVecPermuteOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(context,
                (src_format == "NHWC" && dst_format == "NCHW") ||
                    (src_format == "NCHW" && dst_format == "NHWC"),
                errors::InvalidArgument(strings::StrCat(
                    "Current implementation only supports NCHW-to-NHWC and "
                    "NHWC-to-NCHW format conversion; got source format ",
                    src_format, " and destination format ", dst_format)));
    nhwc_to_nchw_ = (src_format == "NHWC") ? true : false;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 1 || input.dims() == 2,
                errors::InvalidArgument(
                    "input must be a vector or 2D tensor, but got shape ",
                    input.shape().DebugString()));
    if (input.dims() == 1) {
      OP_REQUIRES(
          context, input.NumElements() == 4,
          errors::InvalidArgument("1D input must be of size 4, but got shape ",
                                  input.shape().DebugString()));
    } else if (input.dims() == 2) {
      OP_REQUIRES(
          context, input.dim_size(0) == 4,
          errors::InvalidArgument(
              "First dimension of 2D input must be of size 4, but got shape ",
              input.shape().DebugString()));
      OP_REQUIRES(
          context, input.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input.shape().DebugString()));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    functor::DataFormatVecPermute<Device, T>()(
        context->eigen_device<Device>(), input.flat<T>(), output->flat<T>(),
        nhwc_to_nchw_);
  }

 private:
  bool nhwc_to_nchw_;
};

#define REGISTER_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                \
  template <>                                              \
  void DataFormatDimMap<GPUDevice, T>::operator()(         \
      const GPUDevice& d, typename TTypes<T>::ConstFlat x, \
      typename TTypes<T>::Flat y);                         \
  extern template struct DataFormatDimMap<GPUDevice, T>;
#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC

#define DECLARE_GPU_SPEC(T)                                \
  template <>                                              \
  void DataFormatVecPermute<GPUDevice, T>::operator()(     \
      const GPUDevice& d, typename TTypes<T>::ConstFlat x, \
      typename TTypes<T>::Vec y, bool nhwc_to_nchw);       \
  extern template struct DataFormatVecPermute<GPUDevice, T>;
#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<GPUDevice, T>);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#define REGISTER_GPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<GPUDevice, T>);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
