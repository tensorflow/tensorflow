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

#include "tensorflow/core/kernels/bias_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace {

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width,
                      int32* channel) {
  *batch = 1;
  *width = 1;
  *height = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    int32 channel_dim = value_tensor.dims() - 3;
    int32 height_dim = value_tensor.dims() - 2;
    int32 width_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    *height = static_cast<int32>(value_tensor.dim_size(height_dim));
    *width = static_cast<int32>(value_tensor.dim_size(width_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  }
}

template <class T>
struct AccumulatorType {
  typedef T type;
};

// float is faster on the CPU than half, and also more precise,
// so use float for the temporary accumulators.
template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

}  // namespace

template <typename Device, typename T>
class BiasOp : public BinaryOp<T> {
 public:
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));

    // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
    size_t channel_dim;
    if (data_format_ == FORMAT_NCHW) {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument(
                      "NCHW format supports only 4D input tensor."));
      channel_dim = 1;
    } else {
      channel_dim = input.shape().dims() - 1;  // End of code by intel_tf.
    }

    OP_REQUIRES(
        context,
        bias.shape().dim_size(0) == input.shape().dim_size(channel_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
    if (data_format_ == FORMAT_NCHW) {
      int32 batch, height, width, channel;
      GetBiasValueDims(input, data_format_, &batch, &height, &width, &channel);
      Eigen::DSizes<int32, 4> four_dims(1, channel, 1, 1);
      Eigen::DSizes<int32, 4> broad_cast_dims(batch, 1, height, width);
      const Device& d = context->eigen_device<Device>();
      output->tensor<T, 4>().device(d) =
          input.tensor<T, 4>() +
          bias.tensor<T, 1>().reshape(four_dims).broadcast(broad_cast_dims);
      return;
    }  // End of code by intel_tf.

    switch (input.shape().dims()) {
      case 2:
        Compute<2>(context, input, bias, output);
        break;
      case 3:
        Compute<3>(context, input, bias, output);
        break;
      case 4:
        Compute<4>(context, input, bias, output);
        break;
      case 5:
        Compute<5>(context, input, bias, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 5 supported: ",
                                            input.shape().DebugString()));
    }
  }

  // Add biases for an input matrix of rank Dims, by using the Bias.
  template <int Dims>
  void Compute(OpKernelContext* ctx, const Tensor& input, const Tensor& bias,
               Tensor* output) {
    functor::Bias<Device, T, Dims> functor;
    functor(ctx->eigen_device<Device>(), input.tensor<T, Dims>(), bias.vec<T>(),
            output->tensor<T, Dims>());
  }

 private:
  TensorFormat data_format_;
};

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      BiasOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BiasAdd").Device(DEVICE_SYCL).TypeConstraint<type>("T"),   \
      BiasOp<SYCLDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BiasAddV1").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      BiasOp<SYCLDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNEL);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class BiasGradOp : public OpKernel {
 public:
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(output_backprop.NumElements(),
                        std::numeric_limits<int32>::max()),
        errors::InvalidArgument("BiasGrad requires tensor size <= int32 max"));

    int32 batch, height, width, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (channel == 0) {
      return;  // Nothing to do
    } else if (output_backprop.NumElements() == 0) {
      // Eigen often crashes by design on empty tensors, but setZero is safe
      output->template flat<T>().setZero();
    } else {
      // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
      if (data_format_ == FORMAT_NCHW) {
        OP_REQUIRES(context, output_backprop.dims() == 4,
                    errors::InvalidArgument(
                        "NCHW format supports only 4D input/output tensor."));
        Eigen::DSizes<int, 4> four_dims(batch, channel, height, width);
#ifdef EIGEN_HAS_INDEX_LIST
        using idx0 = Eigen::type2index<0>;
        using idx2 = Eigen::type2index<2>;
        using idx3 = Eigen::type2index<3>;
        Eigen::IndexList<idx0, idx2, idx3> reduction_axes;
#else
        Eigen::array<int, 3> reduction_axes = {0, 2, 3};
#endif
        output->template flat<T>().device(context->eigen_device<Device>()) =
            output_backprop.flat<T>()
                .template cast<typename AccumulatorType<T>::type>()
                .reshape(four_dims)
                .sum(reduction_axes)
                .template cast<T>();  // End of code by intel_tf.
      } else {
        Eigen::DSizes<int, 2> two_dims(batch * height * width, channel);
#ifdef EIGEN_HAS_INDEX_LIST
        Eigen::IndexList<Eigen::type2index<0> > reduction_axis;
#else
        Eigen::array<int, 1> reduction_axis = {0};
#endif
        output->template flat<T>().device(context->eigen_device<Device>()) =
            output_backprop.flat<T>()
                .template cast<typename AccumulatorType<T>::type>()
                .reshape(two_dims)
                .sum(reduction_axis)
                .template cast<T>();
      }
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasGradOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BiasAddGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      BiasGradOp<SYCLDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNEL);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
template <typename T>
class BiasOp<GPUDevice, T> : public BinaryOp<T> {
 public:
  typedef GPUDevice Device;
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));
    int32 batch, height, width, channel;
    GetBiasValueDims(input, data_format_, &batch, &height, &width, &channel);
    OP_REQUIRES(context, bias.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel dimension "
                    "of the input tensor: ",
                    bias.shape().DebugString(), " vs. ", channel, " in ",
                    input.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() > 0) {
      BiasGPU<T>::compute(context->template eigen_device<Device>(),
                          input.flat<T>().data(), bias.flat<T>().data(),
                          output->flat<T>().data(), batch, width, height,
                          channel, data_format_);
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      BiasOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

template <typename T>
class BiasGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NCHW;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));
    int32 batch, height, width, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (channel == 0) return;
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    se::DeviceMemoryBase output_ptr(output->flat<T>().data(),
                                    output->NumElements() * sizeof(T));
    stream->ThenMemZero(&output_ptr, output->NumElements() * sizeof(T));
    if (output_backprop.NumElements() > 0) {
      BiasGradGPU<T>::compute(context->template eigen_device<Device>(),
                              output_backprop.template flat<T>().data(),
                              output->flat<T>().data(), batch, width, height,
                              channel, data_format_);
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
