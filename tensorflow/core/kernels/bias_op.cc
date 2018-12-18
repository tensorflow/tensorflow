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
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace {

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width, int32* depth,
                      int32* channel) {
  *batch = 1;
  *height = 1;
  *width = 1;
  *depth = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    *batch = static_cast<int32>(value_tensor.dim_size(0));
    *channel = static_cast<int32>(value_tensor.dim_size(1));
    *height = static_cast<int32>(value_tensor.dim_size(2));
    if (value_tensor.dims() > 3) {
      *width = static_cast<int32>(value_tensor.dim_size(3));
    }
    if (value_tensor.dims() > 4) {
      *depth = static_cast<int32>(value_tensor.dim_size(4));
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
      channel_dim = 1;  // NCHW always have channel dim in 1 (with 3, 4, 5
                        // dimensions data).
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
      int32 batch, height, width, depth, channel;
      GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                       &channel);
      switch (input.shape().dims()) {
        case 3: {
          Eigen::DSizes<int32, 3> three_dims(1, channel, 1);
          Eigen::DSizes<int32, 3> broad_cast_dims(batch, 1, height);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 3>().device(d) =
              input.tensor<T, 3>() + bias.tensor<T, 1>()
                                         .reshape(three_dims)
                                         .broadcast(broad_cast_dims);
        } break;
        case 4: {
          Eigen::DSizes<int32, 4> four_dims(1, channel, 1, 1);
          Eigen::DSizes<int32, 4> broad_cast_dims(batch, 1, height, width);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 4>().device(d) =
              input.tensor<T, 4>() +
              bias.tensor<T, 1>().reshape(four_dims).broadcast(broad_cast_dims);
        } break;
        case 5: {
          Eigen::DSizes<int32, 5> five_dims(1, channel, 1, 1, 1);
          Eigen::DSizes<int32, 5> broad_cast_dims(batch, 1, height, width,
                                                  depth);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 5>().device(d) =
              input.tensor<T, 5>() +
              bias.tensor<T, 1>().reshape(five_dims).broadcast(broad_cast_dims);
        } break;
        default:
          OP_REQUIRES(context, false,
                      errors::InvalidArgument("Only ranks up to 5 supported: ",
                                              input.shape().DebugString()));
      }
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

    int32 batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
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
        Eigen::DSizes<Eigen::Index, 3> three_dims(batch, channel,
                                                  height * width * depth);
#ifdef EIGEN_HAS_INDEX_LIST
        using idx0 = Eigen::type2index<0>;
        using idx2 = Eigen::type2index<2>;
        Eigen::IndexList<idx0, idx2> reduction_axes;
#else
        Eigen::array<Eigen::Index, 2> reduction_axes = {0, 2};
#endif
        output->template flat<T>().device(context->eigen_device<Device>()) =
            output_backprop.flat<T>()
                .template cast<typename AccumulatorType<T>::type>()
                .reshape(three_dims)
                .sum(reduction_axes)
                .template cast<T>();  // End of code by intel_tf.
      } else {
        Eigen::DSizes<Eigen::Index, 2> two_dims(batch * height * width * depth,
                                                channel);
#ifdef EIGEN_HAS_INDEX_LIST
        Eigen::IndexList<Eigen::type2index<0> > reduction_axis;
#else
        Eigen::array<Eigen::Index, 1> reduction_axis = {0};
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
    int32 batch, height, width, depth, channel;
    GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                     &channel);
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
                          output->flat<T>().data(), batch, width, height, depth,
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

struct BiasGradAutotuneGroup {
  static string name() { return "BiasGrad"; }
};

class BiasAddGradGPUConfig {
 public:
  BiasAddGradGPUConfig() : mode_(BiasAddGradGPUMode::kReduction) {}
  string ToString() const {
    if (mode_ == BiasAddGradGPUMode::kNative) {
      return "native CUDA kernel.";
    }
    if (mode_ == BiasAddGradGPUMode::kReduction) {
      return "cub reduction kernel.";
    }
    return "unknown kernel.";
  }
  BiasAddGradGPUMode get_mode() const { return mode_; }
  void set_mode(BiasAddGradGPUMode val) { mode_ = val; }

  bool operator==(const BiasAddGradGPUConfig& other) const {
    return this->mode_ == other.get_mode();
  }

  bool operator!=(const BiasAddGradGPUConfig& other) const {
    return !(*this == other);
  }

 private:
  BiasAddGradGPUMode mode_;
};

// Encapsulate all the shape information that is used in bias add grad
// operations.
class BiasAddParams {
 public:
  // We use a list to maintain both the shape value and the order (data format).
  using SpatialArray = gtl::InlinedVector<int64, 4>;
  BiasAddParams(const SpatialArray& in_shape, TensorFormat data_format,
                DataType dtype, int device_id)
      : in_shape_(in_shape),
        data_format_(data_format),
        dtype_(dtype),
        device_id_(device_id) {
    for (int64 val : in_shape_) {
      hash_code_ = Hash64Combine(hash_code_, val);
    }
    hash_code_ = Hash64Combine(hash_code_, data_format);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const BiasAddParams& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const BiasAddParams& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    // clang-format off
    return strings::StrCat(
        "(", str_util::Join(in_shape_, ", "), "), ",
        data_format_, ", ", dtype_, ", ", device_id_);
    // clang-format on
  }

 protected:
  using ParamsDataType = std::tuple<SpatialArray, TensorFormat, DataType, int>;

  ParamsDataType get_data_as_tuple() const {
    return std::make_tuple(in_shape_, data_format_, dtype_, device_id_);
  }

  uint64 hash_code_ = 0;

 private:
  SpatialArray in_shape_;
  TensorFormat data_format_;
  DataType dtype_;
  int device_id_;
};

typedef AutoTuneSingleton<BiasGradAutotuneGroup, BiasAddParams,
                          BiasAddGradGPUConfig>
    AutotuneBiasGrad;

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

  void ComputeWithCustomKernel(OpKernelContext* context,
                               const Tensor& output_backprop, int32 batch,
                               int32 width, int32 height, int32 depth,
                               int32 channel, Tensor* output) {
    BiasGradGPU<T>::compute(context->template eigen_device<Device>(),
                            output_backprop.template flat<T>().data(),
                            output->flat<T>().data(), batch, width, height,
                            depth, channel, data_format_);
  }

  void ComputeWithReduceSum(OpKernelContext* context,
                            const Tensor& output_backprop, int32 batch,
                            int32 width, int32 height, int32 depth,
                            int32 channel, Tensor* output) {
    if (data_format_ == FORMAT_NCHW) {
      int32 row_count = batch * channel;
      int32 col_count = height * width * depth;
      Tensor temp_grad_outputs;
      // For 'NCHW' format, we perform reduction twice: first HW, then N.
      TensorShape temp_grad_output_shape{row_count, col_count};
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     temp_grad_output_shape,
                                                     &temp_grad_outputs));
      BiasGradGPU<T>::DoRowReduction(
          context, temp_grad_outputs.flat<T>().data(),
          output_backprop.template flat<T>().data(), row_count, col_count);

      row_count = batch;
      col_count = channel;
      BiasGradGPU<T>::DoColReduction(context, output->flat<T>().data(),
                                     temp_grad_outputs.flat<T>().data(),
                                     row_count, col_count);
    } else {
      // For 'NHWC', we simply apply reduction once on NHW.
      int32 row_count = batch * height * width * depth;
      int32 col_count = channel;
      BiasGradGPU<T>::DoColReduction(
          context, const_cast<T*>(output->flat<T>().data()),
          reinterpret_cast<const T*>(output_backprop.template flat<T>().data()),
          row_count, col_count);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));
    int32 batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (channel == 0) return;
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    se::DeviceMemoryBase output_ptr(output->flat<T>().data(),
                                    output->NumElements() * sizeof(T));
    stream->ThenMemZero(&output_ptr, output->NumElements() * sizeof(T));
    if (output_backprop.NumElements() <= 0) return;

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = output_backprop.dtype();
    BiasAddParams bias_parameters = {
        {batch, height * width * depth, channel},
        data_format_,
        dtype,
        device_id,
    };

    // Autotune two algorithm: customized
    BiasAddGradGPUConfig algo_config;
    if (!AutotuneBiasGrad::GetInstance()->Find(bias_parameters, &algo_config)) {
      BiasGradGPUProfileResult best_result;
      // Initialize the timer.
      perftools::gputools::Timer timer(stream->parent());
      stream->InitTimer(&timer);
      stream->ThenStartTimer(&timer);
      ComputeWithCustomKernel(context, output_backprop, batch, width, height,
                              depth, channel, output);
      stream->ThenStopTimer(&timer);
      uint64 elapsed_microseconds = timer.Microseconds();
      VLOG(1) << "BiasAddGrad " << bias_parameters.ToString()
              << " Native algo latency: " << elapsed_microseconds;
      if (elapsed_microseconds < best_result.elapsed_time()) {
        best_result.set_algorithm(BiasAddGradGPUMode::kNative);
        best_result.set_elapsed_time(elapsed_microseconds);
      }

      // Try reduction and profile.
      stream->ThenStartTimer(&timer);
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
      stream->ThenStopTimer(&timer);

      elapsed_microseconds = timer.Microseconds();
      VLOG(1) << "BiasAddGrad " << bias_parameters.ToString()
              << " Reduction algo latency: " << elapsed_microseconds;
      if (elapsed_microseconds < best_result.elapsed_time()) {
        best_result.set_algorithm(BiasAddGradGPUMode::kReduction);
        best_result.set_elapsed_time(elapsed_microseconds);
      }

      algo_config.set_mode(best_result.algorithm());
      AutotuneBiasGrad::GetInstance()->Insert(bias_parameters, algo_config);

      // Results are already available during autotune, so no need to continue.
      return;
    }

    // Choose the best algorithm based on autotune results.
    if (algo_config.get_mode() == BiasAddGradGPUMode::kReduction) {
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
    } else {
      // Default to the customized kernel.
      ComputeWithCustomKernel(context, output_backprop, batch, width, height,
                              depth, channel, output);
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
