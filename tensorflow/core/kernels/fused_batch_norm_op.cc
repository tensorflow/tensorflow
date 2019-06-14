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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/gpus/cudnn/cudnn.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif  // GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

enum class FusedBatchNormActivationMode { kIdentity, kRelu };

string ToString(FusedBatchNormActivationMode activation_mode) {
  switch (activation_mode) {
    case FusedBatchNormActivationMode::kIdentity:
      return "Identity";
    case FusedBatchNormActivationMode::kRelu:
      return "Relu";
  }
}

Status ParseActivationMode(OpKernelConstruction* context,
                           FusedBatchNormActivationMode* activation_mode) {
  string activation_mode_str;
  TF_RETURN_IF_ERROR(context->GetAttr("activation_mode", &activation_mode_str));

  if (activation_mode_str == "Identity") {
    *activation_mode = FusedBatchNormActivationMode::kIdentity;
    return Status::OK();
  }
  if (activation_mode_str == "Relu") {
    *activation_mode = FusedBatchNormActivationMode::kRelu;
    return Status::OK();
  }
  return errors::InvalidArgument("Unsupported activation mode: ",
                                 activation_mode_str);
}

// Functor used by FusedBatchNormOp to do the computations.
template <typename Device, typename T, typename U>
struct FusedBatchNorm;
// Functor used by FusedBatchNormGradOp to do the computations when
// is_training=True.
template <typename Device, typename T, typename U>
struct FusedBatchNormGrad;

#if GOOGLE_CUDA
using se::DeviceMemory;
using se::ScratchAllocator;
using se::Stream;
using se::port::StatusOr;

template <typename U, typename T>
DeviceMemory<U> CastDeviceMemory(Tensor* tensor) {
  return DeviceMemory<U>::MakeFromByteSize(
      tensor->template flat<T>().data(),
      tensor->template flat<T>().size() * sizeof(T));
}

// A helper to allocate temporary scratch memory for Cudnn BatchNormEx ops. It
// takes the ownership of the underlying memory. The expectation is that the
// memory should be alive for the span of the Cudnn BatchNormEx itself.
template <typename T>
class CudnnBatchNormAllocatorInTemp : public ScratchAllocator {
 public:
  ~CudnnBatchNormAllocatorInTemp() override = default;

  explicit CudnnBatchNormAllocatorInTemp(OpKernelContext* context)
      : context_(context) {}

  int64 GetMemoryLimitInBytes(Stream* stream) override {
    return std::numeric_limits<int64>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(Stream* stream,
                                              int64 byte_size) override {
    Tensor temporary_memory;
    const DataType tf_data_type = DataTypeToEnum<T>::v();
    int64 allocate_count =
        Eigen::divup(byte_size, static_cast<int64>(sizeof(T)));
    Status allocation_status(context_->allocate_temp(
        tf_data_type, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return allocation_status;
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory.template flat<T>().data(),
        temporary_memory.template flat<T>().size() * sizeof(T));
  }

  int64 TotalByteSize() const { return total_byte_size_; }

  Tensor get_allocated_tensor(int index) const {
    return allocated_tensors_[index];
  }

 private:
  int64 total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  std::vector<Tensor> allocated_tensors_;
};

// A helper to allocate memory for Cudnn BatchNormEx as a kernel output. It is
// used by forward pass kernel to feed the output to the backward pass.
// The memory is expected to live long enough after the backward pass is
// finished.
template <typename T>
class CudnnBatchNormAllocatorInOutput : public ScratchAllocator {
 public:
  ~CudnnBatchNormAllocatorInOutput() override {
    if (!output_allocated) {
      Tensor* dummy_reserve_space = nullptr;
      OP_REQUIRES_OK(context_, context_->allocate_output(output_index_, {},
                                                         &dummy_reserve_space));
    }
  }

  CudnnBatchNormAllocatorInOutput(OpKernelContext* context, int output_index)
      : context_(context), output_index_(output_index) {}

  int64 GetMemoryLimitInBytes(Stream* stream) override {
    return std::numeric_limits<int64>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(Stream* stream,
                                              int64 byte_size) override {
    output_allocated = true;
    DCHECK(total_byte_size_ == 0)
        << "Reserve space allocator can only be called once";
    int64 allocate_count =
        Eigen::divup(byte_size, static_cast<int64>(sizeof(T)));

    Tensor* temporary_memory = nullptr;
    Status allocation_status(context_->allocate_output(
        output_index_, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return allocation_status;
    }
    total_byte_size_ += byte_size;
    auto memory_uint8 = DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory->template flat<T>().data(),
        temporary_memory->template flat<T>().size() * sizeof(T));
    return StatusOr<DeviceMemory<uint8>>(memory_uint8);
  }

  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  int output_index_;
  bool output_allocated = false;
};
#else
// A dummy class for the non-GPU environment. Its child classes
// CudnnBatchNormAllocatorInTemp and CudnnBatchNormAllocatorInOutput are used
// to make the non-GPU operations compatible with GPU ones.
class ScratchAllocator {
 public:
  virtual ~ScratchAllocator() {}
};

template <typename T>
class CudnnBatchNormAllocatorInTemp : public ScratchAllocator {
 public:
  explicit CudnnBatchNormAllocatorInTemp(OpKernelContext* context) {}
};

template <typename T>
class CudnnBatchNormAllocatorInOutput : public ScratchAllocator {
 public:
  ~CudnnBatchNormAllocatorInOutput() override {
    Tensor* dummy_reserve_space = nullptr;
    OP_REQUIRES_OK(context_, context_->allocate_output(output_index_, {},
                                                       &dummy_reserve_space));
  }
  CudnnBatchNormAllocatorInOutput(OpKernelContext* context, int output_index)
      : context_(context), output_index_(output_index) {}

 private:
  OpKernelContext* context_;  // not owned
  int output_index_;
};
#endif  // GOOGLE_CUDA

template <bool IsSame, typename Y, typename X, typename T>
struct CastIfNecessary {
  static inline void process(
      Y& y, X& x_shifted, const Eigen::DSizes<Eigen::Index, 2>& rest_by_depth,
      const CPUDevice& d) {
    y.reshape(rest_by_depth).device(d) = x_shifted.template cast<T>();
  }
};

template <typename Y, typename X, typename T>
struct CastIfNecessary<true, Y, X, T> {
  static inline void process(
      Y& y, X& x_shifted, const Eigen::DSizes<Eigen::Index, 2>& rest_by_depth,
      const CPUDevice& d) {
    y.reshape(rest_by_depth).device(d) = x_shifted;
  }
};

template <typename T, typename U>
struct FusedBatchNorm<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  const Tensor& estimated_mean_input,
                  const Tensor& estimated_variance_input,
                  const Tensor& side_input, U epsilon,
                  FusedBatchNormActivationMode activation_mode,
                  Tensor* y_output, Tensor* batch_mean_output,
                  Tensor* batch_var_output, Tensor* saved_mean_output,
                  Tensor* saved_var_output, TensorFormat tensor_format,
                  ScratchAllocator* reserve_space_allocator,
                  ScratchAllocator* workspace_allocator, bool is_training) {
    OP_REQUIRES(context, tensor_format == FORMAT_NHWC,
                errors::Internal("The CPU implementation of FusedBatchNorm "
                                 "only supports NHWC tensor format for now."));
    OP_REQUIRES(context, side_input.dim_size(0) == 0,
                errors::Internal(
                    "The CPU implementation of FusedBatchNorm does not support "
                    "side input."));
    OP_REQUIRES(context,
                activation_mode == FusedBatchNormActivationMode::kIdentity,
                errors::Internal("The CPU implementation of FusedBatchNorm "
                                 "does not support activations."));
    typename TTypes<T, 4>::ConstTensor x(x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec offset(offset_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_mean(estimated_mean_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_variance(
        estimated_variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor y(y_output->tensor<T, 4>());
    typename TTypes<U>::Vec batch_mean(batch_mean_output->vec<U>());
    typename TTypes<U>::Vec batch_var(batch_var_output->vec<U>());
    typename TTypes<U>::Vec saved_mean(saved_mean_output->vec<U>());
    typename TTypes<U>::Vec saved_var(saved_var_output->vec<U>());

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    const int depth = x.dimension(3);
    const int size = x.size();
    const int rest_size = size / depth;
    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduce_dims({0});
    Eigen::array<int, 2> bcast_spec({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0>> reduce_dims;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> bcast_spec;
    bcast_spec.set(0, rest_size);
#endif

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
    const int rest_size_minus_one = (rest_size > 1) ? (rest_size - 1) : 1;
    U rest_size_inv = static_cast<U>(1.0f / static_cast<U>(rest_size));
    // This adjustment is for Bessel's correction
    U rest_size_adjust =
        static_cast<U>(rest_size) / static_cast<U>(rest_size_minus_one);

    Eigen::Tensor<U, 1, Eigen::RowMajor> mean(depth);
    Eigen::Tensor<U, 1, Eigen::RowMajor> variance(depth);
    if (is_training) {
      mean.device(d) = (x_rest_by_depth.sum(reduce_dims) * rest_size_inv);
      batch_mean.device(d) = mean;
      saved_mean.device(d) = mean;
    } else {
      mean.device(d) = estimated_mean;
    }

    auto x_centered =
        x_rest_by_depth - mean.reshape(one_by_depth).broadcast(bcast_spec);

    if (is_training) {
      variance.device(d) = x_centered.square().sum(reduce_dims) * rest_size_inv;
      batch_var.device(d) = variance * rest_size_adjust;
      saved_var.device(d) = variance;
    } else {
      variance.device(d) = estimated_variance;
    }

    auto scaling_factor = ((variance + epsilon).rsqrt() * scale)
                              .eval()
                              .reshape(one_by_depth)
                              .broadcast(bcast_spec);
    auto x_scaled = x_centered * scaling_factor;
    auto x_shifted =
        x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec);

    // Explicitly checks the types of T and U and only casts x_shifted when
    // T != U. (Not doing so caused a 35-50% performance slowdown for
    // some compiler flags.)
    CastIfNecessary<std::is_same<T, U>::value, decltype(y), decltype(x_shifted),
                    T>::process(y, x_shifted, rest_by_depth, d);
  }
};

template <typename T, typename U>
struct FusedBatchNormGrad<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& mean_input, const Tensor& variance_input,
                  U epsilon, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output, Tensor* offset_backprop_output,
                  const Tensor* reserve_space,
                  ScratchAllocator* workspace_allocator,
                  TensorFormat tensor_format) {
    OP_REQUIRES(context, tensor_format == FORMAT_NHWC,
                errors::Internal("The CPU implementation of FusedBatchNormGrad "
                                 "only supports NHWC tensor format for now."));
    typename TTypes<T, 4>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 4>());
    typename TTypes<T, 4>::ConstTensor x(x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec mean(mean_input.vec<U>());
    typename TTypes<U>::ConstVec variance(variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor x_backprop(x_backprop_output->tensor<T, 4>());
    typename TTypes<U>::Vec scale_backprop(scale_backprop_output->vec<U>());
    typename TTypes<U>::Vec offset_backprop(offset_backprop_output->vec<U>());

    // Note: the following formulas are used to compute the gradients for
    // back propagation.
    // x_backprop = scale * rsqrt(variance + epsilon) *
    //              [y_backprop - mean(y_backprop) - (x - mean(x)) *
    //              mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
    // scale_backprop = sum(y_backprop *
    //                  (x - mean(x)) * rsqrt(variance + epsilon))
    // offset_backprop = sum(y_backprop)

    const CPUDevice& d = context->eigen_device<CPUDevice>();
    const int depth = x.dimension(3);
    const int size = x.size();
    const int rest_size = size / depth;
    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduce_dims({0});
    Eigen::array<int, 2> bcast_spec({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0>> reduce_dims;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> bcast_spec;
    bcast_spec.set(0, rest_size);
#endif

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
    U rest_size_inv = static_cast<U>(1.0f / static_cast<U>(rest_size));

    auto x_mean_rest_by_depth =
        mean.reshape(one_by_depth).broadcast(bcast_spec);
    auto x_centered = (x_rest_by_depth - x_mean_rest_by_depth).eval();
    auto coef0 = (variance + epsilon).rsqrt();
    auto coef0_rest_by_depth =
        coef0.eval().reshape(one_by_depth).broadcast(bcast_spec);
    auto x_scaled = x_centered * coef0_rest_by_depth;

    auto y_backprop_rest_by_depth =
        y_backprop.eval().reshape(rest_by_depth).template cast<U>();
    scale_backprop.device(d) =
        (y_backprop_rest_by_depth * x_scaled).sum(reduce_dims);
    auto y_backprop_sum = y_backprop_rest_by_depth.sum(reduce_dims);
    offset_backprop.device(d) = y_backprop_sum;

    auto y_backprop_sum_one_by_depth =
        y_backprop_sum.eval().reshape(one_by_depth);
    auto y_backprop_mean_one_by_depth =
        y_backprop_sum_one_by_depth * rest_size_inv;
    auto y_backprop_mean_rest_by_depth =
        y_backprop_mean_one_by_depth.broadcast(bcast_spec);
    auto y_backprop_centered =
        y_backprop_rest_by_depth - y_backprop_mean_rest_by_depth;
    auto coef1 =
        (scale * coef0).eval().reshape(one_by_depth).broadcast(bcast_spec);
    auto coef2 = (coef0.square() *
                  (y_backprop_rest_by_depth * x_centered).mean(reduce_dims))
                     .eval()
                     .reshape(one_by_depth)
                     .broadcast(bcast_spec);
    x_backprop.reshape(rest_by_depth).device(d) =
        (coef1 * (y_backprop_centered - x_centered * coef2)).template cast<T>();
  }
};

#ifndef GOOGLE_CUDA
namespace {
// See implementation under GOOGLE_CUDA #ifdef below.
bool BatchnormSpatialPersistentEnabled() { return false; }
}  // namespace
#endif

#if GOOGLE_CUDA

namespace {

se::dnn::ActivationMode AsDnnActivationMode(
    const FusedBatchNormActivationMode activation_mode) {
  switch (activation_mode) {
    case FusedBatchNormActivationMode::kIdentity:
      return se::dnn::ActivationMode::kNone;
    case FusedBatchNormActivationMode::kRelu:
      return se::dnn::ActivationMode::kRelu;
  }
}

// NOTE(ezhulenev): See `BatchnormSpatialPersistentEnabled` documentation in the
// `cuda_dnn.cc` for details.
bool BatchnormSpatialPersistentEnabled() {
#if CUDNN_VERSION >= 7402
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
        /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
#else
  return false;
#endif
}
}  // namespace

template <typename T, typename U>
struct FusedBatchNorm<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x,
                  const Tensor& scale, const Tensor& offset,
                  const Tensor& estimated_mean,
                  const Tensor& estimated_variance, const Tensor& side_input,
                  U epsilon, FusedBatchNormActivationMode activation_mode,
                  Tensor* y, Tensor* batch_mean, Tensor* batch_var,
                  Tensor* saved_mean, Tensor* saved_inv_var,
                  TensorFormat tensor_format,
                  ScratchAllocator* reserve_space_allocator,
                  ScratchAllocator* workspace_allocator, bool is_training) {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available"));

    // TODO(ezhulenev): cuDNN doesn't support side input and activation in
    // inference mode. Write custom cuda kernel for that.
    if (!is_training) {
      OP_REQUIRES(
          context, side_input.dim_size(0) == 0,
          errors::Internal(
              "The GPU implementation of FusedBatchNorm does not support "
              "side input in inference mode."));
      OP_REQUIRES(
          context, activation_mode == FusedBatchNormActivationMode::kIdentity,
          errors::Internal("The GPU implementation of FusedBatchNorm "
                           "does not support activations in inference mode."));
    }

    const int64 batch_size = GetTensorDim(x, tensor_format, 'N');
    const int64 channels = GetTensorDim(x, tensor_format, 'C');
    const int64 height = GetTensorDim(x, tensor_format, 'H');
    const int64 width = GetTensorDim(x, tensor_format, 'W');

    // We have reserve_space_3 output only in FusedBatchNormV3 op, and in this
    // case we pass non-nullptr allocators.
    const bool has_reserve_space_3 =
        reserve_space_allocator != nullptr && workspace_allocator != nullptr;

    // Check if cuDNN batch normalization has a fast NHWC implementation:
    //   (1) In inference mode it's always fast.
    //   (2) Tensorflow enabled batchnorm spatial persistence, and
    //       FusedBatchNormV3 passed non-null allocators.
    const bool fast_nhwc_batch_norm =
        !is_training ||
        (BatchnormSpatialPersistentEnabled() &&
         DataTypeToEnum<T>::value == DT_HALF && has_reserve_space_3);

    // If input tensor is in NHWC format, and we have a fast cuDNN
    // implementation, there is no need to do data format conversion.
    TensorFormat compute_format =
        fast_nhwc_batch_norm && tensor_format == FORMAT_NHWC ? FORMAT_NHWC
                                                             : FORMAT_NCHW;

    VLOG(2) << "FusedBatchNorm:"
            << " batch_size: " << batch_size << " channels: " << channels
            << " height: " << height << " width:" << width
            << " x shape: " << x.shape().DebugString()
            << " scale shape: " << scale.shape().DebugString()
            << " offset shape: " << offset.shape().DebugString()
            << " side input shape: " << side_input.shape().DebugString()
            << " activation mode: " << ToString(activation_mode)
            << " tensor format: " << ToString(tensor_format)
            << " compute format: " << ToString(compute_format);

    // If input is empty, return NaN mean/variance
    if (x.shape().num_elements() == 0) {
      functor::SetNanFunctor<U> f;
      f(context->eigen_device<GPUDevice>(), batch_mean->flat<U>());
      f(context->eigen_device<GPUDevice>(), batch_var->flat<U>());
      return;
    }

    Tensor x_maybe_transformed = x;
    Tensor x_transformed;
    Tensor y_transformed;
    se::DeviceMemory<T> y_ptr;

    if (tensor_format == compute_format) {
      y_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*y);
    } else if (tensor_format == FORMAT_NHWC && compute_format == FORMAT_NCHW) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(compute_format, batch_size,
                                                  height, width, channels),
                                  &x_transformed));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(x_maybe_transformed).tensor<T, 4>(),
          x_transformed.tensor<T, 4>());
      x_maybe_transformed = x_transformed;

      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(compute_format, batch_size,
                                                  height, width, channels),
                                  &y_transformed));
      y_ptr = StreamExecutorUtil::AsDeviceMemory<T>(y_transformed);
    } else {
      context->SetStatus(errors::Internal(
          "Unsupported tensor format: ", ToString(tensor_format),
          " and compute format: ", ToString(compute_format)));
      return;
    }

    const se::dnn::DataLayout data_layout =
        compute_format == FORMAT_NHWC ? se::dnn::DataLayout::kBatchYXDepth
                                      : se::dnn::DataLayout::kBatchDepthYX;

    se::dnn::BatchDescriptor x_desc;
    x_desc.set_count(batch_size)
        .set_feature_map_count(channels)
        .set_height(height)
        .set_width(width)
        .set_layout(data_layout);

    se::dnn::BatchDescriptor scale_offset_desc;
    scale_offset_desc.set_count(1)
        .set_feature_map_count(channels)
        .set_height(1)
        .set_width(1)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    auto x_ptr = StreamExecutorUtil::AsDeviceMemory<T>(x_maybe_transformed);
    auto scale_ptr = StreamExecutorUtil::AsDeviceMemory<U>(scale);
    auto offset_ptr = StreamExecutorUtil::AsDeviceMemory<U>(offset);
    auto estimated_mean_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(estimated_mean);
    auto estimated_variance_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(estimated_variance);
    auto side_input_ptr = StreamExecutorUtil::AsDeviceMemory<U>(side_input);
    auto batch_mean_ptr = StreamExecutorUtil::AsDeviceMemory<U>(*batch_mean);

    auto batch_var_ptr = StreamExecutorUtil::AsDeviceMemory<U>(*batch_var);
    auto saved_mean_ptr = StreamExecutorUtil::AsDeviceMemory<U>(*saved_mean);
    auto saved_inv_var_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(*saved_inv_var);

    GPUDevice d = context->eigen_device<GPUDevice>();
    using se::DeviceMemory;
    Tensor inv_var;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<U>::value,
                                        estimated_variance.shape(), &inv_var));
    auto inv_var_ptr = StreamExecutorUtil::AsDeviceMemory<U>(inv_var);
    std::function<const DeviceMemory<U>&()> var_to_inv_var =
        [d, epsilon, estimated_variance,
         &inv_var_ptr]() -> const DeviceMemory<U>& {
      auto estimated_variance_ptr =
          StreamExecutorUtil::AsDeviceMemory<U>(estimated_variance);
      const U* variance =
          static_cast<const U*>(estimated_variance_ptr.opaque());
      U* inv_variance = static_cast<U*>(inv_var_ptr.opaque());
      int channels = inv_var_ptr.ElementCount();
      VarianceToInvVariance<U>()(d, variance, epsilon, channels, inv_variance);
      return inv_var_ptr;
    };
    const int64 sample_size = batch_size * height * width;
    std::function<void()> inv_var_to_var = [d, &batch_var_ptr, epsilon,
                                            sample_size]() {
      U* variance = static_cast<U*>(batch_var_ptr.opaque());
      int channels = batch_var_ptr.ElementCount();
      InvVarianceToVariance<U>()(d, epsilon, sample_size, channels, variance);
    };

    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationForward(
                x_ptr, scale_ptr, offset_ptr, estimated_mean_ptr,
                estimated_variance_ptr, side_input_ptr, x_desc,
                scale_offset_desc, static_cast<double>(epsilon),
                AsDnnActivationMode(activation_mode), &y_ptr, &batch_mean_ptr,
                &batch_var_ptr, &saved_mean_ptr, &saved_inv_var_ptr,
                is_training, std::move(var_to_inv_var),
                std::move(inv_var_to_var), reserve_space_allocator,
                workspace_allocator)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(
          errors::Internal("cuDNN launch failure : input shape (",
                           x.shape().DebugString(), ")"));
    }

    if (tensor_format == FORMAT_NHWC && compute_format == FORMAT_NCHW) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(y_transformed).tensor<T, 4>(),
          y->tensor<T, 4>());
    }
  }
};

template <typename T, typename U>
struct FusedBatchNormGrad<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop,
                  const Tensor& x, const Tensor& scale, const Tensor& mean,
                  const Tensor& inv_variance, U epsilon, Tensor* x_backprop,
                  Tensor* scale_backprop, Tensor* offset_backprop,
                  const Tensor* reserve_space,
                  ScratchAllocator* workspace_allocator,
                  TensorFormat tensor_format) {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available"));

    const int64 batch_size = GetTensorDim(x, tensor_format, 'N');
    const int64 channels = GetTensorDim(x, tensor_format, 'C');
    const int64 height = GetTensorDim(x, tensor_format, 'H');
    const int64 width = GetTensorDim(x, tensor_format, 'W');

    // Check if cuDNN batch normalization has a fast NHWC implementation:
    //   (1) Tensorflow enabled batchnorm spatial persistence, and
    //       FusedBatchNormGradV3 passed non-null reserve space and allocator.
    const bool fast_nhwc_batch_norm = BatchnormSpatialPersistentEnabled() &&
                                      DataTypeToEnum<T>::value == DT_HALF &&
                                      reserve_space != nullptr &&
                                      workspace_allocator != nullptr;

    // If input tensor is in NHWC format, and we have a fast cuDNN
    // implementation, there is no need to do data format conversion.
    TensorFormat compute_format =
        fast_nhwc_batch_norm && tensor_format == FORMAT_NHWC ? FORMAT_NHWC
                                                             : FORMAT_NCHW;

    VLOG(2) << "FusedBatchNormGrad:"
            << " batch_size: " << batch_size << " channels: " << channels
            << " height: " << height << " width: " << width
            << " y_backprop shape: " << y_backprop.shape().DebugString()
            << " x shape: " << x.shape().DebugString()
            << " scale shape: " << scale.shape().DebugString()
            << " tensor format: " << ToString(tensor_format)
            << " compute format: " << ToString(compute_format);

    // Inputs
    Tensor y_backprop_maybe_transformed = y_backprop;
    Tensor x_maybe_transformed = x;
    Tensor y_backprop_transformed;
    Tensor x_transformed;

    // Outputs
    Tensor x_backprop_transformed;
    se::DeviceMemory<T> x_backprop_ptr;

    if (tensor_format == compute_format) {
      x_backprop_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*x_backprop);
    } else if (tensor_format == FORMAT_NHWC && compute_format == FORMAT_NCHW) {
      // Transform inputs from 'NHWC' to 'NCHW'
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &y_backprop_transformed));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(y_backprop_maybe_transformed)
              .tensor<T, 4>(),
          y_backprop_transformed.tensor<T, 4>());
      y_backprop_maybe_transformed = y_backprop_transformed;

      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &x_transformed));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(x_maybe_transformed).tensor<T, 4>(),
          x_transformed.tensor<T, 4>());
      x_maybe_transformed = x_transformed;

      // Allocate memory for transformed outputs in 'NCHW'
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &x_backprop_transformed));
      x_backprop_ptr =
          StreamExecutorUtil::AsDeviceMemory<T>(x_backprop_transformed);
    } else {
      context->SetStatus(errors::Internal(
          "Unsupported tensor format: ", ToString(tensor_format),
          " and compute format: ", ToString(compute_format)));
      return;
    }

    const se::dnn::DataLayout data_layout =
        compute_format == FORMAT_NHWC ? se::dnn::DataLayout::kBatchYXDepth
                                      : se::dnn::DataLayout::kBatchDepthYX;

    se::dnn::BatchDescriptor x_desc;
    x_desc.set_count(batch_size)
        .set_feature_map_count(channels)
        .set_height(height)
        .set_width(width)
        .set_layout(data_layout);

    se::dnn::BatchDescriptor scale_offset_desc;
    scale_offset_desc.set_count(1)
        .set_feature_map_count(channels)
        .set_height(1)
        .set_width(1)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    auto y_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(y_backprop_maybe_transformed);
    auto x_ptr = StreamExecutorUtil::AsDeviceMemory<T>(x_maybe_transformed);
    auto scale_ptr = StreamExecutorUtil::AsDeviceMemory<U>(scale);
    auto mean_ptr = StreamExecutorUtil::AsDeviceMemory<U>(mean);
    auto inv_variance_ptr = StreamExecutorUtil::AsDeviceMemory<U>(inv_variance);
    auto scale_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(*scale_backprop);
    auto offset_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(*offset_backprop);

    // the cudnn kernel outputs inverse variance in forward and reuse it in
    // backward
    DeviceMemory<uint8>* reserve_space_data = nullptr;
    if (reserve_space != nullptr && reserve_space->dims() != 0) {
      auto reserve_space_uint8 = functor::CastDeviceMemory<uint8, U>(
          const_cast<Tensor*>(reserve_space));
      reserve_space_data = &reserve_space_uint8;
    }
    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationBackward(
                y_backprop_ptr, x_ptr, scale_ptr, mean_ptr, inv_variance_ptr,
                x_desc, scale_offset_desc, static_cast<double>(epsilon),
                &x_backprop_ptr, &scale_backprop_ptr, &offset_backprop_ptr,
                reserve_space_data, workspace_allocator)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(
          errors::Internal("cuDNN launch failure : input shape (",
                           x.shape().DebugString(), ")"));
    }
    if (tensor_format == FORMAT_NHWC && compute_format == FORMAT_NCHW) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(x_backprop_transformed).tensor<T, 4>(),
          x_backprop->tensor<T, 4>());
    }
  }
};

// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, U)                                           \
  template <>                                                            \
  void FusedBatchNormFreezeGrad<GPUDevice, T, U>::operator()(            \
      const GPUDevice& d, const Tensor& y_backprop_input,                \
      const Tensor& x_input, const Tensor& scale_input,                  \
      const Tensor& mean_input, const Tensor& variance_input, U epsilon, \
      Tensor* x_backprop_output, Tensor* scale_backprop_output,          \
      Tensor* offset_backprop_output, typename TTypes<U>::Vec scratch1,  \
      typename TTypes<U>::Vec scratch2);                                 \
  extern template struct FusedBatchNormFreezeGrad<GPUDevice, T, U>;
DECLARE_GPU_SPEC(float, float);
DECLARE_GPU_SPEC(Eigen::half, float);

#endif  // GOOGLE_CUDA
}  // namespace functor

template <typename Device, typename T, typename U>
class FusedBatchNormOpBase : public OpKernel {
  using FbnActivationMode = functor::FusedBatchNormActivationMode;

 protected:
  explicit FusedBatchNormOpBase(OpKernelConstruction* context,
                                bool is_batch_norm_ex = false)
      : OpKernel(context), empty_side_input_(DataTypeToEnum<T>::value, {0}) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (!is_batch_norm_ex) {
      has_side_input_ = false;
      activation_mode_ = FbnActivationMode::kIdentity;
    } else {
      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode_));

      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(context, num_side_inputs >= 0 && num_side_inputs <= 1,
                  errors::InvalidArgument(
                      "FusedBatchNorm accepts at most one side input."));
      has_side_input_ = (num_side_inputs == 1);
      if (has_side_input_) {
        OP_REQUIRES(
            context, activation_mode_ != FbnActivationMode::kIdentity,
            errors::InvalidArgument("Identity activation is not supported with "
                                    "non-empty side input"));
      }
    }

    if (activation_mode_ != FbnActivationMode::kIdentity) {
      OP_REQUIRES(
          context, is_training_,
          errors::InvalidArgument("FusedBatchNorm with activation supported "
                                  "only for is_training=True."));
      // NOTE(ezhulenev): Following requirements are coming from implementation
      // details of cudnnBatchNormalizationForwardTrainingEx.
      OP_REQUIRES(context, DataTypeToEnum<T>::value == DT_HALF,
                  errors::InvalidArgument("FusedBatchNorm with activation "
                                          "supports only DT_HALF data type."));
      OP_REQUIRES(context, tensor_format_ == FORMAT_NHWC,
                  errors::InvalidArgument("FusedBatchNorm with activation "
                                          "supports only NHWC tensor format."));
      OP_REQUIRES(context, functor::BatchnormSpatialPersistentEnabled(),
                  errors::InvalidArgument(
                      "FusedBatchNorm with activation must run with cuDNN "
                      "spatial persistence mode enabled."));
    }
  }

  // If use_reserved_space is true, we need to handle the 5th output (a reserved
  // space) and a new cudnn batch norm will be called if the version > 7.4.2.
  // If use_reserved_space is false, we don't have 5th output.
  virtual void ComputeWithReservedSpace(OpKernelContext* context,
                                        bool use_reserved_space) {
    const Tensor& x = context->input(0);
    const Tensor& scale = context->input(1);
    const Tensor& offset = context->input(2);
    const Tensor& estimated_mean = context->input(3);
    const Tensor& estimated_variance = context->input(4);
    const Tensor& side_input =
        has_side_input_ ? context->input(5) : empty_side_input_;

    OP_REQUIRES(context, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    OP_REQUIRES(context, estimated_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        estimated_mean.shape().DebugString()));
    OP_REQUIRES(
        context, estimated_variance.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                estimated_variance.shape().DebugString()));
    if (has_side_input_) {
      OP_REQUIRES(context, side_input.shape() == x.shape(),
                  errors::InvalidArgument(
                      "side_input shape must be equal to input shape: ",
                      side_input.shape().DebugString(),
                      " != ", x.shape().DebugString()));
    }

    if (activation_mode_ != FbnActivationMode::kIdentity) {
      // NOTE(ezhulenev): This requirement is coming from implementation
      // details of cudnnBatchNormalizationForwardTrainingEx.
      OP_REQUIRES(
          context, x.dim_size(3) % 4 == 0,
          errors::InvalidArgument("FusedBatchNorm with activation requires "
                                  "channel dimension to be a multiple of 4."));
    }

    if (is_training_) {
      OP_REQUIRES(
          context, estimated_mean.dim_size(0) == 0,
          errors::InvalidArgument("estimated_mean must be empty for training",
                                  estimated_mean.shape().DebugString()));
      OP_REQUIRES(context, estimated_variance.dim_size(0) == 0,
                  errors::InvalidArgument(
                      "estimated_variance must be empty for training",
                      estimated_variance.shape().DebugString()));
    }

    Tensor* y = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, x.shape(), &y));
    Tensor* batch_mean = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, scale.shape(), &batch_mean));
    Tensor* batch_var = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, scale.shape(), &batch_var));
    Tensor* saved_mean = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, scale.shape(), &saved_mean));
    Tensor* saved_maybe_inv_var = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(4, scale.shape(),
                                                     &saved_maybe_inv_var));

    if (!use_reserved_space) {
      functor::FusedBatchNorm<Device, T, U>()(
          context, x, scale, offset, estimated_mean, estimated_variance,
          side_input, epsilon_, activation_mode_, y, batch_mean, batch_var,
          saved_mean, saved_maybe_inv_var, tensor_format_, nullptr, nullptr,
          is_training_);
    } else {
      functor::CudnnBatchNormAllocatorInOutput<U> reserve_space_allocator(
          context, 5);
      functor::CudnnBatchNormAllocatorInTemp<uint8> workspace_allocator(
          context);
      functor::FusedBatchNorm<Device, T, U>()(
          context, x, scale, offset, estimated_mean, estimated_variance,
          side_input, epsilon_, activation_mode_, y, batch_mean, batch_var,
          saved_mean, saved_maybe_inv_var, tensor_format_,
          &reserve_space_allocator, &workspace_allocator, is_training_);
    }
  }

 private:
  U epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_;
  FbnActivationMode activation_mode_;
  Tensor empty_side_input_;
};

template <typename Device, typename T, typename U>
class FusedBatchNormOp : public FusedBatchNormOpBase<Device, T, U> {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* context)
      : FusedBatchNormOpBase<Device, T, U>(context) {}

  void Compute(OpKernelContext* context) override {
    FusedBatchNormOpBase<Device, T, U>::ComputeWithReservedSpace(context,
                                                                 false);
  }
};

template <typename Device, typename T, typename U>
class FusedBatchNormOpV3 : public FusedBatchNormOpBase<Device, T, U> {
 public:
  explicit FusedBatchNormOpV3(OpKernelConstruction* context)
      : FusedBatchNormOpBase<Device, T, U>(context) {}

  void Compute(OpKernelContext* context) override {
    FusedBatchNormOpBase<Device, T, U>::ComputeWithReservedSpace(context, true);
  }
};

template <typename Device, typename T, typename U>
class FusedBatchNormOpEx : public FusedBatchNormOpBase<Device, T, U> {
  static constexpr bool kWithSideInputAndActivation = true;

 public:
  explicit FusedBatchNormOpEx(OpKernelConstruction* context)
      : FusedBatchNormOpBase<Device, T, U>(context,
                                           kWithSideInputAndActivation) {}

  void Compute(OpKernelContext* context) override {
    FusedBatchNormOpBase<Device, T, U>::ComputeWithReservedSpace(context, true);
  }
};

template <typename Device, typename T, typename U>
class FusedBatchNormGradOpBase : public OpKernel {
 protected:
  explicit FusedBatchNormGradOpBase(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  virtual void ComputeWithReservedSpace(OpKernelContext* context,
                                        bool use_reserved_space) {
    const Tensor& y_backprop = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& scale = context->input(2);
    // When is_training=True, batch mean and variance/inverted variance are
    // saved in the forward pass to be reused here. When is_training=False,
    // population mean and variance need to be forwarded here to compute the
    // gradients.
    const Tensor& saved_mean_or_pop_mean = context->input(3);
    // The Eigen implementation saves variance in the forward pass, while cuDNN
    // saves inverted variance.
    const Tensor& saved_maybe_inv_var_or_pop_var = context->input(4);

    OP_REQUIRES(context, y_backprop.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        y_backprop.shape().DebugString()));
    OP_REQUIRES(context, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(
        context, saved_mean_or_pop_mean.dims() == 1,
        errors::InvalidArgument("saved mean must be 1-dimensional",
                                saved_mean_or_pop_mean.shape().DebugString()));
    OP_REQUIRES(context, saved_maybe_inv_var_or_pop_var.dims() == 1,
                errors::InvalidArgument(
                    "saved variance must be 1-dimensional",
                    saved_maybe_inv_var_or_pop_var.shape().DebugString()));

    Tensor* x_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, x.shape(), &x_backprop));

    const TensorShape& scale_offset_shape = scale.shape();
    Tensor* scale_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, scale_offset_shape,
                                                     &scale_backprop));
    Tensor* offset_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, scale_offset_shape,
                                                     &offset_backprop));
    // Two placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    // They are filled with zeros so as to avoid NaN outputs.
    Tensor* placeholder_1 = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(3, TensorShape({}), &placeholder_1));
    functor::SetZeroFunctor<Device, float> f;
    f(context->eigen_device<Device>(), placeholder_1->flat<U>());
    Tensor* placeholder_2 = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(4, TensorShape({}), &placeholder_2));
    f(context->eigen_device<Device>(), placeholder_2->flat<U>());

    // If input is empty, set gradients w.r.t scale/offset to zero.
    if (x.shape().num_elements() == 0) {
      functor::SetZeroFunctor<Device, U> f;
      f(context->eigen_device<Device>(), scale_backprop->flat<U>());
      f(context->eigen_device<Device>(), offset_backprop->flat<U>());
      return;
    }

    const Tensor* reserve_space_data = nullptr;
    functor::CudnnBatchNormAllocatorInTemp<uint8>* workspace_allocator_ptr =
        nullptr;

#if CUDNN_VERSION >= 7402
    functor::CudnnBatchNormAllocatorInTemp<uint8> workspace_allocator(context);
    if (use_reserved_space) {
      const Tensor& reserve_space = context->input(5);
      reserve_space_data = &reserve_space;
      workspace_allocator_ptr = &workspace_allocator;
    }
#endif  // CUDNN_VERSION >= 7402

    if (is_training_) {
      functor::FusedBatchNormGrad<Device, T, U>()(
          context, y_backprop, x, scale, saved_mean_or_pop_mean,
          saved_maybe_inv_var_or_pop_var, epsilon_, x_backprop, scale_backprop,
          offset_backprop, reserve_space_data, workspace_allocator_ptr,
          tensor_format_);
    } else {
      // Necessary layout conversion is currently done in python.
      CHECK(tensor_format_ == FORMAT_NHWC)
          << "The implementation of FusedBatchNormGrad with is_training=False "
             "only support "
          << "NHWC tensor format for now.";
      Tensor scratch1, scratch2;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            scale_offset_shape, &scratch1));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            scale_offset_shape, &scratch2));
      functor::FusedBatchNormFreezeGrad<Device, T, U>()(
          context->eigen_device<Device>(), y_backprop, x, scale,
          saved_mean_or_pop_mean, saved_maybe_inv_var_or_pop_var, epsilon_,
          x_backprop, scale_backprop, offset_backprop, scratch1.vec<U>(),
          scratch2.vec<U>());
    }
  }

 private:
  U epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
};

template <typename Device, typename T, typename U>
class FusedBatchNormGradOp : public FusedBatchNormGradOpBase<Device, T, U> {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* context)
      : FusedBatchNormGradOpBase<Device, T, U>(context) {}

  void Compute(OpKernelContext* context) override {
    FusedBatchNormGradOpBase<Device, T, U>::ComputeWithReservedSpace(context,
                                                                     false);
  }
};

template <typename Device, typename T, typename U>
class FusedBatchNormGradOpV3 : public FusedBatchNormGradOpBase<Device, T, U> {
 public:
  explicit FusedBatchNormGradOpV3(OpKernelConstruction* context)
      : FusedBatchNormGradOpBase<Device, T, U>(context) {}

  void Compute(OpKernelContext* context) override {
    FusedBatchNormGradOpBase<Device, T, U>::ComputeWithReservedSpace(context,
                                                                     true);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    FusedBatchNormOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNormGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    FusedBatchNormGradOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<CPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<CPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV3")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOpV3<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV3")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOpV3<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV3")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOpV3<CPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV3")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOpV3<CPUDevice, Eigen::half, float>);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNorm").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    FusedBatchNormOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNormGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    FusedBatchNormGradOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<GPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<GPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOpV3<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("_FusedBatchNormEx")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOpEx<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOpV3<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOpV3<GPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("_FusedBatchNormEx")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOpEx<GPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV3")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOpV3<GPUDevice, Eigen::half, float>);

#endif

}  // namespace tensorflow
