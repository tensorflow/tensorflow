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

#include <atomic>

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/kernels/redux_functor.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using se::DeviceMemory;
using se::ScratchAllocator;
using se::Stream;
using se::port::StatusOr;
#endif

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

template <typename T, typename U>
struct FusedBatchNorm<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  const Tensor& estimated_mean_input,
                  const Tensor& estimated_variance_input,
                  const Tensor* side_input, U epsilon,
                  FusedBatchNormActivationMode activation_mode,
                  Tensor* y_output, Tensor* batch_mean_output,
                  Tensor* batch_var_output, Tensor* saved_mean_output,
                  Tensor* saved_var_output, TensorFormat tensor_format,
                  bool use_reserved_space, bool is_training) {
    OP_REQUIRES(context, side_input == nullptr,
                errors::Internal(
                    "The CPU implementation of FusedBatchNorm does not support "
                    "side input."));
    OP_REQUIRES(context,
                activation_mode == FusedBatchNormActivationMode::kIdentity,
                errors::Internal("The CPU implementation of FusedBatchNorm "
                                 "does not support activations."));

    if (use_reserved_space) {
      Tensor* dummy_reserve_space = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(5, {}, &dummy_reserve_space));
      // Initialize the memory, to avoid sanitizer alerts.
      dummy_reserve_space->flat<U>()(0) = U();
    }
    Tensor transformed_x;
    Tensor transformed_y;
    if (tensor_format == FORMAT_NCHW) {
      const int64 in_batch = GetTensorDim(x_input, tensor_format, 'N');
      const int64 in_rows = GetTensorDim(x_input, tensor_format, 'H');
      const int64 in_cols = GetTensorDim(x_input, tensor_format, 'W');
      const int64 in_depths = GetTensorDim(x_input, tensor_format, 'C');
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NHWC, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_x));
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NHWC, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_y));
      // Perform NCHW to NHWC
      std::vector<int32> perm = {0, 2, 3, 1};
      OP_REQUIRES_OK(
          context, ::tensorflow::DoTranspose(context->eigen_device<CPUDevice>(),
                                             x_input, perm, &transformed_x));
    } else {
      transformed_x = x_input;
      transformed_y = *y_output;
    }
    typename TTypes<T, 4>::Tensor x(transformed_x.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec offset(offset_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_mean(estimated_mean_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_variance(
        estimated_variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor y(transformed_y.tensor<T, 4>());
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
    BlockingCounter barrier(1);
    std::atomic<uint8> task_counter;
    auto on_done = [&]() {
      uint8 count = --task_counter;
      if (count == 0) {
        if (tensor_format == FORMAT_NCHW) {
          // Perform NHWC to NCHW
          const std::vector<int32> perm = {0, 3, 1, 2};
          const Status s =
              ::tensorflow::DoTranspose(context->eigen_device<CPUDevice>(),
                                        transformed_y, perm, y_output);
          if (!s.ok()) {
            context->SetStatus(
                errors::InvalidArgument("Transpose failed: ", s));
          }
        }
        barrier.DecrementCount();
      }
    };
    if (is_training) {
      mean.device(d) = (x_rest_by_depth.sum(reduce_dims) * rest_size_inv);
      auto x_centered =
          x_rest_by_depth - mean.reshape(one_by_depth).broadcast(bcast_spec);

      variance.device(d) = x_centered.square().sum(reduce_dims) * rest_size_inv;
      auto scaling_factor = ((variance + epsilon).rsqrt() * scale)
                                .eval()
                                .reshape(one_by_depth)
                                .broadcast(bcast_spec);
      auto x_scaled = x_centered * scaling_factor;
      auto x_shifted =
          (x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec))
              .template cast<T>();

      task_counter = 5;
      y.reshape(rest_by_depth).device(d, on_done) = x_shifted;
      batch_var.device(d, on_done) = variance * rest_size_adjust;
      saved_var.device(d, on_done) = variance;
      batch_mean.device(d, on_done) = mean;
      saved_mean.device(d, on_done) = mean;
    } else {  // is_training == false
      auto x_centered =
          x_rest_by_depth -
          estimated_mean.reshape(one_by_depth).broadcast(bcast_spec);
      auto scaling_factor = ((estimated_variance + epsilon).rsqrt() * scale)
                                .eval()
                                .reshape(one_by_depth)
                                .broadcast(bcast_spec);
      auto x_scaled = x_centered * scaling_factor;
      auto x_shifted =
          (x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec))
              .template cast<T>();

      task_counter = 3;
      y.reshape(rest_by_depth).device(d, on_done) = x_shifted;
      mean.device(d, on_done) = estimated_mean;
      variance.device(d, on_done) = estimated_variance;
    }
    barrier.Wait();
  }
};

template <typename T, typename U>
struct FusedBatchNormGrad<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& mean_input, const Tensor& variance_input,
                  U epsilon, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output, Tensor* offset_backprop_output,
                  bool use_reserved_space, TensorFormat tensor_format) {
    Tensor transformed_y_backprop_input;
    Tensor transformed_x_input;
    Tensor transformed_x_backprop_output;
    if (tensor_format == FORMAT_NCHW) {
      const int64 in_batch = GetTensorDim(x_input, tensor_format, 'N');
      const int64 in_rows = GetTensorDim(x_input, tensor_format, 'H');
      const int64 in_cols = GetTensorDim(x_input, tensor_format, 'W');
      const int64 in_depths = GetTensorDim(x_input, tensor_format, 'C');
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NHWC, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_y_backprop_input));
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NHWC, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_x_input));
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NHWC, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_x_backprop_output));
      // Perform NCHW to NHWC
      std::vector<int32> perm = {0, 2, 3, 1};
      OP_REQUIRES_OK(
          context, ::tensorflow::DoTranspose(context->eigen_device<CPUDevice>(),
                                             y_backprop_input, perm,
                                             &transformed_y_backprop_input));
      OP_REQUIRES_OK(context, ::tensorflow::DoTranspose(
                                  context->eigen_device<CPUDevice>(), x_input,
                                  perm, &transformed_x_input));
    } else {
      transformed_y_backprop_input = y_backprop_input;
      transformed_x_input = x_input;
      transformed_x_backprop_output = *x_backprop_output;
    }
    typename TTypes<T, 4>::Tensor y_backprop(
        transformed_y_backprop_input.tensor<T, 4>());
    typename TTypes<T, 4>::Tensor x(transformed_x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec mean(mean_input.vec<U>());
    typename TTypes<U>::ConstVec variance(variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor x_backprop(
        transformed_x_backprop_output.tensor<T, 4>());
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
    Eigen::array<int, 2> bcast_spec({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> bcast_spec;
    bcast_spec.set(0, rest_size);
#endif

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
    U rest_size_inv = static_cast<U>(1.0f / static_cast<U>(rest_size));

    // Eigen is notoriously bad at reducing outer dimension, so we materialize
    // all temporary tensors that require reduction, and then use Eigen redux
    // functor, that is optimized for this particular task.
    //
    // All reductions are of this type: [rest_size, depth] -> [depth].
    using ScalarSum = Eigen::internal::scalar_sum_op<U>;
    const functor::ReduceOuterDimensions<T, U, U, ScalarSum> redux_sum_t;
    const functor::ReduceOuterDimensions<U, U, U, ScalarSum> redux_sum_u;

    auto scratch_dtype = DataTypeToEnum<U>::value;

    // Allocate a temporary workspace of [depth] shape.
    Tensor scratch_one_by_depth;
    OP_REQUIRES_OK(context, context->allocate_temp(scratch_dtype, {depth},
                                                   &scratch_one_by_depth));

    // Maybe allocate a temporary workspace of [rest_size, depth] shape.
    Tensor scratch_rest_by_depth;
    if (std::is_same<T, U>::value) {
      OP_REQUIRES(context,
                  scratch_rest_by_depth.CopyFrom(transformed_x_backprop_output,
                                                 {rest_size, depth}),
                  errors::Internal("Failed to copy a tensor"));
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_temp(scratch_dtype, {rest_size, depth},
                                            &scratch_rest_by_depth));
    }

    typename TTypes<U, 2>::Tensor scratch_tensor(
        scratch_rest_by_depth.tensor<U, 2>());
    typename TTypes<U>::Vec scratch_vector(scratch_one_by_depth.vec<U>());

    auto x_mean_rest_by_depth =
        mean.reshape(one_by_depth).broadcast(bcast_spec);
    auto x_centered = (x_rest_by_depth - x_mean_rest_by_depth);
    auto coef0_one_by_depth =
        (variance.reshape(one_by_depth) + epsilon).rsqrt();
    auto coef0_rest_by_depth = coef0_one_by_depth.broadcast(bcast_spec);
    auto x_scaled = x_centered * coef0_rest_by_depth;

    auto y_backprop_rest_by_depth =
        y_backprop.reshape(rest_by_depth).template cast<U>();

    // Compute `scale_backprop_output`:
    //   scale_backprop =
    //     (y_backprop_rest_by_depth * x_scaled).sum(reduce_dims)
    scratch_tensor.device(d) = y_backprop_rest_by_depth * x_scaled;
    redux_sum_u(d, rest_by_depth, scratch_rest_by_depth, scale_backprop_output);

    // Compute 'offset_backprop_output':
    //   offset_backprop =
    //     y_backprop_rest_by_depth.sum(reduce_dims)
    redux_sum_t(d, rest_by_depth, transformed_y_backprop_input,
                offset_backprop_output);
    auto y_backprop_sum = offset_backprop;

    auto y_backprop_sum_one_by_depth = y_backprop_sum.reshape(one_by_depth);
    auto y_backprop_mean_one_by_depth =
        y_backprop_sum_one_by_depth * rest_size_inv;
    auto y_backprop_mean_rest_by_depth =
        y_backprop_mean_one_by_depth.broadcast(bcast_spec);
    auto y_backprop_centered =
        y_backprop_rest_by_depth - y_backprop_mean_rest_by_depth;

    // Compute expression:
    //   y_backprop_centered_mean =
    //     (y_backprop_rest_by_depth * x_centered).mean(reduce_dims)
    scratch_tensor.device(d) = y_backprop_rest_by_depth * x_centered;
    redux_sum_u(d, rest_by_depth, scratch_rest_by_depth, &scratch_one_by_depth);
    auto y_backprop_centered_mean =
        scratch_vector.reshape(one_by_depth) / static_cast<U>(rest_size);

    auto coef1 = (scale.reshape(one_by_depth) * coef0_one_by_depth)
                     .broadcast(bcast_spec);
    auto coef2 = (coef0_one_by_depth.square() * y_backprop_centered_mean)
                     .broadcast(bcast_spec);

    x_backprop.reshape(rest_by_depth).device(d) =
        (coef1 * (y_backprop_centered - x_centered * coef2)).template cast<T>();

    if (tensor_format == FORMAT_NCHW) {
      // Perform NHWC to NCHW
      std::vector<int32> perm = {0, 3, 1, 2};
      OP_REQUIRES_OK(
          context, ::tensorflow::DoTranspose(context->eigen_device<CPUDevice>(),
                                             transformed_x_backprop_output,
                                             perm, x_backprop_output));
    }
  }
};

template <typename T, typename U>
struct FusedBatchNormFreezeGrad<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& pop_mean_input,
                  const Tensor& pop_variance_input, U epsilon,
                  Tensor* x_backprop_output, Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {
    typename TTypes<T, 4>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 4>());
    typename TTypes<T, 4>::ConstTensor input(x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec pop_mean(pop_mean_input.vec<U>());
    typename TTypes<U>::ConstVec pop_var(pop_variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor x_backprop(x_backprop_output->tensor<T, 4>());
    typename TTypes<U>::Vec scale_backprop(scale_backprop_output->vec<U>());

    const int depth = pop_mean.dimension(0);
    const int rest_size = input.size() / depth;

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    // Allocate two temporary workspaces of [depth] shape.
    Tensor scratch1_vec, scratch2_vec;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {depth}, &scratch1_vec));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {depth}, &scratch2_vec));

    // Maybe allocate a temporary workspace of [rest_size, depth] shape.
    Tensor scratch3_tensor;
    if (std::is_same<T, U>::value) {
      OP_REQUIRES(
          context,
          scratch3_tensor.CopyFrom(*x_backprop_output, {rest_size, depth}),
          errors::Internal("Failed to copy a tensor"));
    } else {
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                     {rest_size, depth},
                                                     &scratch3_tensor));
    }

    typename TTypes<U>::Vec scratch1(scratch1_vec.vec<U>());
    typename TTypes<U>::Vec scratch2(scratch2_vec.vec<U>());
    typename TTypes<U, 2>::Tensor scratch3(scratch3_tensor.tensor<U, 2>());

    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 2> rest_by_one({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> rest_by_one;
    rest_by_one.set(0, rest_size);
#endif

    // Sum reduction along the 0th dimension using custom CPU functor.
    using ScalarSum = Eigen::internal::scalar_sum_op<U>;
    const functor::ReduceOuterDimensions<T, U, U, ScalarSum> redux_sum_t;
    const functor::ReduceOuterDimensions<U, U, U, ScalarSum> redux_sum_u;

    // offset_backprop  = sum(y_backprop)
    // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var + epsilon))
    // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))

    // NOTE: DEFAULT DEVICE comment is added to expression assignments that
    // we don't want to be executed in a thread pool.

    auto y_backprop_rest_by_depth =
        y_backprop.reshape(rest_by_depth).template cast<U>();
    auto input_rest_by_depth = input.reshape(rest_by_depth).template cast<U>();

    // offset_backprop  = sum(y_backprop)
    redux_sum_t(d, rest_by_depth, y_backprop_input, offset_backprop_output);

    // scratch1 = rsqrt(pop_var + epsilon)
    scratch1 = (pop_var + pop_var.constant(epsilon)).rsqrt();  // DEFAULT DEVICE

    // scratch2 = sum(y_backprop * (x - mean))
    scratch3.device(d) =
        y_backprop_rest_by_depth *
        (input_rest_by_depth -
         pop_mean.reshape(one_by_depth).broadcast(rest_by_one));
    redux_sum_u(d, rest_by_depth, scratch3_tensor, &scratch2_vec);

    x_backprop.reshape(rest_by_depth).device(d) =
        (y_backprop_rest_by_depth *
         ((scratch1.reshape(one_by_depth) * scale.reshape(one_by_depth))
              .broadcast(rest_by_one)))
            .template cast<T>();
    scale_backprop = scratch2 * scratch1;  // DEFAULT DEVICE
  }
};

#if !GOOGLE_CUDA
namespace {
// See implementation under GOOGLE_CUDA #ifdef below.
// This is a CUDA specific feature, do not enable it for non-CUDA builds
bool BatchnormSpatialPersistentEnabled() { return false; }
}  // namespace
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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

#if GOOGLE_CUDA
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
#endif

}  // namespace

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

  int64 GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64 byte_size) override {
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

  int64 GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64 byte_size) override {
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

template <typename T, typename U>
struct FusedBatchNorm<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x,
                  const Tensor& scale, const Tensor& offset,
                  const Tensor& estimated_mean,
                  const Tensor& estimated_variance, const Tensor* side_input,
                  U epsilon, FusedBatchNormActivationMode activation_mode,
                  Tensor* y, Tensor* batch_mean, Tensor* batch_var,
                  Tensor* saved_mean, Tensor* saved_inv_var,
                  TensorFormat tensor_format, bool use_reserved_space,
                  bool is_training) {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available"));

    const int64 batch_size = GetTensorDim(x, tensor_format, 'N');
    const int64 channels = GetTensorDim(x, tensor_format, 'C');
    const int64 height = GetTensorDim(x, tensor_format, 'H');
    const int64 width = GetTensorDim(x, tensor_format, 'W');

    // If use_reserved_space we have reserve_space_3 output (only in
    // FusedBatchNormV3 op).

#if GOOGLE_CUDA
    // Check if cuDNN batch normalization has a fast NHWC implementation:
    //   (1) In inference mode it's always fast.
    //   (2) Tensorflow enabled batchnorm spatial persistence, we are called
    //   from
    //       FusedBatchNormV3, i.e. use_reserved_space is true.
    const bool fast_nhwc_batch_norm =
        !is_training ||
        (BatchnormSpatialPersistentEnabled() &&
         DataTypeToEnum<T>::value == DT_HALF && use_reserved_space);
#else
    // fast NHWC implementation is a CUDA only feature
    const bool fast_nhwc_batch_norm = false;
#endif

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
            << " activation mode: " << ToString(activation_mode)
            << " tensor format: " << ToString(tensor_format)
            << " compute format: " << ToString(compute_format);

    auto maybe_make_dummy_output = [context, use_reserved_space]() -> Status {
      if (use_reserved_space) {
        Tensor* dummy_reserve_space = nullptr;
        return context->allocate_output(5, {}, &dummy_reserve_space);
      }
      return Status::OK();
    };

    // If input is empty, return NaN mean/variance
    if (x.shape().num_elements() == 0) {
      OP_REQUIRES_OK(context, maybe_make_dummy_output());
      functor::SetNanFunctor<U> f;
      f(context->eigen_device<GPUDevice>(), batch_mean->flat<U>());
      f(context->eigen_device<GPUDevice>(), batch_var->flat<U>());
      return;
    }

    // In inference mode we use custom CUDA kernel, because cuDNN does not
    // support side input and activations for inference.
    const bool has_side_input = side_input != nullptr;
    const bool has_activation =
        activation_mode != FusedBatchNormActivationMode::kIdentity;

    if (!is_training && (has_side_input || has_activation)) {
      OP_REQUIRES_OK(context, maybe_make_dummy_output());
      FusedBatchNormInferenceFunctor<GPUDevice, T, U> inference_functor;

      if (has_side_input) {
        inference_functor(context, tensor_format, x.tensor<T, 4>(),
                          scale.vec<U>(), offset.vec<U>(),
                          estimated_mean.vec<U>(), estimated_variance.vec<U>(),
                          side_input->tensor<T, 4>(), epsilon, activation_mode,
                          y->tensor<T, 4>());
      } else {
        typename TTypes<T, 4>::ConstTensor empty_tensor(nullptr, 0, 0, 0, 0);
        inference_functor(context, tensor_format, x.tensor<T, 4>(),
                          scale.vec<U>(), offset.vec<U>(),
                          estimated_mean.vec<U>(), estimated_variance.vec<U>(),
                          empty_tensor, epsilon, activation_mode,
                          y->tensor<T, 4>());
      }
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
    auto side_input_ptr =
        side_input != nullptr
            ? StreamExecutorUtil::AsDeviceMemory<U>(*side_input)
            : se::DeviceMemory<U>();
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

    std::unique_ptr<functor::CudnnBatchNormAllocatorInOutput<U>>
        reserve_space_allocator;
    std::unique_ptr<functor::CudnnBatchNormAllocatorInTemp<uint8>>
        workspace_allocator;
    if (use_reserved_space) {
      reserve_space_allocator.reset(
          new functor::CudnnBatchNormAllocatorInOutput<U>(context, 5));
      workspace_allocator.reset(
          new functor::CudnnBatchNormAllocatorInTemp<uint8>(context));
    }
    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationForward(
                x_ptr, scale_ptr, offset_ptr, estimated_mean_ptr,
                estimated_variance_ptr, side_input_ptr, x_desc,
                scale_offset_desc, static_cast<double>(epsilon),
                AsDnnActivationMode(activation_mode), &y_ptr, &batch_mean_ptr,
                &batch_var_ptr, &saved_mean_ptr, &saved_inv_var_ptr,
                is_training, std::move(var_to_inv_var),
                std::move(inv_var_to_var), reserve_space_allocator.get(),
                workspace_allocator.get())
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(
          errors::Internal("cuDNN launch failure : input shape (",
                           x.shape().DebugString(), ")"));
      return;
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
                  bool use_reserved_space, TensorFormat tensor_format) {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available"));

    const int64 batch_size = GetTensorDim(x, tensor_format, 'N');
    const int64 channels = GetTensorDim(x, tensor_format, 'C');
    const int64 height = GetTensorDim(x, tensor_format, 'H');
    const int64 width = GetTensorDim(x, tensor_format, 'W');

#if GOOGLE_CUDA
    // Check if cuDNN batch normalization has a fast NHWC implementation:
    //   (1) Tensorflow enabled batchnorm spatial persistence, and
    //       FusedBatchNormGradV3 passed non-null reserve space and allocator.
    const bool fast_nhwc_batch_norm = BatchnormSpatialPersistentEnabled() &&
                                      DataTypeToEnum<T>::value == DT_HALF &&
                                      use_reserved_space;
#else
    // fast NHWC implementation is a CUDA only feature
    const bool fast_nhwc_batch_norm = false;
#endif

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

    std::unique_ptr<functor::CudnnBatchNormAllocatorInTemp<uint8>>
        workspace_allocator;
    DeviceMemory<uint8>* reserve_space_data_ptr = nullptr;
    DeviceMemory<uint8> reserve_space_data;
#if CUDNN_VERSION >= 7402
    if (use_reserved_space) {
      const Tensor& reserve_space = context->input(5);
      workspace_allocator.reset(
          new functor::CudnnBatchNormAllocatorInTemp<uint8>(context));

      // the cudnn kernel outputs inverse variance in forward and reuse it in
      // backward
      if (reserve_space.dims() != 0) {
        reserve_space_data = functor::CastDeviceMemory<uint8, U>(
            const_cast<Tensor*>(&reserve_space));
        reserve_space_data_ptr = &reserve_space_data;
      }
    }
#endif  // CUDNN_VERSION >= 7402

    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationBackward(
                y_backprop_ptr, x_ptr, scale_ptr, mean_ptr, inv_variance_ptr,
                x_desc, scale_offset_desc, static_cast<double>(epsilon),
                &x_backprop_ptr, &scale_backprop_ptr, &offset_backprop_ptr,
                reserve_space_data_ptr, workspace_allocator.get())
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
#define DECLARE_GPU_SPEC(T, U)                                                 \
  template <>                                                                  \
  void FusedBatchNormFreezeGrad<GPUDevice, T, U>::operator()(                  \
      OpKernelContext* context, const Tensor& y_backprop_input,                \
      const Tensor& x_input, const Tensor& scale_input,                        \
      const Tensor& mean_input, const Tensor& variance_input, U epsilon,       \
      Tensor* x_backprop_output, Tensor* scale_backprop_output,                \
      Tensor* offset_backprop_output);                                         \
  extern template struct FusedBatchNormFreezeGrad<GPUDevice, T, U>;            \
  template <>                                                                  \
  void FusedBatchNormInferenceFunctor<GPUDevice, T, U>::operator()(            \
      OpKernelContext* context, TensorFormat tensor_format,                    \
      typename TTypes<T, 4>::ConstTensor in,                                   \
      typename TTypes<U>::ConstVec scale, typename TTypes<U>::ConstVec offset, \
      typename TTypes<U>::ConstVec estimated_mean,                             \
      typename TTypes<U>::ConstVec estimated_variance,                         \
      typename TTypes<T, 4>::ConstTensor side_input, U epsilon,                \
      FusedBatchNormActivationMode activation_mode,                            \
      typename TTypes<T, 4>::Tensor out);                                      \
  extern template struct FusedBatchNormInferenceFunctor<GPUDevice, T, U>;

DECLARE_GPU_SPEC(float, float);
DECLARE_GPU_SPEC(Eigen::half, float);

#undef DECLARE_GPU_SPEC

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace functor

template <typename Device, typename T, typename U>
class FusedBatchNormOpBase : public OpKernel {
  using FbnActivationMode = functor::FusedBatchNormActivationMode;

 protected:
  explicit FusedBatchNormOpBase(OpKernelConstruction* context,
                                bool is_batch_norm_ex = false)
      : OpKernel(context) {
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
      if (has_side_input_ && is_training_) {
        OP_REQUIRES(
            context, activation_mode_ != FbnActivationMode::kIdentity,
            errors::InvalidArgument("Identity activation is not supported with "
                                    "non-empty side input"));
      }
    }

    if (activation_mode_ != FbnActivationMode::kIdentity && is_training_) {
      // NOTE(ezhulenev): Following requirements are coming from implementation
      // details of cudnnBatchNormalizationForwardTrainingEx used in training
      // mode. In inference mode we call custom CUDA kernel that supports all
      // data formats and data types.
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
    const Tensor* side_input = has_side_input_ ? &context->input(5) : nullptr;

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
      OP_REQUIRES(context, side_input->shape() == x.shape(),
                  errors::InvalidArgument(
                      "side_input shape must be equal to input shape: ",
                      side_input->shape().DebugString(),
                      " != ", x.shape().DebugString()));
    }

    if (activation_mode_ != FbnActivationMode::kIdentity) {
      // NOTE(ezhulenev): This requirement is coming from implementation
      // details of cudnnBatchNormalizationForwardTrainingEx.
      OP_REQUIRES(
          context, !is_training_ || x.dim_size(3) % 4 == 0,
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

    functor::FusedBatchNorm<Device, T, U>()(
        context, x, scale, offset, estimated_mean, estimated_variance,
        side_input, epsilon_, activation_mode_, y, batch_mean, batch_var,
        saved_mean, saved_maybe_inv_var, tensor_format_, use_reserved_space,
        is_training_);
  }

 private:
  U epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
  bool has_side_input_;
  FbnActivationMode activation_mode_;
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

    if (is_training_) {
      functor::FusedBatchNormGrad<Device, T, U>()(
          context, y_backprop, x, scale, saved_mean_or_pop_mean,
          saved_maybe_inv_var_or_pop_var, epsilon_, x_backprop, scale_backprop,
          offset_backprop, use_reserved_space, tensor_format_);
    } else {
      // Necessary layout conversion is currently done in python.
      CHECK(tensor_format_ == FORMAT_NHWC)
          << "The implementation of FusedBatchNormGrad with is_training=False "
             "only support "
          << "NHWC tensor format for now.";
      functor::FusedBatchNormFreezeGrad<Device, T, U>()(
          context, y_backprop, x, scale, saved_mean_or_pop_mean,
          saved_maybe_inv_var_or_pop_var, epsilon_, x_backprop, scale_backprop,
          offset_backprop);
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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
