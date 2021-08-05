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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_fp16.h"
typedef __half2 half2;
#endif
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// TODO(ezhulenev): Use CUB reductions on GPU.
template <typename T, typename U>
struct FusedBatchNormFreezeGrad<GPUDevice, T, U> {
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
    typename TTypes<U>::Vec offset_backprop(offset_backprop_output->vec<U>());

    const int depth = pop_mean.dimension(0);
    const int rest_size = input.size() / depth;

    // Allocate two temporary workspaces of [depth] shape.
    Tensor scratch1_vec, scratch2_vec;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {depth}, &scratch1_vec));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<U>::value,
                                                   {depth}, &scratch2_vec));

    typename TTypes<U>::Vec scratch1(scratch1_vec.vec<U>());
    typename TTypes<U>::Vec scratch2(scratch2_vec.vec<U>());

    const GPUDevice& d = context->eigen_device<GPUDevice>();

    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduction_axis{0};
    Eigen::array<int, 2> rest_by_one({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0> > reduction_axis;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > rest_by_one;
    rest_by_one.set(0, rest_size);
#endif

    OP_REQUIRES(
        context, !OpDeterminismRequired(),
        errors::Unimplemented(
            "A deterministic GPU implementation of fused batch-norm backprop,"
            " when training is disabled, is not currently available."));

    // offset_backprop  = sum(y_backprop)
    // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var + epsilon))
    // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))

    auto y_backprop_rest_by_depth =
        y_backprop.reshape(rest_by_depth).template cast<U>();
    auto input_rest_by_depth = input.reshape(rest_by_depth).template cast<U>();

    offset_backprop.device(d) = y_backprop_rest_by_depth.sum(reduction_axis);

    // scratch1 = rsqrt(pop_var + epsilon)
    scratch1.device(d) = (pop_var + pop_var.constant(epsilon)).rsqrt();

    // scratch2 = sum(y_backprop * (x - mean))
    scratch2.device(d) =
        (y_backprop_rest_by_depth *
         (input_rest_by_depth -
          pop_mean.reshape(one_by_depth).broadcast(rest_by_one)))
            .sum(reduction_axis);

    x_backprop.reshape(rest_by_depth).device(d) =
        (y_backprop_rest_by_depth *
         ((scratch1 * scale).reshape(one_by_depth).broadcast(rest_by_one)))
            .template cast<T>();
    scale_backprop.device(d) = scratch2 * scratch1;
  }
};

template struct FusedBatchNormFreezeGrad<GPUDevice, float, float>;
template struct FusedBatchNormFreezeGrad<GPUDevice, Eigen::half, float>;

// -------------------------------------------------------------------------- //
// FusedBatchNormInferenceFunctor implementation.                             //
// -------------------------------------------------------------------------- //

// Generic kernel, that does all computations by converting input to U data
// type. We use it when CUDA architecture doesn't have fast arithmetic fot the
// T data type (e.g. no fp16 in old GPU generations).
template <typename T, typename U, TensorFormat tensor_format,
          bool add_side_input, FusedBatchNormActivationMode activation_mode,
          bool is_generic_kernel>
struct FusedBatchNormInferenceKernel {
  static_assert(tensor_format == FORMAT_NHWC || tensor_format == FORMAT_NCHW,
                "Unsupported data format");

  __device__ static void run(int32 count, int32 channels_size,
                             int32 inner_dim_size, const T* __restrict__ in,
                             const U* __restrict__ scale,
                             const U* __restrict__ offset,
                             const U* __restrict__ mean,
                             const U* __restrict__ var,
                             const T* __restrict__ side_input, float epsilon,
                             T* __restrict__ out) {
    int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 total_device_threads = gridDim.x * blockDim.x;

    while (index < count) {
      const int channel = (tensor_format == FORMAT_NHWC)
                              ? index % channels_size
                              : (index / inner_dim_size) % channels_size;

      U in_v = U(in[index]);
      U scale_v = scale[channel];
      U offset_v = offset[channel];
      U mean_v = mean[channel];
      U var_v = var[channel];

      U scaling_factor_v = rsqrt(var_v + epsilon) * scale_v;
      static_assert(std::is_same<U, float>::value, "U data type must be float");
      U shifted_v = fmaf(in_v - mean_v, scaling_factor_v, offset_v);

      if (add_side_input) {
        shifted_v += U(side_input[index]);
      }

      if (activation_mode == FusedBatchNormActivationMode::kIdentity) {
        out[index] = T(shifted_v);
      } else if (activation_mode == FusedBatchNormActivationMode::kRelu) {
        out[index] = T(shifted_v < U(0) ? U(0) : shifted_v);
      }

      index += total_device_threads;
    }
  }
};

// Specialization for T=Eigen::half and U=float.
template <TensorFormat tensor_format, bool add_side_input,
          FusedBatchNormActivationMode activation_mode>
struct FusedBatchNormInferenceKernel<Eigen::half, float, tensor_format,
                                     add_side_input, activation_mode,
                                     /*is_generic_kernel=*/false> {
#if TENSORFLOW_USE_ROCM
  using IT = __half;
#else
  using IT = Eigen::half;
#endif
  using T = Eigen::half;
  using U = float;

  // If CUDA architecture doesn't support fast fp16 computation, we will
  // fallback on generic kernel defined above.
  using GenericKernel =
      FusedBatchNormInferenceKernel<T, U, tensor_format, add_side_input,
                                    activation_mode,
                                    /*is_generic_kernel=*/true>;

  __device__ static void run(int32 count, int32 channels_size,
                             int32 inner_dim_size, const T* __restrict__ _in,
                             const U* __restrict__ scale,
                             const U* __restrict__ offset,
                             const U* __restrict__ mean,
                             const U* __restrict__ var,
                             const T* __restrict__ _side_input, float epsilon,
                             T* __restrict__ _out) {
    // Old GPUs do not have (or have very slow) fp16 arithmetic.
#if (__CUDA_ARCH__ >= 610) || TENSORFLOW_USE_ROCM
    const IT* in = reinterpret_cast<const IT*>(_in);
    const IT* side_input = reinterpret_cast<const IT*>(_side_input);
    IT* out = reinterpret_cast<IT*>(_out);

    int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 total_device_threads = gridDim.x * blockDim.x;

    int32 half2_count = count >> 1;

    half epsilon_h = __float2half(epsilon);
    half2 epsilon_h2 = __float2half2_rn(epsilon);

    const int32 max_channel_size = channels_size - 1;

    while (index < half2_count) {
      int32 channel[2];
      if (tensor_format == FORMAT_NHWC) {
        channel[0] = (2 * index) % channels_size;
        channel[1] = channel[0] == max_channel_size ? 0 : channel[0] + 1;
      } else {
        channel[0] = ((2 * index) / inner_dim_size) % channels_size;
        channel[1] = ((2 * index + 1) / inner_dim_size) % channels_size;
      }

      half2 in_v = reinterpret_cast<const half2*>(in)[index];
      half2 scale_v = __floats2half2_rn(scale[channel[0]], scale[channel[1]]);
      half2 offset_v =
          __floats2half2_rn(offset[channel[0]], offset[channel[1]]);
      half2 mean_v = __floats2half2_rn(mean[channel[0]], mean[channel[1]]);
      half2 var_v = __floats2half2_rn(var[channel[0]], var[channel[1]]);

      half2 scaling_factor_v =
          __hmul2(h2rsqrt(__hadd2(var_v, epsilon_h2)), scale_v);
      half2 shifted_v =
          __hfma2(__hsub2(in_v, mean_v), scaling_factor_v, offset_v);

      if (add_side_input) {
        shifted_v = __hadd2(shifted_v,
                            reinterpret_cast<const half2*>(side_input)[index]);
      }

      if (activation_mode == FusedBatchNormActivationMode::kIdentity) {
        reinterpret_cast<half2*>(out)[index] = shifted_v;

      } else if (activation_mode == FusedBatchNormActivationMode::kRelu) {
        const half2 kZeroH = __float2half2_rn(0.f);
        const half2 mask_h = __hgt2(shifted_v, kZeroH);
        reinterpret_cast<half2*>(out)[index] = __hmul2(mask_h, shifted_v);
      }

      index += total_device_threads;
    }

    if ((count & 0x1) == 1 && index == half2_count) {
      index = count - 1;

      const int32 channel = (tensor_format == FORMAT_NHWC)
                                ? index % channels_size
                                : (index / inner_dim_size) % channels_size;

      half in_v = in[index];
      half scale_v = __float2half(scale[channel]);
      half offset_v = __float2half(offset[channel]);
      half mean_v = __float2half(mean[channel]);
      half var_v = __float2half(var[channel]);

      half scaling_factor_v = __hmul(hrsqrt(__hadd(var_v, epsilon_h)), scale_v);
      half shifted_v = __hfma(__hsub(in_v, mean_v), scaling_factor_v, offset_v);

      if (add_side_input) {
        shifted_v = __hadd(shifted_v, side_input[index]);
      }

      if (activation_mode == FusedBatchNormActivationMode::kIdentity) {
        out[index] = shifted_v;

      } else if (activation_mode == FusedBatchNormActivationMode::kRelu) {
        const half kZeroH = __float2half(0.f);
        const half mask_h = __hgt(shifted_v, kZeroH);
        out[index] = __hmul(mask_h, shifted_v);
      }
    }

#else
    GenericKernel::run(count, channels_size, inner_dim_size, _in, scale, offset,
                       mean, var, _side_input, epsilon, _out);
#endif  // __CUDA_ARCH__ >= 610
  }
};

template <typename T, typename U, TensorFormat tensor_format,
          bool add_side_input, FusedBatchNormActivationMode activation_mode>
__global__ void FusedBatchNormInferenceMetaKernel(
    int32 count, int32 channels_size, int32 inner_dim_size, const T* in,
    const U* scale, const U* offset, const U* mean, const U* var,
    const T* side_input, float epsilon, T* out) {
  // We prefer to run non-generic specialization, for the given types T and U.
  FusedBatchNormInferenceKernel<T, U, tensor_format, add_side_input,
                                activation_mode,
#if TENSORFLOW_USE_ROCM
                                false
#else
                                // TODO(b/135435976): Temporary disable
                                // non-generic kernel implementation.
                                /*is_generic_kernel=*/true
#endif
                                >::run(count, channels_size, inner_dim_size, in,
                                       scale, offset, mean, var, side_input,
                                       epsilon, out);
}

template <typename T, typename U>
struct FusedBatchNormInferenceFunctor<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, TensorFormat tensor_format,
                  typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<U>::ConstVec scale,
                  typename TTypes<U>::ConstVec offset,
                  typename TTypes<U>::ConstVec estimated_mean,
                  typename TTypes<U>::ConstVec estimated_variance,
                  typename TTypes<T, 4>::ConstTensor side_input, U epsilon,
                  FusedBatchNormActivationMode activation_mode,
                  typename TTypes<T, 4>::Tensor out) {
    const auto& d = context->eigen_device<GPUDevice>();

    const int32 count = out.size();
    if (count == 0) return;

    bool launched = false;
#if TENSORFLOW_USE_ROCM
    constexpr int32 kThreadInBlock = 1024;
#else
    constexpr int32 kThreadInBlock = 512;
#endif

#define LAUNCH(DATA_FORMAT, ADD_SIDE_INPUT, ACTIVATION, CHANNEL_SIZE,          \
               INNER_DIM_SIZE)                                                 \
  launched = true;                                                             \
                                                                               \
  GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(                   \
      std::is_same<T, Eigen::half>::value ? Eigen::divup(count, 2) : count, d, \
      FusedBatchNormInferenceMetaKernel<T, U, DATA_FORMAT, ADD_SIDE_INPUT,     \
                                        ACTIVATION>,                           \
      0, kThreadInBlock);                                                      \
                                                                               \
  TF_CHECK_OK(GpuLaunchKernel(                                                 \
      FusedBatchNormInferenceMetaKernel<T, U, DATA_FORMAT, ADD_SIDE_INPUT,     \
                                        ACTIVATION>,                           \
      config.block_count, config.thread_per_block, 0, d.stream(), count,       \
      CHANNEL_SIZE, INNER_DIM_SIZE, in.data(), scale.data(), offset.data(),    \
      estimated_mean.data(), estimated_variance.data(), side_input.data(),     \
      epsilon, out.data()));

    const bool no_side_input = side_input.dimensions().TotalSize() == 0;
    const bool add_side_input = side_input.dimensions().TotalSize() != 0;

    using Activation = FusedBatchNormActivationMode;
    const bool no_activation = activation_mode == Activation::kIdentity;
    const bool relu_activation = activation_mode == Activation::kRelu;

    if (tensor_format == FORMAT_NHWC) {
      const int c = in.dimensions()[3];

      if (no_activation && no_side_input) {
        LAUNCH(FORMAT_NHWC, false, Activation::kIdentity, c, 1);
      } else if (relu_activation && no_side_input) {
        LAUNCH(FORMAT_NHWC, false, Activation::kRelu, c, 1);
      } else if (no_activation && add_side_input) {
        LAUNCH(FORMAT_NHWC, true, Activation::kIdentity, c, 1);
      } else if (relu_activation && add_side_input) {
        LAUNCH(FORMAT_NHWC, true, Activation::kRelu, c, 1);
      }

    } else if (tensor_format == FORMAT_NCHW) {
      const int c = in.dimensions()[1];
      const int inner = in.dimensions()[2] * in.dimensions()[3];

      if (no_activation && no_side_input) {
        LAUNCH(FORMAT_NCHW, false, Activation::kIdentity, c, inner);
      } else if (relu_activation && no_side_input) {
        LAUNCH(FORMAT_NCHW, false, Activation::kRelu, c, inner);
      } else if (no_activation && add_side_input) {
        LAUNCH(FORMAT_NCHW, true, Activation::kIdentity, c, inner);
      } else if (relu_activation && add_side_input) {
        LAUNCH(FORMAT_NCHW, true, Activation::kRelu, c, inner);
      }
    }
#undef LAUNCH

    OP_REQUIRES(context, launched,
                errors::InvalidArgument("Unsupported launch configuration"));
  }
};

template struct FusedBatchNormInferenceFunctor<GPUDevice, float, float>;
template struct FusedBatchNormInferenceFunctor<GPUDevice, Eigen::half, float>;

}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/kernels/fused_batch_norm_op.h"

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
