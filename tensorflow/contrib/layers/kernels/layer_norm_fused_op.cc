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

#define EIGEN_USE_THREADS

#include "./layer_norm_fused_op.h"
#include "tensorflow/core/framework/op_kernel.h"

// temporarily hard coding WARP_SIZE for CUDA kernels.
#define WARP_SIZE 32
namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchLayerNormOp;

template <typename Device, typename T>
struct LaunchLayerNormBiasAddOp;

template <typename Device, typename T>
struct LaunchLayerNormFusedOp;

#if GOOGLE_CUDA
template <typename T>
struct LayerNormGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, T* output);
};
template <typename T>
struct LaunchLayerNormOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const LayerNormFusedArgs args,
                     const T* input, T* output) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    LayerNormGPULaunch<T>().Run(d, args, input, output);
  }
};

template <typename T>
struct LayerNormBiasAddGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* beta, T* output);
};
template <typename T>
struct LaunchLayerNormBiasAddOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const LayerNormFusedArgs args,
                     const T* input, const T* beta, T* output) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    LayerNormBiasAddGPULaunch<T>().Run(d, args, input, beta, output);
  }
};

template <typename T>
struct LayerNormFusedGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* gamma, const T* beta, T* output);
};
template <typename T>
struct LaunchLayerNormFusedOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const LayerNormFusedArgs args,
                     const T* input, const T* gamma, const T* beta, T* output) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    LayerNormFusedGPULaunch<T>().Run(d, args, input, gamma, beta, output);
  }
};
#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class LayerNormOp : public OpKernel {
 public:
  explicit LayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    OP_REQUIRES(
        context, input.dims() >= 2,
        errors::InvalidArgument("input dimensions must be larger than 2D",
                                input.shape().DebugString()));

    const int32 last_dim = input.dims() - 1;
    const int32 depth = input.dim_size(last_dim);

    int32 n_slices = 1;
    for (int i = 0; i < last_dim; ++i) {
      n_slices *= input.dim_size(i);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    VLOG(2) << "LayerNormCustom: "
            << "depth:" << depth << ", "
            << "n_slices:" << n_slices;

    LayerNormFusedArgs args;
    args.depth = depth;
    args.n_slices = n_slices;
    args.n_inputs = n_slices * depth;
    args.epsilon = epsilon_;

    if (depth <= WARP_SIZE) {
      int tmp_depth = depth;
      int slice_size = 1;
      while (tmp_depth >>= 1) slice_size *= 2;
      args.slice_size = slice_size >= depth ? slice_size : slice_size * 2;
    } else {
      int slice_size = (depth / WARP_SIZE) * WARP_SIZE;
      args.slice_size =
          slice_size >= depth ? slice_size : slice_size + WARP_SIZE;
    }
    auto input_ptr = input.template flat<T>().data();
    auto output_ptr = output->template flat<T>().data();

    LaunchLayerNormOp<Device, T>::launch(context, args, input_ptr, output_ptr);
  }

 private:
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(LayerNormOp);
};

template <typename Device, typename T>
class LayerNormBiasAddOp : public OpKernel {
 public:
  explicit LayerNormBiasAddOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& beta = context->input(1);

    OP_REQUIRES(
        context, input.dims() >= 2,
        errors::InvalidArgument("input dimensions must be larger than 2D",
                                input.shape().DebugString()));
    OP_REQUIRES(context, beta.dims() == 1,
                errors::InvalidArgument("beta dimension must be 1D",
                                        beta.shape().DebugString()));

    const int32 last_dim = input.dims() - 1;
    const int32 depth = input.dim_size(last_dim);

    OP_REQUIRES(context, depth == beta.dim_size(0),
                errors::InvalidArgument(
                    "input depth and beta must have the same size: ", depth,
                    " vs ", beta.dim_size(0)));

    int32 n_slices = 1;
    for (int i = 0; i < last_dim; ++i) {
      n_slices *= input.dim_size(i);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    VLOG(2) << "LayerNormBiasAddCustom: "
            << "depth:" << depth << ", "
            << "n_slices:" << n_slices;

    LayerNormFusedArgs args;
    args.depth = depth;
    args.n_slices = n_slices;
    args.n_inputs = n_slices * depth;
    args.epsilon = epsilon_;

    if (depth <= WARP_SIZE) {
      int tmp_depth = depth;
      int slice_size = 1;
      while (tmp_depth >>= 1) slice_size *= 2;
      args.slice_size = slice_size >= depth ? slice_size : slice_size * 2;
    } else {
      int slice_size = (depth / WARP_SIZE) * WARP_SIZE;
      args.slice_size =
          slice_size >= depth ? slice_size : slice_size + WARP_SIZE;
    }
    auto input_ptr = input.template flat<T>().data();
    auto beta_ptr = beta.template flat<T>().data();
    auto output_ptr = output->template flat<T>().data();

    LaunchLayerNormBiasAddOp<Device, T>::launch(context, args, input_ptr,
                                                beta_ptr, output_ptr);
  }

 private:
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(LayerNormBiasAddOp);
};

template <typename Device, typename T>
class LayerNormFusedOp : public OpKernel {
 public:
  explicit LayerNormFusedOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& gamma = context->input(1);
    const Tensor& beta = context->input(2);

    OP_REQUIRES(
        context, input.dims() >= 2,
        errors::InvalidArgument("input dimensions must be larger than 2D",
                                input.shape().DebugString()));
    OP_REQUIRES(context, gamma.dims() == 1,
                errors::InvalidArgument("gamma dimension must be 1D",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(context, beta.dims() == 1,
                errors::InvalidArgument("beta dimension must be 1D",
                                        beta.shape().DebugString()));

    const int32 last_dim = input.dims() - 1;
    const int32 depth = input.dim_size(last_dim);

    OP_REQUIRES(context, depth == gamma.dim_size(0),
                errors::InvalidArgument(
                    "input depth and gamma must have the same size: ", depth,
                    " vs ", gamma.dim_size(0)));
    OP_REQUIRES(context, depth == beta.dim_size(0),
                errors::InvalidArgument(
                    "input depth and beta must have the same size: ", depth,
                    " vs ", beta.dim_size(0)));

    int32 n_slices = 1;
    for (int i = 0; i < last_dim; ++i) {
      n_slices *= input.dim_size(i);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    VLOG(2) << "LayerNormFusedCustom: "
            << "depth:" << depth << ", "
            << "n_slices:" << n_slices;

    LayerNormFusedArgs args;
    args.depth = depth;
    args.n_slices = n_slices;
    args.n_inputs = n_slices * depth;
    args.epsilon = epsilon_;

    if (depth <= WARP_SIZE) {
      int tmp_depth = depth;
      int slice_size = 1;
      while (tmp_depth >>= 1) slice_size *= 2;
      args.slice_size = slice_size >= depth ? slice_size : slice_size * 2;
    } else {
      int slice_size = (depth / WARP_SIZE) * WARP_SIZE;
      args.slice_size =
          slice_size >= depth ? slice_size : slice_size + WARP_SIZE;
    }
    auto input_ptr = input.template flat<T>().data();
    auto gamma_ptr = gamma.template flat<T>().data();
    auto beta_ptr = beta.template flat<T>().data();
    auto output_ptr = output->template flat<T>().data();

    LaunchLayerNormFusedOp<Device, T>::launch(context, args, input_ptr,
                                              gamma_ptr, beta_ptr, output_ptr);
  }

 private:
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(LayerNormFusedOp);
};

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("LayerNormCustom").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    LayerNormOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("LayerNormCustom").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    LayerNormOp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("LayerNormBiasAddCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        LayerNormBiasAddOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("LayerNormBiasAddCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        LayerNormBiasAddOp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("LayerNormFusedCustom").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    LayerNormFusedOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("LayerNormFusedCustom").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    LayerNormFusedOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
