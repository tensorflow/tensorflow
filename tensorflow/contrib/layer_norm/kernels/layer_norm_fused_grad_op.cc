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

#include "tensorflow/contrib/layer_norm/kernels/layer_norm_fused_op.h"
#include "tensorflow/core/framework/op_kernel.h"

// temporarily hard coding warp_size for CUDA kernels.
#define WARP_SIZE 32
namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchLayerNormBackpropOp;

template <typename Device, typename T>
struct LaunchLayerNormBiasAddBackpropOp;

template <typename Device, typename T>
struct LaunchLayerNormFusedBackpropOp;

#if GOOGLE_CUDA
template <typename T>
struct LayerNormBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* out_back, T* in_back);
};
template <typename T>
struct LaunchLayerNormBackpropOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const LayerNormFusedArgs args,
                     const T* input, const T* out_back, T* in_back) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    LayerNormBackpropGPULaunch<T>().Run(d, args, input, out_back, in_back);
  }
};

template <typename T>
struct LayerNormBiasAddBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* out_back, T* in_back, T* beta_back);
};
template <typename T>
struct LaunchLayerNormBiasAddBackpropOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const LayerNormFusedArgs args,
                     const T* input, const T* out_back, T* in_back,
                     T* beta_back) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    LayerNormBiasAddBackpropGPULaunch<T>().Run(d, args, input, out_back,
                                               in_back, beta_back);
  }
};

template <typename T>
struct LayerNormFusedBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* out_back, const T* gamma, T* in_back,
                  T* gamma_back, T* beta_back);
};
template <typename T>
struct LaunchLayerNormFusedBackpropOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const LayerNormFusedArgs args,
                     const T* input, const T* out_back, const T* gamma,
                     T* in_back, T* gamma_back, T* beta_back) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    LayerNormFusedBackpropGPULaunch<T>().Run(d, args, input, out_back, gamma,
                                             in_back, gamma_back, beta_back);
  }
};
#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class LayerNormBackpropOp : public OpKernel {
 public:
  explicit LayerNormBackpropOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& out_back = context->input(1);

    OP_REQUIRES(
        context, input.dims() >= 2,
        errors::InvalidArgument("input dimensions must be larger than 2D",
                                input.shape().DebugString()));

    const int32 last_dim = input.dims() - 1;
    const int32 depth = input.dim_size(last_dim);

    int32 n_slices = input.NumElements() / depth;

    Tensor* in_back = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &in_back));

    VLOG(2) << "LayerNormBackpropCustom: "
            << "depth:" << depth << ", "
            << "n_slices:" << n_slices;

    LayerNormFusedArgs args;
    args.depth = depth;
    args.n_slices = n_slices;
    args.n_inputs = n_slices * depth;
    args.epsilon = epsilon_;

    if (depth <= WARP_SIZE) {
      int slice_size = 1;
      while (slice_size < depth) {
        slice_size <<= 1;
      }
      args.slice_size = slice_size >= depth ? slice_size : slice_size * 2;
    } else {
      int slice_size = slice_size =
          (depth + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
      args.slice_size = slice_size;
    }

    auto input_ptr = input.template flat<T>().data();
    auto out_back_ptr = out_back.template flat<T>().data();

    auto in_back_ptr = in_back->template flat<T>().data();

    LaunchLayerNormBackpropOp<Device, T>::launch(context, args, input_ptr,
                                                 out_back_ptr, in_back_ptr);
  }

 private:
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(LayerNormBackpropOp);
};

template <typename Device, typename T>
class LayerNormBiasAddBackpropOp : public OpKernel {
 public:
  explicit LayerNormBiasAddBackpropOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& out_back = context->input(1);
    const Tensor& beta = context->input(2);

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

    Tensor* in_back = nullptr;
    Tensor* beta_back = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &in_back));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, beta.shape(), &beta_back));

    VLOG(2) << "LayerNormBiasAddBackpropCustom: "
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
    auto out_back_ptr = out_back.template flat<T>().data();

    auto in_back_ptr = in_back->template flat<T>().data();
    auto beta_back_ptr = beta_back->template flat<T>().data();

    LaunchLayerNormBiasAddBackpropOp<Device, T>::launch(
        context, args, input_ptr, out_back_ptr, in_back_ptr, beta_back_ptr);
  }

 private:
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(LayerNormBiasAddBackpropOp);
};

template <typename Device, typename T>
class LayerNormFusedBackpropOp : public OpKernel {
 public:
  explicit LayerNormFusedBackpropOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& out_back = context->input(1);
    const Tensor& gamma = context->input(2);

    OP_REQUIRES(
        context, input.dims() >= 2,
        errors::InvalidArgument("input dimensions must be larger than 2D",
                                input.shape().DebugString()));
    OP_REQUIRES(context, gamma.dims() == 1,
                errors::InvalidArgument("gamma dimension must be 1D",
                                        gamma.shape().DebugString()));
    const int32 last_dim = input.dims() - 1;
    const int32 depth = input.dim_size(last_dim);

    OP_REQUIRES(context, depth == gamma.dim_size(0),
                errors::InvalidArgument(
                    "input depth and gamma must have the same size: ", depth,
                    " vs ", gamma.dim_size(0)));

    int32 n_slices = 1;
    for (int i = 0; i < last_dim; ++i) {
      n_slices *= input.dim_size(i);
    }

    Tensor* in_back = nullptr;
    Tensor* gamma_back = nullptr;
    Tensor* beta_back = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &in_back));
    // gamma_back and beta_back's shape will be the same as gamma;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, gamma.shape(), &gamma_back));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, gamma.shape(), &beta_back));

    VLOG(2) << "LayerNormFusedBackpropCustom: "
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
    auto out_back_ptr = out_back.template flat<T>().data();
    auto gamma_ptr = gamma.template flat<T>().data();

    auto in_back_ptr = in_back->template flat<T>().data();
    auto gamma_back_ptr = gamma_back->template flat<T>().data();
    auto beta_back_ptr = beta_back->template flat<T>().data();

    LaunchLayerNormFusedBackpropOp<Device, T>::launch(
        context, args, input_ptr, out_back_ptr, gamma_ptr, in_back_ptr,
        gamma_back_ptr, beta_back_ptr);
  }

 private:
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(LayerNormFusedBackpropOp);
};

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LayerNormBackpropCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        LayerNormBackpropOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("LayerNormBackpropCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        LayerNormBackpropOp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("LayerNormBiasAddBackpropCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        LayerNormBiasAddBackpropOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("LayerNormBiasAddBackpropCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        LayerNormBiasAddBackpropOp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("LayerNormFusedBackpropCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        LayerNormFusedBackpropOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("LayerNormFusedBackpropCustom")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        LayerNormFusedBackpropOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
