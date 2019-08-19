/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batch_matmul_op_impl.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

template <typename Device, typename Scalar>
class BatchGemmOp : public OpKernel {
 public:
  explicit BatchGemmOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &transpose_b_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
  }

  ~BatchGemmOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);

    ValidateInputTensors(ctx, in0, in1);

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            in0.shape().DebugString(), " vs. ", in1.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();
    auto d0 = in0.dim_size(in0.dims() - 2);
    auto d1 = in0.dim_size(in0.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in0, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in0.shape().DebugString()));
    auto d2 = in1.dim_size(in1.dims() - 2);
    auto d3 = in1.dim_size(in1.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in1, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in1.shape().DebugString()));
    if (transpose_a_) std::swap(d0, d1);
    if (transpose_b_) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", transpose_a_, " ", transpose_b_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);

    Tensor out_reshaped;
    OP_REQUIRES(ctx, out_reshaped.CopyFrom(in2, out_shape),
                errors::Internal("Failed to reshape output from ",
                                 in2.shape().DebugString(), " to ",
                                 out_shape.DebugString()));
    ctx->set_output(0, out_reshaped);

    if (out_reshaped.NumElements() == 0) {
      return;
    }
    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out_reshaped.flat<Scalar>());
      return;
    }

    LaunchBatchMatMul<Device, Scalar>::Launch(ctx, in0_reshaped, in1_reshaped,
                                              transpose_a_, transpose_b_, bcast,
                                              &out_reshaped, alpha_, beta_);
  }

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) {
    // Enable broadcasting support. Validity of broadcasting is checked in
    // BaseBatchMatMulOp.
    OP_REQUIRES(
        ctx, in0.dims() >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims()));
    OP_REQUIRES(
        ctx, in1.dims() >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims()));
  }

  bool transpose_a_;
  bool transpose_b_;
  float alpha_;
  float beta_;
};

#define REGISTER_BATCH_GEMM_GPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BatchGemm").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      BatchGemmOp<GPUDevice, TYPE>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_float(REGISTER_BATCH_GEMM_GPU);
TF_CALL_double(REGISTER_BATCH_GEMM_GPU);
TF_CALL_half(REGISTER_BATCH_GEMM_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
