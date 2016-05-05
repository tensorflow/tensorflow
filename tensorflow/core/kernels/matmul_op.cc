/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fill_functor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

#endif  // GOOGLE_CUDA

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMatMul;

// On CPUs, we ignore USE_CUBLAS
template <typename T>
struct LaunchMatMulCPU {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      Tensor* out) {
    functor::MatMulFunctor<CPUDevice, T>()(ctx->eigen_device<CPUDevice>(),
                                           out->matrix<T>(), a.matrix<T>(),
                                           b.matrix<T>(), dim_pair);
  }
};

template <typename T, bool USE_CUBLAS>
struct LaunchMatMul<CPUDevice, T, USE_CUBLAS> : public LaunchMatMulCPU<T> {};

#if GOOGLE_CUDA

template <typename T>
struct LaunchMatMul<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      Tensor* out) {
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose,
        perftools::gputools::blas::Transpose::kTranspose};
    const uint64 m = a.dim_size(1 - dim_pair[0].first);
    const uint64 k = a.dim_size(dim_pair[0].first);
    const uint64 n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;
    auto blas_transpose_a = trans[transpose_a];
    auto blas_transpose_b = trans[transpose_b];

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto a_ptr = AsDeviceMemory(a.template flat<T>().data());
    auto b_ptr = AsDeviceMemory(b.template flat<T>().data());
    auto c_ptr = AsDeviceMemory(out->template flat<T>().data());

    // Cublas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A' (' stands for transpose)
    bool blas_launch_status =
        stream->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k, 1.0f,
                             b_ptr, transpose_b ? k : n, a_ptr,
                             transpose_a ? m : k, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal(
          "Blas SGEMM launch failed : a.shape=(", a.dim_size(0), ", ",
          a.dim_size(1), "), b.shape=(", b.dim_size(0), ", ", b.dim_size(1),
          "), m=", m, ", n=", n, ", k=", k));
    }
  }
};

#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class MatMulOp : public OpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(ctx,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-compatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    LaunchMatMul<Device, T, USE_CUBLAS>::launch(ctx, this, a, b, dim_pair, out);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

namespace functor {

// Partial specialization MatMulFunctor<Device=CPUDevice, T>.
template <typename T>
struct MatMulFunctor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    MatMul<CPUDevice>(d, out, in0, in1, dim_pair);
  }
};

}  // end namespace functor

#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      MatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatMul").Device(DEVICE_CPU).TypeConstraint<T>("T").Label("eigen"), \
      MatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>)

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MatMul").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      MatMulOp<GPUDevice, T, true /* cublas, true by default */>); \
  REGISTER_KERNEL_BUILDER(Name("MatMul")                           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .Label("cublas"),                    \
                          MatMulOp<GPUDevice, T, true /* cublas */>)

REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(complex64);
#if GOOGLE_CUDA
REGISTER_GPU(float);
// REGISTER_GPU(double);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
