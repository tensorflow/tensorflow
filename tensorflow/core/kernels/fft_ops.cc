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

#define EIGEN_USE_THREADS

// See docs in ../ops/fft_ops.cc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

namespace {
// TODO(vrv/zhifengc): Refactor AsDeviceMemory() into GPUUtil.
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // end namespace

class FFT2DGPUBase : public OpKernel {
 public:
  explicit FFT2DGPUBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const TensorShape& shape = in.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(shape),
                errors::InvalidArgument("Input is not a matrix: ",
                                        shape.DebugString()));
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &out));
    if (shape.num_elements() == 0) {
      return;
    }
    DoFFT(ctx, in, out);
  }

 protected:
  virtual bool IsForward() = 0;

 private:
  void DoFFT(OpKernelContext* ctx, const Tensor& in, Tensor* out) {
    auto* stream = ctx->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    const TensorShape& shape = in.shape();
    auto n0 = shape.dim_size(0);
    auto n1 = shape.dim_size(1);
    auto src = AsDeviceMemory<complex64>(in.flat<complex64>().data());
    auto dst = AsDeviceMemory<complex64>(out->flat<complex64>().data());

    auto plan = stream->parent()->AsFft()->Create2dPlan(
        stream, n0, n1,
        IsForward() ? perftools::gputools::fft::Type::kC2CForward
                    : perftools::gputools::fft::Type::kC2CInverse,
        false /* not inplace */);
    OP_REQUIRES(
        ctx, stream->ThenFft(plan.get(), src, &dst).ok(),
        errors::Internal("c2c fft failed : in.shape=", shape.DebugString()));
    if (!IsForward()) {
      auto alpha = complex64(1.f / (n0 * n1));
      OP_REQUIRES(
          ctx, stream->ThenBlasScal(n0 * n1, alpha, &dst, 1).ok(),
          errors::Internal("BlasScal failed : in.shape=", shape.DebugString()));
    }
  }
};

template <bool forward>
class FFT2DGPU : public FFT2DGPUBase {
 public:
  explicit FFT2DGPU(OpKernelConstruction* ctx) : FFT2DGPUBase(ctx) {}

 protected:
  bool IsForward() override { return forward; }
};
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_GPU), FFT2DGPU<true>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_GPU), FFT2DGPU<false>);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
