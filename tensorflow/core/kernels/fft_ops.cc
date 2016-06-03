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

class FFTGPUBase : public OpKernel {
 public:
  explicit FFTGPUBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const TensorShape& shape = in.shape();
    if (IsBatched()) {
      OP_REQUIRES(
          ctx, shape.dims() >= Rank(),
          errors::InvalidArgument("Input must have rank of at least ", Rank(),
                                  " but got: ", shape.DebugString()));
    } else {
      OP_REQUIRES(ctx, shape.dims() == Rank(),
                  errors::InvalidArgument("Input must be of rank ", Rank(),
                                          " but got: ", shape.DebugString()));
    }
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &out));
    if (shape.num_elements() == 0) {
      return;
    }
    DoFFT(ctx, in, out);
  }

 protected:
  virtual int Rank() const = 0;
  virtual bool IsForward() const = 0;
  virtual bool IsBatched() const = 0;

 private:
  void DoFFT(OpKernelContext* ctx, const Tensor& in, Tensor* out) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    const TensorShape& shape = in.shape();
    auto src = AsDeviceMemory<complex64>(in.flat<complex64>().data());
    auto dst = AsDeviceMemory<complex64>(out->flat<complex64>().data());

    const int rank = Rank();
    int batch_size = 1;
    for (int i = 0; i < shape.dims() - rank; ++i) {
      batch_size *= shape.dim_size(i);
    }
    uint64 data_length = 1;
    uint64 data_dims[3];
    for (int i = 0; i < rank; ++i) {
      auto dim = shape.dim_size(shape.dims() - rank + i);
      data_length *= dim;
      data_dims[i] = dim;
    }

    constexpr uint64* kInputEmbed = nullptr;
    constexpr uint64 kInputStride = 1;
    constexpr uint64 kInputDistance = 1;
    constexpr uint64* kOutputEmbed = nullptr;
    constexpr uint64 kOutputStride = 1;
    constexpr uint64 kOutputDistance = 1;
    constexpr bool kInPlaceFft = false;

    auto plan = stream->parent()->AsFft()->CreateBatchedPlan(
        stream, rank, data_dims, kInputEmbed, kInputStride, kInputDistance,
        kOutputEmbed, kOutputStride, kOutputDistance,
        IsForward() ? perftools::gputools::fft::Type::kC2CForward
                    : perftools::gputools::fft::Type::kC2CInverse,
        kInPlaceFft, batch_size);

    OP_REQUIRES(
        ctx, stream->ThenFft(plan.get(), src, &dst).ok(),
        errors::Internal("c2c fft failed : in.shape=", shape.DebugString()));
    if (!IsForward()) {
      auto alpha = complex64(1.f / data_length);
      OP_REQUIRES(
          ctx, stream->ThenBlasScal(shape.num_elements(), alpha, &dst, 1).ok(),
          errors::Internal("BlasScal failed : in.shape=", shape.DebugString()));
    }
  }
};

template <bool Forward, bool Batched, int FFTRank>
class FFTGPU : public FFTGPUBase {
 public:
  static_assert(FFTRank >= 1 && FFTRank <= 3,
                "Only 1D, 2D and 3D FFTs supported.");
  explicit FFTGPU(OpKernelConstruction* ctx) : FFTGPUBase(ctx) {}

 protected:
  int Rank() const override { return FFTRank; }
  bool IsForward() const override { return Forward; }
  bool IsBatched() const override { return Batched; }
};

REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_GPU), FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_GPU),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_GPU),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_GPU),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_GPU),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_GPU),
                        FFTGPU<false, false, 3>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT").Device(DEVICE_GPU),
                        FFTGPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT").Device(DEVICE_GPU),
                        FFTGPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT2D").Device(DEVICE_GPU),
                        FFTGPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT2D").Device(DEVICE_GPU),
                        FFTGPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT3D").Device(DEVICE_GPU),
                        FFTGPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT3D").Device(DEVICE_GPU),
                        FFTGPU<false, true, 3>);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
