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
    const int fft_rank = Rank();
    OP_REQUIRES(
        ctx, shape.dims() >= fft_rank,
        errors::InvalidArgument("Input must have rank of at least ", fft_rank,
                                " but got: ", shape.DebugString()));

    Tensor* out;
    TensorShape output_shape = shape;
    uint64 fft_shape[3] = {0, 0, 0};

    // In R2C or C2R mode, we use a second input to specify the FFT length
    // instead of inferring it from the input shape.
    if (IsReal()) {
      const Tensor& fft_length = ctx->input(1);
      OP_REQUIRES(ctx,
                  fft_length.shape().dims() == 1 &&
                      fft_length.shape().dim_size(0) == fft_rank,
                  errors::InvalidArgument("fft_length must  have shape [",
                                          fft_rank, "]"));

      auto fft_length_as_vec = fft_length.vec<int32>();
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] = fft_length_as_vec(i);
        uint64 dim = IsForward() && i == fft_rank - 1 && fft_shape[i] != 0
                         ? fft_shape[i] / 2 + 1
                         : fft_shape[i];
        output_shape.set_dim(output_shape.dims() - fft_rank + i, dim);
      }
    } else {
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] =
            output_shape.dim_size(output_shape.dims() - fft_rank + i);
      }
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
    if (shape.num_elements() == 0) {
      return;
    }

    DoFFT(ctx, in, fft_shape, out);
  }

 protected:
  virtual int Rank() const = 0;
  virtual bool IsForward() const = 0;
  virtual bool IsReal() const = 0;

 private:
  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();

    const int fft_rank = Rank();
    int batch_size = 1;
    for (int i = 0; i < input_shape.dims() - fft_rank; ++i) {
      batch_size *= input_shape.dim_size(i);
    }
    uint64 input_embed[3];
    uint64 input_stride = 1;
    uint64 input_distance = 1;
    uint64 output_embed[3];
    uint64 output_stride = 1;
    uint64 output_distance = 1;

    for (int i = 0; i < fft_rank; ++i) {
      auto dim_offset = input_shape.dims() - fft_rank + i;
      input_embed[i] = input_shape.dim_size(dim_offset);
      input_distance *= input_shape.dim_size(dim_offset);
      output_embed[i] = output_shape.dim_size(dim_offset);
      output_distance *= output_shape.dim_size(dim_offset);
    }

    constexpr bool kInPlaceFft = false;
    const auto kFftType =
        IsReal() ? (IsForward() ? perftools::gputools::fft::Type::kR2C
                                : perftools::gputools::fft::Type::kC2R)
                 : (IsForward() ? perftools::gputools::fft::Type::kC2CForward
                                : perftools::gputools::fft::Type::kC2CInverse);

    auto plan = stream->parent()->AsFft()->CreateBatchedPlan(
        stream, fft_rank, fft_shape, input_embed, input_stride, input_distance,
        output_embed, output_stride, output_distance, kFftType, kInPlaceFft,
        batch_size);

    if (IsReal()) {
      if (IsForward()) {
        auto src = AsDeviceMemory<float>(in.flat<float>().data());
        auto dst = AsDeviceMemory<complex64>(out->flat<complex64>().data());
        OP_REQUIRES(
            ctx, stream->ThenFft(plan.get(), src, &dst).ok(),
            errors::Internal("fft failed : type=", static_cast<int>(kFftType),
                             " in.shape=", input_shape.DebugString()));
      } else {
        auto src = AsDeviceMemory<complex64>(in.flat<complex64>().data());
        auto dst = AsDeviceMemory<float>(out->flat<float>().data());
        OP_REQUIRES(
            ctx, stream->ThenFft(plan.get(), src, &dst).ok(),
            errors::Internal("fft failed : type=", static_cast<int>(kFftType),
                             " in.shape=", input_shape.DebugString()));
        auto alpha = 1.f / output_distance;
        OP_REQUIRES(
            ctx,
            stream->ThenBlasScal(output_shape.num_elements(), alpha, &dst, 1)
                .ok(),
            errors::Internal("BlasScal failed : in.shape=",
                             input_shape.DebugString()));
      }
    } else {
      auto src = AsDeviceMemory<complex64>(in.flat<complex64>().data());
      auto dst = AsDeviceMemory<complex64>(out->flat<complex64>().data());
      OP_REQUIRES(
          ctx, stream->ThenFft(plan.get(), src, &dst).ok(),
          errors::Internal("fft failed : type=", static_cast<int>(kFftType),
                           " in.shape=", input_shape.DebugString()));
      if (!IsForward()) {
        auto alpha = complex64(1.f / output_distance);
        OP_REQUIRES(
            ctx,
            stream->ThenBlasScal(output_shape.num_elements(), alpha, &dst, 1)
                .ok(),
            errors::Internal("BlasScal failed : in.shape=",
                             input_shape.DebugString()));
      }
    }
  }
};

template <bool Forward, bool _Real, int FFTRank>
class FFTGPU : public FFTGPUBase {
 public:
  static_assert(FFTRank >= 1 && FFTRank <= 3,
                "Only 1D, 2D and 3D FFTs supported.");
  explicit FFTGPU(OpKernelConstruction* ctx) : FFTGPUBase(ctx) {}

 protected:
  int Rank() const override { return FFTRank; }
  bool IsForward() const override { return Forward; }
  bool IsReal() const override { return _Real; }
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

REGISTER_KERNEL_BUILDER(
    Name("RFFT").Device(DEVICE_GPU).HostMemory("fft_length"),
    FFTGPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT").Device(DEVICE_GPU).HostMemory("fft_length"),
    FFTGPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT2D").Device(DEVICE_GPU).HostMemory("fft_length"),
    FFTGPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT2D").Device(DEVICE_GPU).HostMemory("fft_length"),
    FFTGPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT3D").Device(DEVICE_GPU).HostMemory("fft_length"),
    FFTGPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT3D").Device(DEVICE_GPU).HostMemory("fft_length"),
    FFTGPU<false, true, 3>);

// Deprecated kernels.
REGISTER_KERNEL_BUILDER(Name("BatchFFT").Device(DEVICE_GPU),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT").Device(DEVICE_GPU),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT2D").Device(DEVICE_GPU),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT2D").Device(DEVICE_GPU),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT3D").Device(DEVICE_GPU),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT3D").Device(DEVICE_GPU),
                        FFTGPU<false, false, 3>);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
