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

// See docs in ../ops/spectral_ops.cc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif

namespace tensorflow {

class FFTBase : public OpKernel {
 public:
  explicit FFTBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const TensorShape& input_shape = in.shape();
    const int fft_rank = Rank();
    OP_REQUIRES(
        ctx, input_shape.dims() >= fft_rank,
        errors::InvalidArgument("Input must have rank of at least ", fft_rank,
                                " but got: ", input_shape.DebugString()));

    Tensor* out;
    TensorShape output_shape = input_shape;
    uint64 fft_shape[3] = {0, 0, 0};

    // In R2C or C2R mode, we use a second input to specify the FFT length
    // instead of inferring it from the input shape.
    if (IsReal()) {
      const Tensor& fft_length = ctx->input(1);
      OP_REQUIRES(ctx,
                  fft_length.shape().dims() == 1 &&
                      fft_length.shape().dim_size(0) == fft_rank,
                  errors::InvalidArgument("fft_length must have shape [",
                                          fft_rank, "]"));

      auto fft_length_as_vec = fft_length.vec<int32>();
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] = fft_length_as_vec(i);
        // Each input dimension must have length of at least fft_shape[i]. For
        // IRFFTs, the inner-most input dimension must have length of at least
        // fft_shape[i] / 2 + 1.
        bool inner_most = (i == fft_rank - 1);
        uint64 min_input_dim_length =
            !IsForward() && inner_most ? fft_shape[i] / 2 + 1 : fft_shape[i];
        auto input_index = input_shape.dims() - fft_rank + i;
        OP_REQUIRES(
            ctx,
            // We pass through empty tensors, so special case them here.
            input_shape.dim_size(input_index) == 0 ||
                input_shape.dim_size(input_index) >= min_input_dim_length,
            errors::InvalidArgument(
                "Input dimension ", input_index,
                " must have length of at least ", min_input_dim_length,
                " but got: ", input_shape.dim_size(input_index)));
        uint64 dim = IsForward() && inner_most && fft_shape[i] != 0
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
    if (input_shape.num_elements() == 0) {
      return;
    }

    DoFFT(ctx, in, fft_shape, out);
  }

 protected:
  virtual int Rank() const = 0;
  virtual bool IsForward() const = 0;
  virtual bool IsReal() const = 0;

  // The function that actually computes the FFT.
  virtual void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
                     Tensor* out) = 0;
};

typedef Eigen::ThreadPoolDevice CPUDevice;

template <bool Forward, bool _Real, int FFTRank>
class FFTCPU : public FFTBase {
 public:
  using FFTBase::FFTBase;

 protected:
  int Rank() const override { return FFTRank; }
  bool IsForward() const override { return Forward; }
  bool IsReal() const override { return _Real; }

  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
    // Create the axes (which are always trailing).
    const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
    auto device = ctx->eigen_device<CPUDevice>();

    if (!IsReal()) {
      auto input = Tensor(in).flat_inner_dims<complex64, FFTRank + 1>();
      // Compute the FFT using eigen.
      auto output = out->flat_inner_dims<complex64, FFTRank + 1>();
      constexpr auto direction =
          Forward ? Eigen::FFT_FORWARD : Eigen::FFT_REVERSE;
      output.device(device) =
          input.template fft<Eigen::BothParts, direction>(axes);
    } else {
      if (IsForward()) {
        auto input = Tensor(in).flat_inner_dims<float, FFTRank + 1>();
        const auto input_dims = input.dimensions();

        // Slice input to fft_shape on its inner-most dimensions.
        Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> input_slice_sizes;
        input_slice_sizes[0] = input_dims[0];
        TensorShape temp_shape{input_dims[0]};
        for (int i = 1; i <= FFTRank; ++i) {
          input_slice_sizes[i] = fft_shape[i - 1];
          temp_shape.AddDim(fft_shape[i - 1]);
        }

        auto output = out->flat_inner_dims<complex64, FFTRank + 1>();
        const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;

        // Compute the full FFT using a temporary tensor.
        Tensor temp;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<complex64>::v(),
                                               temp_shape, &temp));
        auto full_fft = temp.flat_inner_dims<complex64, FFTRank + 1>();
        full_fft.device(device) =
            input.slice(zero_start_indices, input_slice_sizes)
                .template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);

        // Slice away the negative frequency components.
        output.device(device) =
            full_fft.slice(zero_start_indices, output.dimensions());
      } else {
        // Reconstruct the full FFT and take the inverse.
        auto input = Tensor(in).flat_inner_dims<complex64, FFTRank + 1>();
        auto output = out->flat_inner_dims<float, FFTRank + 1>();
        const auto input_dims = input.dimensions();

        // Calculate the shape of the temporary tensor for the full FFT and the
        // region we will slice from input given fft_shape. We slice input to
        // fft_shape on its inner-most dimensions, except the last (which we
        // slice to fft_shape[-1] / 2 + 1).
        Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> input_slice_sizes;
        input_slice_sizes[0] = input_dims[0];
        TensorShape full_fft_shape;
        full_fft_shape.AddDim(input_dims[0]);
        for (auto i = 1; i <= FFTRank; i++) {
          input_slice_sizes[i] =
              i == FFTRank ? fft_shape[i - 1] / 2 + 1 : fft_shape[i - 1];
          full_fft_shape.AddDim(fft_shape[i - 1]);
        }

        Tensor temp;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<complex64>::v(),
                                               full_fft_shape, &temp));
        auto full_fft = temp.flat_inner_dims<complex64, FFTRank + 1>();

        // Calculate the starting point and range of the source of
        // negative frequency part.
        auto neg_sizes = input_slice_sizes;
        neg_sizes[FFTRank] =
            fft_shape[FFTRank - 1] - input_slice_sizes[FFTRank];
        Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_target_indices;
        neg_target_indices[FFTRank] = input_slice_sizes[FFTRank];

        const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> start_indices;
        Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_start_indices;
        neg_start_indices[FFTRank] = 1;

        full_fft.slice(start_indices, input_slice_sizes).device(device) =
            input.slice(start_indices, input_slice_sizes);

        // First, conduct IFFTs on outer dimensions. We save computation (and
        // avoid touching uninitialized memory) by slicing full_fft to the
        // subregion we wrote input to.
        if (FFTRank > 1) {
          const auto outer_axes =
              Eigen::ArrayXi::LinSpaced(FFTRank - 1, 1, FFTRank - 1);
          full_fft.slice(start_indices, input_slice_sizes).device(device) =
              full_fft.slice(start_indices, input_slice_sizes)
                  .template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(
                      outer_axes);
        }

        // Reconstruct the full FFT by appending reversed and conjugated
        // spectrum as the negative frequency part.
        Eigen::array<bool, FFTRank + 1> reverse_last_axis;
        for (auto i = 0; i <= FFTRank; i++) {
          reverse_last_axis[i] = i == FFTRank;
        }

        if (neg_sizes[FFTRank] != 0) {
          full_fft.slice(neg_target_indices, neg_sizes).device(device) =
              full_fft.slice(neg_start_indices, neg_sizes)
                  .reverse(reverse_last_axis)
                  .conjugate();
        }

        auto inner_axis = Eigen::array<int, 1>{FFTRank};
        output.device(device) =
            full_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(
                inner_axis);
      }
    }
  }
};

// Use labels to distinguish between internal and open source versions
// of these kernels.
#ifdef PLATFORM_GOOGLE
#define FFT_LABEL "eigen"
#else
#define FFT_LABEL ""
#endif

REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<false, false, 3>);

REGISTER_KERNEL_BUILDER(Name("RFFT").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(Name("IRFFT").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(Name("RFFT2D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(Name("IRFFT2D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(Name("RFFT3D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(Name("IRFFT3D").Device(DEVICE_CPU).Label(FFT_LABEL),
                        FFTCPU<false, true, 3>);

#undef FFT_LABEL

#if GOOGLE_CUDA

namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

// A class to provide scratch-space allocator for Stream-Executor Cufft
// callback. Tensorflow is responsible for releasing the temporary buffers after
// the kernel finishes.
// TODO(yangzihao): Refactor redundant code in subclasses of ScratchAllocator
// into base class.
class CufftScratchAllocator : public se::ScratchAllocator {
 public:
  ~CufftScratchAllocator() override {}
  CufftScratchAllocator(int64 memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  int64 GetMemoryLimitInBytes(se::Stream* stream) override {
    return memory_limit_;
  }
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      se::Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;
    if (byte_size > memory_limit_) {
      return se::port::StatusOr<se::DeviceMemory<uint8>>();
    }
    AllocationAttributes allocation_attr;
    allocation_attr.no_retry_on_failure = true;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<se::DeviceMemory<uint8>>();
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return se::port::StatusOr<se::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 memory_limit_;
  int64 total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

}  // end namespace

int64 GetCufftWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes) {
  const char* workspace_limit_in_mb_str = getenv(envvar_in_mb.c_str());
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64 scratch_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar(envvar_in_mb, default_value_in_bytes,
                                        &scratch_limit_in_mb);
    if (!status.ok()) {
      LOG(WARNING) << "Invalid value for env-var " << envvar_in_mb << ": "
                   << workspace_limit_in_mb_str;
    } else {
      return scratch_limit_in_mb * (1 << 20);
    }
  }
  return default_value_in_bytes;
}

class FFTGPUBase : public FFTBase {
 public:
  using FFTBase::FFTBase;

 protected:
  static int64 CufftScratchSize;
  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
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
    const uint64 input_stride = 1;
    uint64 input_distance = 1;
    uint64 output_embed[3];
    const uint64 output_stride = 1;
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
        IsReal() ? (IsForward() ? se::fft::Type::kR2C : se::fft::Type::kC2R)
                 : (IsForward() ? se::fft::Type::kC2CForward
                                : se::fft::Type::kC2CInverse);

    CufftScratchAllocator scratch_allocator(CufftScratchSize, ctx);
    auto plan =
        stream->parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
            stream, fft_rank, fft_shape, input_embed, input_stride,
            input_distance, output_embed, output_stride, output_distance,
            kFftType, kInPlaceFft, batch_size, &scratch_allocator);

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

int64 FFTGPUBase::CufftScratchSize = GetCufftWorkspaceLimit(
    // default value is in bytes despite the name of the environment variable
    "TF_CUFFT_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
);

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
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
