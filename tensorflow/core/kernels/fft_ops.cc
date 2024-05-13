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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ducc/google/fft.h"  // from @ducc
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/ThreadPool"  // from @eigen_archive
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if defined(GOOGLE_CUDA) && GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // CUDA_VERSION
#endif

namespace tensorflow {

namespace {

using std::size_t;
using Shape = ducc0::google::Shape;
using Stride = ducc0::google::Stride;
absl::Status DuccFftImpl(const Eigen::ThreadPoolDevice& device,
                         const Tensor& in, Tensor* out,
                         const uint64_t* fft_shape,
                         const std::vector<size_t>& axes, bool forward) {
  const size_t fft_rank = axes.size();
  Shape in_shape(in.dims());
  Stride in_stride(in.dims());
  Shape out_shape(out->dims());
  Stride out_stride(out->dims());

  size_t next_stride = 1;
  for (int i = in.dims(); i-- > 0;) {
    in_shape[i] = in.dim_size(i);
    in_stride[i] = next_stride;
    next_stride *= in_shape[i];
  }
  next_stride = 1;
  for (int i = out->dims(); i-- > 0;) {
    out_shape[i] = out->dim_size(i);
    out_stride[i] = next_stride;
    next_stride *= out_shape[i];
  }

  // DUCC doesn't handle the case where fft_size[i] < input_size[i],
  // so manually adjust inputs if required.  If doing irfft, the limit
  // of the last axis is actually fft_size[i]/2 + 1.
  const bool is_iffrt = !(forward || out->dtype() == DT_COMPLEX128 ||
                          out->dtype() == DT_COMPLEX64);
  for (int i = 0; i < fft_rank; ++i) {
    int limit = (is_iffrt && (i == (fft_rank - 1))) ? fft_shape[i] / 2 + 1
                                                    : fft_shape[i];
    if (in_shape[axes[i]] > limit) {
      in_shape[axes[i]] = limit;
    }
  }

  double inv_scale = 1.0;
  for (int i = 0; i < fft_rank; ++i) {
    inv_scale *= out_shape[axes[i]];
  }
  double scale = forward ? 1.0 : 1.0 / inv_scale;

  Eigen::ThreadPoolInterface* thread_pool = device.getPool();

  if (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128) {
    auto input = in.template flat<complex128>();
    auto output = out->template flat<complex128>();
    ducc0::google::c2c<double>(input.data(), in_shape, in_stride, output.data(),
                               out_shape, out_stride, axes, forward, scale,
                               thread_pool);
  } else if (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) {
    auto input = in.template flat<complex64>();
    auto output = out->template flat<complex64>();
    ducc0::google::c2c<float>(input.data(), in_shape, in_stride, output.data(),
                              out_shape, out_stride, axes, forward,
                              static_cast<float>(scale), thread_pool);
  } else if (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128 &&
             forward) {
    auto input = in.flat<double>();
    auto output = out->flat<complex128>();
    ducc0::google::r2c<double>(input.data(), in_shape, in_stride, output.data(),
                               out_shape, out_stride, axes, forward, scale,
                               thread_pool);
  } else if (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64 &&
             forward) {
    auto input = in.flat<float>();
    auto output = out->flat<complex64>();
    ducc0::google::r2c<float>(input.data(), in_shape, in_stride, output.data(),
                              out_shape, out_stride, axes, forward,
                              static_cast<float>(scale), thread_pool);
  } else if (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE &&
             !forward) {
    auto input = in.flat<complex128>();
    auto output = out->flat<double>();
    ducc0::google::c2r<double>(input.data(), in_shape, in_stride, output.data(),
                               out_shape, out_stride, axes, forward, scale,
                               thread_pool);
  } else if (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT &&
             !forward) {
    auto input = in.flat<complex64>();
    auto output = out->flat<float>();
    ducc0::google::c2r<float>(input.data(), in_shape, in_stride, output.data(),
                              out_shape, out_stride, axes, forward,
                              static_cast<float>(scale), thread_pool);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid FFT parameters, in.dtype=", in.dtype(),
                     ", out->dtype=", out->dtype(), ", forward=", forward));
  }
  return absl::OkStatus();
}

}  // namespace

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
        OP_REQUIRES(ctx, fft_length_as_vec(i) >= 0,
                    errors::InvalidArgument(
                        "fft_length[", i,
                        "] must >= 0, but got: ", fft_length_as_vec(i)));
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

    if (IsReal()) {
      if (IsForward()) {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64) ||
                (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128),
            errors::InvalidArgument("Wrong types for forward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      } else {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT) ||
                (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE),
            errors::InvalidArgument("Wrong types for backward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      }
    } else {
      OP_REQUIRES(
          ctx,
          (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) ||
              (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128),
          errors::InvalidArgument("Wrong types for FFT: in=", in.dtype(),
                                  " out=", out->dtype()));
    }

    if (input_shape.num_elements() == 0) {
      DCHECK_EQ(0, output_shape.num_elements());
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

class FFTNBase : public OpKernel {
 public:
  explicit FFTNBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const TensorShape& input_shape = in.shape();

    Tensor* out;
    TensorShape output_shape = input_shape;

    const Tensor& fft_length = ctx->input(1);
    const Tensor& axes = ctx->input(2);
    unsigned int input_rank = input_shape.dims();
    const int fft_rank = axes.dim_size(0);
    std::vector<uint64_t> fft_shape(fft_rank);
    std::vector<int32_t> axes_shape(fft_rank);  // List of axes to transform.

    OP_REQUIRES(ctx, input_rank >= fft_rank,
                absl::InvalidArgumentError(
                    absl::StrCat("Input must have rank of at least ", fft_rank,
                                 " but got: ", input_shape.DebugString())));
    auto axes_as_vec = axes.vec<int32>();
    // TODO(b/295964813): fftn() ops now doesn't work for arbitrary axes.
    for (int i = 0; i < fft_rank; ++i) {
      axes_shape[i] = axes_as_vec(i) % input_rank;
      if (axes_as_vec(i) < 0) {
        axes_shape[i] = axes_as_vec(i) + input_rank;
      } else {
        axes_shape[i] = axes_as_vec(i);
      }
      if (i > 0)
        OP_REQUIRES(ctx, (axes_shape[i - 1] + 1) == axes_shape[i],
                    absl::InvalidArgumentError(
                        "axes must be successive and ascending."));
    }
    OP_REQUIRES(ctx, (axes_shape[fft_rank - 1] == input_rank - 1),
                absl::InvalidArgumentError(
                    "The last axis to perform transform on must be -1."));

    // In R2C or C2R mode, we use a second input to specify the FFT length
    // instead of inferring it from the input shape.
    OP_REQUIRES(ctx,
                fft_length.shape().dims() == 1 &&
                    fft_length.shape().dim_size(0) == fft_rank,
                absl::InvalidArgumentError(absl::StrCat(
                    "fft_length must have shape [", fft_rank,
                    "], but got: ", fft_length.shape().dim_size(0), ".")));
    auto fft_length_as_vec = fft_length.vec<int32>();
    for (int i = 0; i < fft_rank; ++i) {
      OP_REQUIRES(ctx, fft_length_as_vec(i) >= 0,
                  absl::InvalidArgumentError(absl::StrCat(
                      "fft_length[", i,
                      "] must >= 0, but got: ", fft_length_as_vec(i))));
      fft_shape[i] = fft_length_as_vec(i);
      if (IsReal()) {
        bool inner_most = (i == fft_rank - 1);
        uint64 min_input_dim_length =
            !IsForward() && inner_most ? fft_shape[i] / 2 + 1 : fft_shape[i];
        auto input_index = input_rank - fft_rank + i;
        OP_REQUIRES(
            ctx,
            // We pass through empty tensors, so special case them here.
            input_shape.dim_size(input_index) == 0 ||
                input_shape.dim_size(input_index) >= min_input_dim_length,
            absl::InvalidArgumentError(absl::StrCat(
                "Input dimension ", input_index,
                " must have length of at least ", min_input_dim_length,
                " but got: ", input_shape.dim_size(input_index))));
        uint64 dim = IsForward() && inner_most && fft_shape[i] != 0
                         ? fft_shape[i] / 2 + 1
                         : fft_shape[i];
        output_shape.set_dim(output_shape.dims() - fft_rank + i, dim);
      } else {
        output_shape.set_dim(output_shape.dims() - fft_rank + i, fft_shape[i]);
      }
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (IsReal()) {
      if (IsForward()) {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64) ||
                (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128),
            absl::InvalidArgumentError(absl::StrCat(
                "Wrong types for forward real FFT: in=", in.dtype(),
                " out=", out->dtype())));
      } else {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT) ||
                (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE),
            absl::InvalidArgumentError(absl::StrCat(
                "Wrong types for backward real FFT: in=", in.dtype(),
                " out=", out->dtype())));
      }
    } else {
      OP_REQUIRES(
          ctx,
          (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) ||
              (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128),
          absl::InvalidArgumentError(absl::StrCat(
              "Wrong types for FFT: in=", in.dtype(), " out=", out->dtype())));
    }

    if (input_shape.num_elements() == 0) {
      DCHECK_EQ(0, output_shape.num_elements());
      return;
    }
    DoFFTN(ctx, in, fft_shape.data(), axes_shape.data(), out);
  }

 protected:
  virtual bool IsReal() const = 0;
  virtual bool IsForward() const = 0;

  // The function that actually computes the FFT.
  virtual void DoFFTN(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
                      int32* axes_shape, Tensor* out) = 0;
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
    std::vector<size_t> axes(Rank());
    int batch_dims = in.dims() - FFTRank;

    for (int i = 0; i < Rank(); ++i) {
      axes[i] = batch_dims + i;
    }

    OP_REQUIRES_OK(ctx, DuccFftImpl(ctx->eigen_device<CPUDevice>(), in, out,
                                    fft_shape, axes, Forward));
  }
};

REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_CPU), FFTCPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_CPU),
                        FFTCPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_CPU),
                        FFTCPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_CPU),
                        FFTCPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_CPU),
                        FFTCPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_CPU),
                        FFTCPU<false, false, 3>);

REGISTER_KERNEL_BUILDER(Name("RFFT").Device(DEVICE_CPU), FFTCPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(Name("IRFFT").Device(DEVICE_CPU),
                        FFTCPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(Name("RFFT2D").Device(DEVICE_CPU),
                        FFTCPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(Name("IRFFT2D").Device(DEVICE_CPU),
                        FFTCPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(Name("RFFT3D").Device(DEVICE_CPU),
                        FFTCPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(Name("IRFFT3D").Device(DEVICE_CPU),
                        FFTCPU<false, true, 3>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

namespace {

// Info required for caching an FFT plan.
struct FftPlanInfo {
  int rank = 0;
  gtl::InlinedVector<uint64_t, 3> shape{};
  gtl::InlinedVector<uint64_t, 3> input_embed{};
  uint64_t input_stride = 1;
  uint64_t input_distance = 0;
  gtl::InlinedVector<uint64_t, 3> output_embed{};
  uint64_t output_stride = 1;
  uint64_t output_distance = 0;
  se::fft::Type type = se::fft::Type::kInvalid;
  int batch = 0;
  int device_id = 0;

  FftPlanInfo() = default;

  template <typename H>
  friend inline H AbslHashValue(H h, const FftPlanInfo& key) {
    return H::combine(std::move(h), key.rank, key.shape, key.input_embed,
                      key.input_stride, key.input_distance, key.output_embed,
                      key.output_stride, key.output_distance, key.type,
                      key.batch, key.device_id);
  }

  friend inline bool operator==(const FftPlanInfo& lhs,
                                const FftPlanInfo& rhs) {
    return lhs.rank == rhs.rank && lhs.shape == rhs.shape &&
           lhs.input_embed == rhs.input_embed &&
           lhs.input_stride == rhs.input_stride &&
           lhs.input_distance == rhs.input_distance &&
           lhs.output_embed == rhs.output_embed &&
           lhs.output_stride == rhs.output_stride &&
           lhs.output_distance == rhs.output_distance && lhs.type == rhs.type &&
           lhs.batch == rhs.batch && lhs.device_id == rhs.device_id;
  }

  // Create a key to be used for caching plans.
  static FftPlanInfo Create(int rank, const uint64_t* shape,
                            const uint64_t* input_embed, uint64_t input_stride,
                            uint64_t input_distance,
                            const uint64_t* output_embed,
                            uint64_t output_stride, uint64_t output_distance,
                            se::fft::Type type, int batch, int device_id) {
    FftPlanInfo info;
    info.rank = rank;
    info.shape.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      info.shape.push_back(shape[i]);
    }
    if (input_embed != nullptr) {
      info.input_embed.reserve(rank);
      for (int i = 0; i < rank; ++i) {
        info.input_embed.push_back(input_embed[i]);
      }
      info.input_stride = input_stride;
      info.input_distance = input_distance;
    }
    if (output_embed != nullptr) {
      info.output_embed.reserve(rank);
      for (int i = 0; i < rank; ++i) {
        info.output_embed.push_back(output_embed[i]);
      }
      info.output_stride = output_stride;
      info.output_distance = output_distance;
    }
    info.type = type;
    info.batch = batch;
    info.device_id = device_id;
    return info;
  }
};

// Multimap for storing FFT plans.
//
// Plans can be inserted into the cache as long as there is capacity.  They
// can only be extracted from the cache for use.  The multimap is to allow
// inserting multiple identical plans, since each can only have one simultaneous
// user.
//
// Thread-safe after initialization.
class FftPlanCache {
 public:
  using Key = FftPlanInfo;
  using Value = std::unique_ptr<se::fft::Plan>;

  FftPlanCache(size_t capacity)
      : mutex_(), size_(0), capacity_(capacity), cache_() {}

  // Finds and removes a plan from the cache if it exists.  Otherwise,
  // returns std::nullopt.
  std::optional<Value> Extract(const Key& key) {
    tsl::mutex_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return std::nullopt;
    }
    Value value = std::move(it->second.back());
    it->second.pop_back();
    if (it->second.empty()) {
      cache_.erase(it);
    }
    --size_;
    // Explicitly create an optional to avoid a compiler bug with gcc-7.
    return std::optional<Value>(std::move(value));
  }

  // Inserts a plan into the cache as long as there is still capacity.
  void Insert(Key key, Value value) {
    tsl::mutex_lock lock(mutex_);
    if (size_ < capacity_) {
      auto it_inserted = cache_.try_emplace(std::move(key));
      it_inserted.first->second.push_back(std::move(value));
      ++size_;
    } else {
      static bool already_warned = false;
      if (!already_warned) {
        LOG(WARNING) << "The CUDA FFT plan cache capacity of " << capacity_
                     << " has been exceeded. This may lead to extra time being"
                     << " spent constantly creating new plans."
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
                     << " For CUDA 11.x, there is also a memory leak in cuFFT "
                     << " plan creation which may cause GPU memory usage to "
                     << " slowly increase.  If this causes an issue, try"
                     << " modifying your fft parameters to increase cache hits,"
                     << " or build TensorFlow with CUDA 10.x or 12.x, or use"
                     << " explicit device placement to run frequently-changing"
                     << " FFTs on CPU."
#endif
            ;  // NOLINT
        already_warned = true;
      }
    }
  }

 private:
  tsl::mutex mutex_;
  size_t size_ TF_GUARDED_BY(mutex_);
  size_t capacity_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<Key, std::vector<Value>> cache_ TF_GUARDED_BY(mutex_);
};

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
  CufftScratchAllocator(int64_t memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  int64_t GetMemoryLimitInBytes() override { return memory_limit_; }
  tsl::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64_t byte_size) override {
    Tensor temporary_memory;
    if (byte_size > memory_limit_) {
      return tsl::StatusOr<se::DeviceMemory<uint8>>();
    }
    AllocationAttributes allocation_attr;
    allocation_attr.retry_on_failure = false;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return tsl::StatusOr<se::DeviceMemory<uint8>>();
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return tsl::StatusOr<se::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64_t TotalByteSize() { return total_byte_size_; }

 private:
  int64_t memory_limit_;
  int64_t total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

}  // end namespace

int64_t GetCufftWorkspaceLimit(const string& envvar_in_mb,
                               int64_t default_value_in_bytes) {
  const char* workspace_limit_in_mb_str = getenv(envvar_in_mb.c_str());
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64_t scratch_limit_in_mb = -1;
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
  static const int64_t kCufftScratchSize;
  // Capacity is somewhat arbitrary.  Plans don't take up any GPU memory
  // since the scratch space is provided externally.  We don't anticipate
  // ever hitting this limit in practice.
  static constexpr size_t kFftPlanCacheCapacity = 512;

  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, absl::InternalError("No GPU stream available."));

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
    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    const auto kFftType =
        IsReal()
            ? (IsForward()
                   ? (is_complex128 ? se::fft::Type::kD2Z : se::fft::Type::kR2C)
                   : (is_complex128 ? se::fft::Type::kZ2D
                                    : se::fft::Type::kC2R))
            : (IsForward() ? (is_complex128 ? se::fft::Type::kZ2ZForward
                                            : se::fft::Type::kC2CForward)
                           : (is_complex128 ? se::fft::Type::kZ2ZInverse
                                            : se::fft::Type::kC2CInverse));

    CufftScratchAllocator scratch_allocator(kCufftScratchSize, ctx);

    // Plan cache singleton with safe no-destructor initialization.
    static FftPlanCache* plan_cache = new FftPlanCache(kFftPlanCacheCapacity);

    // CUDA cufft plans are device-specific, so grab the GPU device ID.
    int device_id = ctx->device()->tensorflow_accelerator_device_info()->gpu_id;

    // Look for plan in cache.
    FftPlanInfo plan_info =
        FftPlanInfo::Create(fft_rank, fft_shape, input_embed, input_stride,
                            input_distance, output_embed, output_stride,
                            output_distance, kFftType, batch_size, device_id);
    std::unique_ptr<se::fft::Plan> plan = nullptr;
    {
      auto plan_or = plan_cache->Extract(plan_info);
      if (plan_or.has_value()) {
        plan = std::move(*plan_or);
      }
    }

    // Create a new plan if one doesn't exist.  Otherwise, we need only set
    // the scratch allocator.
    auto fft = stream->parent()->AsFft();
    OP_REQUIRES(ctx, fft != nullptr, absl::InternalError("No FFT for stream."));
    if (plan == nullptr) {
      plan = fft->CreateBatchedPlanWithScratchAllocator(
          stream, fft_rank, fft_shape, input_embed, input_stride,
          input_distance, output_embed, output_stride, output_distance,
          kFftType, kInPlaceFft, batch_size, &scratch_allocator);
    } else {
      fft->UpdatePlanWithScratchAllocator(stream, plan.get(),
                                          &scratch_allocator);
    }

    OP_REQUIRES(
        ctx, plan != nullptr,
        errors::Internal(
            "Failed to create cuFFT batched plan with scratch allocator"));

    if (IsReal()) {
      if (IsForward()) {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_DOUBLE);
          DCHECK_EQ(out->dtype(), DT_COMPLEX128);
          DoFFTInternal<double, complex128>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_FLOAT);
          DCHECK_EQ(out->dtype(), DT_COMPLEX64);
          DoFFTInternal<float, complex64>(ctx, stream, plan.get(), kFftType,
                                          output_distance, in, out);
        }
      } else {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_COMPLEX128);
          DCHECK_EQ(out->dtype(), DT_DOUBLE);
          DoFFTInternal<complex128, double>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_COMPLEX64);
          DCHECK_EQ(out->dtype(), DT_FLOAT);
          DoFFTInternal<complex64, float>(ctx, stream, plan.get(), kFftType,
                                          output_distance, in, out);
        }
      }
    } else {
      if (is_complex128) {
        DCHECK_EQ(in.dtype(), DT_COMPLEX128);
        DCHECK_EQ(out->dtype(), DT_COMPLEX128);
        DoFFTInternal<complex128, complex128>(ctx, stream, plan.get(), kFftType,
                                              output_distance, in, out);
      } else {
        DCHECK_EQ(in.dtype(), DT_COMPLEX64);
        DCHECK_EQ(out->dtype(), DT_COMPLEX64);
        DoFFTInternal<complex64, complex64>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
      }
    }

    plan_cache->Insert(std::move(plan_info), std::move(plan));
  }

 private:
  template <typename T>
  struct RealTypeFromComplexType {
    typedef T RealT;
  };

  template <typename T>
  struct RealTypeFromComplexType<std::complex<T>> {
    typedef T RealT;
  };

  template <typename InT, typename OutT>
  void DoFFTInternal(OpKernelContext* ctx, se::Stream* stream,
                     se::fft::Plan* plan, const se::fft::Type fft_type,
                     const uint64 output_distance, const Tensor& in,
                     Tensor* out) {
    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();
    auto src =
        AsDeviceMemory<InT>(in.flat<InT>().data(), input_shape.num_elements());
    auto dst = AsDeviceMemory<OutT>(out->flat<OutT>().data(),
                                    output_shape.num_elements());
    auto fft = stream->parent()->AsFft();
    OP_REQUIRES(ctx, fft != nullptr, absl::InternalError("No FFT for stream."));
    OP_REQUIRES(
        ctx, fft->DoFft(stream, plan, src, &dst),
        errors::Internal("fft failed : type=", static_cast<int>(fft_type),
                         " in.shape=", input_shape.DebugString()));
    if (!IsForward()) {
      typedef typename RealTypeFromComplexType<OutT>::RealT RealT;
      RealT alpha = 1.0 / output_distance;
      auto blas = stream->parent()->AsBlas();
      OP_REQUIRES(ctx, blas != nullptr,
                  absl::InternalError("No Blas for stream."));
      OP_REQUIRES(
          ctx,
          blas->DoBlasScal(stream, output_shape.num_elements(), alpha, &dst, 1),
          errors::Internal("BlasScal failed : in.shape=",
                           input_shape.DebugString()));
    }
  }
};

const int64_t FFTGPUBase::kCufftScratchSize = GetCufftWorkspaceLimit(
    // default value is in bytes despite the name of the environment variable
    "TF_CUFFT_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
);

class FFTNGPUBase : public FFTNBase {
 public:
  using FFTNBase::FFTNBase;

 protected:
  static const int64_t kCufftScratchSize;
  // Capacity is somewhat arbitrary.  Plans don't take up any GPU memory
  // since the scratch space is provided externally.  We don't anticipate
  // ever hitting this limit in practice.
  static constexpr size_t kFftPlanCacheCapacity = 512;

  void DoFFTN(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
              int32* axes_shape, Tensor* out) override {
    int fft_rank = ctx->input(2).dim_size(0);
    // TODO(b/295966566): fftn() ops only support lower dimensions now (1~3).
    OP_REQUIRES(
        ctx, fft_rank >= 1 && fft_rank <= 3,
        absl::InvalidArgumentError("Only 1D, 2D and 3D FFTs supported."));
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, absl::InternalError("No GPU stream available."));

    Eigen::Map<Eigen::ArrayXi> axes(axes_shape, fft_rank);
    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();

    int batch_size = 1;
    uint64 input_stride = 1;
    uint64 output_stride = 1;
    uint64 input_embed[3];
    uint64 output_embed[3];

    uint64 input_distance = 1;
    uint64 output_distance = 1;
    int ax = 0;
    for (int i = 0; i < input_shape.dims(); ++i) {
      if (i >= axes[0]) {
        input_embed[ax] = input_shape.dim_size(i);
        input_distance *= input_shape.dim_size(i);
        output_embed[ax++] = output_shape.dim_size(i);
        output_distance *= output_shape.dim_size(i);
      } else {
        batch_size *= input_shape.dim_size(i);
      }
    }

    constexpr bool kInPlaceFft = false;
    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    const auto kFftType =
        IsReal()
            ? (IsForward()
                   ? (is_complex128 ? se::fft::Type::kD2Z : se::fft::Type::kR2C)
                   : (is_complex128 ? se::fft::Type::kZ2D
                                    : se::fft::Type::kC2R))
            : (IsForward() ? (is_complex128 ? se::fft::Type::kZ2ZForward
                                            : se::fft::Type::kC2CForward)
                           : (is_complex128 ? se::fft::Type::kZ2ZInverse
                                            : se::fft::Type::kC2CInverse));

    CufftScratchAllocator scratch_allocator(kCufftScratchSize, ctx);

    // Plan cache singleton with safe no-destructor initialization.
    static FftPlanCache* plan_cache = new FftPlanCache(kFftPlanCacheCapacity);

    // CUDA cufft plans are device-specific, so grab the GPU device ID.
    int device_id = ctx->device()->tensorflow_accelerator_device_info()->gpu_id;

    // Look for plan in cache.
    FftPlanInfo plan_info =
        FftPlanInfo::Create(fft_rank, fft_shape, input_embed, input_stride,
                            input_distance, output_embed, output_stride,
                            output_distance, kFftType, batch_size, device_id);
    std::unique_ptr<se::fft::Plan> plan = nullptr;
    {
      auto plan_or = plan_cache->Extract(plan_info);
      if (plan_or.has_value()) {
        plan = std::move(*plan_or);
      }
    }
    auto fft = stream->parent()->AsFft();
    OP_REQUIRES(ctx, fft != nullptr, absl::InternalError("No FFT for stream."));
    // Create a new plan if one doesn't exist.  Otherwise, we need only set
    // the scratch allocator.
    if (plan == nullptr) {
      plan = fft->CreateBatchedPlanWithScratchAllocator(
          stream, fft_rank, fft_shape, input_embed, input_stride,
          input_distance, output_embed, output_stride, output_distance,
          kFftType, kInPlaceFft, batch_size, &scratch_allocator);
    } else {
      fft->UpdatePlanWithScratchAllocator(stream, plan.get(),
                                          &scratch_allocator);
    }

    OP_REQUIRES(
        ctx, plan != nullptr,
        absl::InternalError(
            "Failed to create cuFFT batched plan with scratch allocator"));
    if (IsReal()) {
      if (IsForward()) {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_DOUBLE);
          DCHECK_EQ(out->dtype(), DT_COMPLEX128);
          DoFFTInternal<double, complex128>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_FLOAT);
          DCHECK_EQ(out->dtype(), DT_COMPLEX64);
          DoFFTInternal<float, complex64>(ctx, stream, plan.get(), kFftType,
                                          output_distance, in, out);
        }
      } else {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_COMPLEX128);
          DCHECK_EQ(out->dtype(), DT_DOUBLE);
          DoFFTInternal<complex128, double>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_COMPLEX64);
          DCHECK_EQ(out->dtype(), DT_FLOAT);
          DoFFTInternal<complex64, float>(ctx, stream, plan.get(), kFftType,
                                          output_distance, in, out);
        }
      }
    } else {
      if (is_complex128) {
        DCHECK_EQ(in.dtype(), DT_COMPLEX128);
        DCHECK_EQ(out->dtype(), DT_COMPLEX128);
        DoFFTInternal<complex128, complex128>(ctx, stream, plan.get(), kFftType,
                                              output_distance, in, out);
      } else {
        DCHECK_EQ(in.dtype(), DT_COMPLEX64);
        DCHECK_EQ(out->dtype(), DT_COMPLEX64);
        DoFFTInternal<complex64, complex64>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
      }
    }
    plan_cache->Insert(std::move(plan_info), std::move(plan));
  }

 private:
  template <typename T>
  struct RealTypeFromComplexType {
    typedef T RealT;
  };

  template <typename T>
  struct RealTypeFromComplexType<std::complex<T>> {
    typedef T RealT;
  };

  template <typename InT, typename OutT>
  void DoFFTInternal(OpKernelContext* ctx, se::Stream* stream,
                     se::fft::Plan* plan, const se::fft::Type fft_type,
                     const uint64 output_distance, const Tensor& in,
                     Tensor* out) {
    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();
    auto src =
        AsDeviceMemory<InT>(in.flat<InT>().data(), input_shape.num_elements());
    auto dst = AsDeviceMemory<OutT>(out->flat<OutT>().data(),
                                    output_shape.num_elements());
    auto fft = stream->parent()->AsFft();
    OP_REQUIRES(ctx, fft != nullptr, absl::InternalError("No FFT for stream."));
    OP_REQUIRES(ctx, fft->DoFft(stream, plan, src, &dst),
                absl::InternalError(absl::StrCat(
                    "fft failed : type=", static_cast<int>(fft_type),
                    " in.shape=", input_shape.DebugString())));
    if (!IsForward()) {
      typedef typename RealTypeFromComplexType<OutT>::RealT RealT;
      RealT alpha = 1.0 / output_distance;
      auto blas = stream->parent()->AsBlas();
      OP_REQUIRES(ctx, blas != nullptr,
                  absl::InternalError("No blas for stream."));
      OP_REQUIRES(
          ctx,
          blas->DoBlasScal(stream, output_shape.num_elements(), alpha, &dst, 1),
          absl::InternalError(absl::StrCat("BlasScal failed : in.shape=",
                                           input_shape.DebugString())));
    }
  }
};

const int64_t FFTNGPUBase::kCufftScratchSize = GetCufftWorkspaceLimit(
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

template <bool Forward, bool _Real>
class FFTNGPU : public FFTNGPUBase {
 public:
  explicit FFTNGPU(OpKernelConstruction* ctx) : FFTNGPUBase(ctx) {}

 protected:
  bool IsForward() const override { return Forward; }
  bool IsReal() const override { return _Real; }
};

// Register GPU kernels with priority 1 so that if a custom FFT CPU kernel is
// registered with priority 1 (to override the default CPU kernel), the
// CPU kernel does not outrank the GPU kernel.
REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 3>);
REGISTER_KERNEL_BUILDER(Name("FFTND")
                            .Device(DEVICE_GPU)
                            .HostMemory("fft_length")
                            .HostMemory("axes")
                            .Priority(1),
                        FFTNGPU<true, false>);
REGISTER_KERNEL_BUILDER(Name("IFFTND")
                            .Device(DEVICE_GPU)
                            .HostMemory("fft_length")
                            .HostMemory("axes")
                            .Priority(1),
                        FFTNGPU<false, false>);

REGISTER_KERNEL_BUILDER(
    Name("RFFT").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT2D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT2D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT3D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT3D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 3>);
REGISTER_KERNEL_BUILDER(Name("RFFTND")
                            .Device(DEVICE_GPU)
                            .HostMemory("fft_length")
                            .HostMemory("axes")
                            .Priority(1),
                        FFTNGPU<true, true>);
REGISTER_KERNEL_BUILDER(Name("IRFFTND")
                            .Device(DEVICE_GPU)
                            .HostMemory("fft_length")
                            .HostMemory("axes")
                            .Priority(1),
                        FFTNGPU<false, true>);

// Deprecated kernels.
REGISTER_KERNEL_BUILDER(Name("BatchFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 3>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
