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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/work_sharder.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if defined(GOOGLE_CUDA) && GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // CUDA_VERSION
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

    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    if (!IsReal()) {
      // Compute the FFT using Eigen.
      constexpr auto direction =
          Forward ? Eigen::FFT_FORWARD : Eigen::FFT_REVERSE;
      if (is_complex128) {
        DCHECK_EQ(in.dtype(), DT_COMPLEX128);
        DCHECK_EQ(out->dtype(), DT_COMPLEX128);
        auto input = Tensor(in).flat_inner_dims<complex128, FFTRank + 1>();
        auto output = out->flat_inner_dims<complex128, FFTRank + 1>();
        output.device(device) =
            input.template fft<Eigen::BothParts, direction>(axes);
      } else {
        DCHECK_EQ(in.dtype(), DT_COMPLEX64);
        DCHECK_EQ(out->dtype(), DT_COMPLEX64);
        auto input = Tensor(in).flat_inner_dims<complex64, FFTRank + 1>();
        auto output = out->flat_inner_dims<complex64, FFTRank + 1>();
        output.device(device) =
            input.template fft<Eigen::BothParts, direction>(axes);
      }
    } else {
      if (IsForward()) {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_DOUBLE);
          DCHECK_EQ(out->dtype(), DT_COMPLEX128);
          DoRealForwardFFT<double, complex128>(ctx, fft_shape, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_FLOAT);
          DCHECK_EQ(out->dtype(), DT_COMPLEX64);
          DoRealForwardFFT<float, complex64>(ctx, fft_shape, in, out);
        }
      } else {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_COMPLEX128);
          DCHECK_EQ(out->dtype(), DT_DOUBLE);
          DoRealBackwardFFT<complex128, double>(ctx, fft_shape, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_COMPLEX64);
          DCHECK_EQ(out->dtype(), DT_FLOAT);
          DoRealBackwardFFT<complex64, float>(ctx, fft_shape, in, out);
        }
      }
    }
  }

  template <typename RealT, typename ComplexT>
  void DoRealForwardFFT(OpKernelContext* ctx, uint64* fft_shape,
                        const Tensor& in, Tensor* out) {
    // Create the axes (which are always trailing).
    const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
    auto device = ctx->eigen_device<CPUDevice>();
    auto input = Tensor(in).flat_inner_dims<RealT, FFTRank + 1>();
    const auto input_dims = input.dimensions();

    // Slice input to fft_shape on its inner-most dimensions.
    Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> input_slice_sizes;
    input_slice_sizes[0] = input_dims[0];
    TensorShape temp_shape{input_dims[0]};
    for (int i = 1; i <= FFTRank; ++i) {
      input_slice_sizes[i] = fft_shape[i - 1];
      OP_REQUIRES_OK(ctx, temp_shape.AddDimWithStatus(fft_shape[i - 1]));
    }
    OP_REQUIRES(ctx, temp_shape.num_elements() > 0,
                errors::InvalidArgument("Obtained a FFT shape of 0 elements: ",
                                        temp_shape.DebugString()));

    auto output = out->flat_inner_dims<ComplexT, FFTRank + 1>();
    const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;

    // Compute the full FFT using a temporary tensor.
    Tensor temp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<ComplexT>::v(),
                                           temp_shape, &temp));
    auto full_fft = temp.flat_inner_dims<ComplexT, FFTRank + 1>();
    full_fft.device(device) =
        input.slice(zero_start_indices, input_slice_sizes)
            .template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);

    // Slice away the negative frequency components.
    output.device(device) =
        full_fft.slice(zero_start_indices, output.dimensions());
  }

  template <typename ComplexT, typename RealT>
  void DoRealBackwardFFT(OpKernelContext* ctx, uint64* fft_shape,
                         const Tensor& in, Tensor* out) {
    auto device = ctx->eigen_device<CPUDevice>();
    // Reconstruct the full FFT and take the inverse.
    auto input = Tensor(in).flat_inner_dims<ComplexT, FFTRank + 1>();
    auto output = out->flat_inner_dims<RealT, FFTRank + 1>();
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
      OP_REQUIRES_OK(ctx, full_fft_shape.AddDimWithStatus(fft_shape[i - 1]));
    }
    OP_REQUIRES(ctx, full_fft_shape.num_elements() > 0,
                errors::InvalidArgument("Obtained a FFT shape of 0 elements: ",
                                        full_fft_shape.DebugString()));

    Tensor temp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<ComplexT>::v(),
                                           full_fft_shape, &temp));
    auto full_fft = temp.flat_inner_dims<ComplexT, FFTRank + 1>();

    // Calculate the starting point and range of the source of
    // negative frequency part.
    auto neg_sizes = input_slice_sizes;
    neg_sizes[FFTRank] = fft_shape[FFTRank - 1] - input_slice_sizes[FFTRank];
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
              .template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(outer_axes);
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
        full_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(inner_axis);
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

  FftPlanInfo() = default;

  template <typename H>
  friend inline H AbslHashValue(H h, const FftPlanInfo& key) {
    return H::combine(std::move(h), key.rank, key.shape, key.input_embed,
                      key.input_stride, key.input_distance, key.output_embed,
                      key.output_stride, key.output_distance, key.type,
                      key.batch);
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
           lhs.batch == rhs.batch;
  }

  // Create a key to be used for caching plans.
  static FftPlanInfo Create(int rank, const uint64_t* shape,
                            const uint64_t* input_embed, uint64_t input_stride,
                            uint64_t input_distance,
                            const uint64_t* output_embed,
                            uint64_t output_stride, uint64_t output_distance,
                            se::fft::Type type, int batch) {
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

    // Look for plan in cache.
    FftPlanInfo plan_info = FftPlanInfo::Create(
        fft_rank, fft_shape, input_embed, input_stride, input_distance,
        output_embed, output_stride, output_distance, kFftType, batch_size);
    std::unique_ptr<se::fft::Plan> plan = nullptr;
    {
      auto plan_or = plan_cache->Extract(plan_info);
      if (plan_or.has_value()) {
        plan = std::move(*plan_or);
      }
    }

    // Create a new plan if one doesn't exist.  Otherwise, we need only set
    // the scratch allocator.
    if (plan == nullptr) {
      plan = stream->parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
          stream, fft_rank, fft_shape, input_embed, input_stride,
          input_distance, output_embed, output_stride, output_distance,
          kFftType, kInPlaceFft, batch_size, &scratch_allocator);
    } else {
      stream->parent()->AsFft()->UpdatePlanWithScratchAllocator(
          stream, plan.get(), &scratch_allocator);
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
    OP_REQUIRES(
        ctx, stream->ThenFft(plan, src, &dst).ok(),
        errors::Internal("fft failed : type=", static_cast<int>(fft_type),
                         " in.shape=", input_shape.DebugString()));
    if (!IsForward()) {
      typedef typename RealTypeFromComplexType<OutT>::RealT RealT;
      RealT alpha = 1.0 / output_distance;
      OP_REQUIRES(
          ctx,
          stream->ThenBlasScal(output_shape.num_elements(), alpha, &dst, 1)
              .ok(),
          errors::Internal("BlasScal failed : in.shape=",
                           input_shape.DebugString()));
    }
  }
};

const int64_t FFTGPUBase::kCufftScratchSize = GetCufftWorkspaceLimit(
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

// Register GPU kernels with priority 1 so that if a custom FFT CPU kernel is
// registered with priority 1 (to override the default Eigen CPU kernel), the
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

}  // end namespace tensorflow
