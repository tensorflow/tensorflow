/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_fft.h"

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "oneapi/mkl/dft.hpp"
#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/onemkl_util.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace sycl {

using gpu::GpuMemory;
using gpu::GpuMemoryMutable;

namespace dft = oneapi::mkl::dft;

namespace {

template <dft::precision P, dft::domain D>
using Descriptor = dft::descriptor<P, D>;

// The real scalar type oneMKL uses for a given descriptor precision.
template <dft::precision P>
using RealScalar =
    std::conditional_t<P == dft::precision::DOUBLE, double, float>;

// Maps the element types of a DoFft() call onto the oneMKL descriptor that has
// to back the plan.
template <typename InputT, typename OutputT>
struct OneMklFftTraits;

#define STREAM_EXECUTOR_SYCL_FFT_TRAITS(__input, __output, __precision,       \
                                        __domain)                             \
  template <>                                                                 \
  struct OneMklFftTraits<__input, __output> {                                 \
    static constexpr dft::precision kPrecision = dft::precision::__precision; \
    static constexpr dft::domain kDomain = dft::domain::__domain;             \
  }

// input type, output type, precision, domain:
STREAM_EXECUTOR_SYCL_FFT_TRAITS(std::complex<float>, std::complex<float>,
                                SINGLE, COMPLEX);
STREAM_EXECUTOR_SYCL_FFT_TRAITS(std::complex<double>, std::complex<double>,
                                DOUBLE, COMPLEX);
STREAM_EXECUTOR_SYCL_FFT_TRAITS(float, std::complex<float>, SINGLE, REAL);
STREAM_EXECUTOR_SYCL_FFT_TRAITS(double, std::complex<double>, DOUBLE, REAL);

#undef STREAM_EXECUTOR_SYCL_FFT_TRAITS

// Returns true for the transforms oneMKL computes with compute_forward().
bool IsForward(fft::Type type) {
  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kZ2ZForward:
    case fft::Type::kR2C:
    case fft::Type::kD2Z:
      return true;
    default:
      return false;
  }
}

// Returns true if the transform's backward domain holds the conjugate-even
// (i.e. only n/2+1 elements along the last axis) representation of the signal.
bool IsRealTransform(fft::Type type) {
  return type == fft::Type::kR2C || type == fft::Type::kD2Z;
}

// Returns the layout of one of the transform's domains, expressed as the oneMKL
// stride vector {offset, stride_of_dim_0, ..., stride_of_dim_rank-1}.
//
// `embed` is the plan's `input_embed` or `output_embed`: the storage dimensions
// of the data, which may exceed the transform itself when the signal is padded.
// `elem_stride` is the distance between two successive elements along the last
// axis. If `embed` is null the data is assumed to be densely packed, in which
// case the dimensions are `fft_length`, except that the last axis of a
// conjugate-even domain only holds n/2+1 elements.
std::vector<std::int64_t> DomainStrides(int rank, const std::uint64_t* embed,
                                        std::uint64_t elem_stride,
                                        const std::uint64_t* fft_length,
                                        bool conjugate_even) {
  std::vector<std::int64_t> dims(rank);
  for (int i = 0; i < rank; ++i) {
    dims[i] = embed != nullptr ? embed[i] : fft_length[i];
  }
  if (embed == nullptr && conjugate_even) {
    dims[rank - 1] = dims[rank - 1] / 2 + 1;
  }

  // Row-major strides: the last axis is the fastest varying one. The dimensions
  // arrive in major-to-minor order from the FFT thunk.
  std::vector<std::int64_t> strides(rank + 1);
  // Zero offset: the data begins at the input or output pointer.
  strides[0] = 0;
  std::int64_t stride = elem_stride;
  for (int i = rank; i >= 1; --i) {
    strides[i] = stride;
    stride *= dims[i - 1];
  }
  return strides;
}

// Returns the SYCL queue backing `stream`.
::sycl::queue* SyclQueue(Stream* stream) {
  return absl::bit_cast<::sycl::queue*>(
      stream->platform_specific_handle().stream);
}

// An FFT plan holding a committed oneMKL DFT descriptor. The descriptor type
// depends on the precision and the domain of the transform, so the plan keeps
// it in a variant and hands it back to the (templated) execute path.
//
// Like the other fft::Plan implementations, a plan is not internally
// synchronized: callers serialize access to it. FftThunk does so by holding
// FftPlan::mu across both the creation and the execution of a plan.
class SyclFftPlan : public fft::Plan {
 public:
  absl::Status Initialize(Stream* stream, int rank, std::uint64_t* fft_length,
                          std::uint64_t* input_embed,
                          std::uint64_t input_stride,
                          std::uint64_t input_distance,
                          std::uint64_t* output_embed,
                          std::uint64_t output_stride,
                          std::uint64_t output_distance, fft::Type type,
                          bool in_place_fft, int batch_count);

  // Returns the descriptor of the plan, or null if the plan was created for a
  // different precision or domain.
  template <dft::precision P, dft::domain D>
  Descriptor<P, D>* GetDescriptor() {
    auto* holder = std::get_if<std::unique_ptr<Descriptor<P, D>>>(&descriptor_);
    return holder == nullptr ? nullptr : holder->get();
  }

  // Commits this plan's descriptor to `queue`, unless it is already committed
  // there.
  //
  // oneMKL needs a descriptor committed after configuration and before any
  // transform. commit() validates the configuration, factorizes the lengths to
  // pick a computation method and allocates the internal work area, throwing if
  // it cannot; ExecMklFunc() turns that into a Status.
  //
  // It also binds the descriptor to `queue`: compute_forward() and
  // compute_backward() take no queue and enqueue on the last-committed one.
  // Since FftPlanCache caches a plan per device ordinal rather than per stream,
  // a plan reused on another stream must be re-committed to run there. Commit
  // is the expensive planning step, so an already-committed queue is skipped --
  // safe because each plan owns exactly one descriptor.
  absl::Status CommitOn(::sycl::queue* queue);

 private:
  // Sets up the descriptor for the given precision and domain: applies the
  // configuration below to a fresh descriptor, stores it in `descriptor_` along
  // with the `commit_` that goes with it, and commits it to the queue of
  // `stream`. `lengths` are the logical transform dimensions, the strides and
  // distances describe the forward and the backward domain respectively.
  template <dft::precision P, dft::domain D>
  absl::Status ConfigureDescriptor(Stream* stream,
                                   std::vector<std::int64_t> lengths,
                                   const std::vector<std::int64_t>& fwd_strides,
                                   const std::vector<std::int64_t>& bwd_strides,
                                   std::int64_t fwd_distance,
                                   std::int64_t bwd_distance, int batch_count);

  std::variant<
      std::monostate,
      std::unique_ptr<Descriptor<dft::precision::SINGLE, dft::domain::COMPLEX>>,
      std::unique_ptr<Descriptor<dft::precision::SINGLE, dft::domain::REAL>>,
      std::unique_ptr<Descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX>>,
      std::unique_ptr<Descriptor<dft::precision::DOUBLE, dft::domain::REAL>>>
      descriptor_;

  // Commits the type-erased descriptor above to a queue.
  std::function<void(::sycl::queue&)> commit_;

  // Queue the descriptor above is currently committed to. Only ever compared
  // against the descriptor it belongs to; it says nothing about the descriptors
  // of other plans that may run on the same queue.
  ::sycl::queue* committed_queue_ = nullptr;
};

template <dft::precision P, dft::domain D>
absl::Status SyclFftPlan::ConfigureDescriptor(
    Stream* stream, std::vector<std::int64_t> lengths,
    const std::vector<std::int64_t>& fwd_strides,
    const std::vector<std::int64_t>& bwd_strides, std::int64_t fwd_distance,
    std::int64_t bwd_distance, int batch_count) {
  auto descriptor = std::make_unique<Descriptor<P, D>>(std::move(lengths));
  // set_value() reports a rejected parameter by throwing, and ExecMklFunc()
  // turns that into a Status. Setting everything inside one lambda therefore
  // needs a single try/catch instead of one per parameter: the first rejected
  // parameter skips the rest, and its message is returned below.
  absl::Status status = ExecMklFunc([&] {
    Descriptor<P, D>& desc = *descriptor;
    desc.set_value(dft::config_param::NUMBER_OF_TRANSFORMS,
                   static_cast<std::int64_t>(batch_count));
    desc.set_value(dft::config_param::PLACEMENT,
                   dft::config_value::NOT_INPLACE);
    if constexpr (D == dft::domain::COMPLEX) {
      desc.set_value(dft::config_param::COMPLEX_STORAGE,
                     dft::config_value::COMPLEX_COMPLEX);
    }
    // The backward domain of a real transform holds the n/2+1 complex
    // elements of the conjugate-even spectrum interleaved, which is the
    // layout XLA expects. That is the only storage oneMKL
    // supports
    desc.set_value(dft::config_param::FWD_STRIDES, fwd_strides);
    desc.set_value(dft::config_param::BWD_STRIDES, bwd_strides);
    desc.set_value(dft::config_param::FWD_DISTANCE, fwd_distance);
    desc.set_value(dft::config_param::BWD_DISTANCE, bwd_distance);
    // XLA expects unnormalized transforms.
    desc.set_value(dft::config_param::FORWARD_SCALE, RealScalar<P>{1});
  });
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to configure the oneMKL DFT descriptor: ", status.message()));
  }

  Descriptor<P, D>* descriptor_ptr = descriptor.get();
  commit_ = [descriptor_ptr](::sycl::queue& queue) {
    descriptor_ptr->commit(queue);
  };
  descriptor_ = std::move(descriptor);
  return CommitOn(SyclQueue(stream));
}

absl::Status SyclFftPlan::CommitOn(::sycl::queue* queue) {
  // `committed_queue_` is per-plan state, not shared: it records the queue that
  // *this* plan's one descriptor is committed on. A different plan on the same
  // queue has its own, still null, so it commits its own descriptor rather than
  // skipping here.
  if (queue == committed_queue_) {
    return absl::OkStatus();
  }
  absl::Status status = ExecMklFunc([&] { commit_(*queue); });
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to commit the oneMKL DFT descriptor: ", status.message()));
  }
  committed_queue_ = queue;
  return absl::OkStatus();
}

absl::Status SyclFftPlan::Initialize(
    Stream* stream, int rank, std::uint64_t* fft_length,
    std::uint64_t* input_embed, std::uint64_t input_stride,
    std::uint64_t input_distance, std::uint64_t* output_embed,
    std::uint64_t output_stride, std::uint64_t output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
  if (rank < 1 || rank > 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("oneMKL DFT only supports rank 1, 2 or 3, given: ", rank));
  }
  if (in_place_fft) {
    return absl::UnimplementedError(
        "In-place FFT is not implemented for the SYCL platform.");
  }
  if (!std::holds_alternative<std::monostate>(descriptor_)) {
    return absl::InternalError("oneMKL DFT plan should be uninitialized.");
  }
  if (!IsForward(type)) {
    return absl::UnimplementedError(
        "Inverse FFT is not implemented for the SYCL platform.");
  }
  const bool conjugate_even = IsRealTransform(type);

  std::vector<std::int64_t> lengths(rank);
  for (int i = 0; i < rank; ++i) {
    lengths[i] = static_cast<std::int64_t>(fft_length[i]);
  }

  // The forward domain holds the (real or complex) signal, which is the input,
  // and the backward domain its spectrum, which is the output.
  const std::int64_t fwd_distance = static_cast<std::int64_t>(input_distance);
  const std::int64_t bwd_distance = static_cast<std::int64_t>(output_distance);
  const std::vector<std::int64_t> fwd_strides = DomainStrides(
      rank, input_embed, input_stride, fft_length, /*conjugate_even=*/false);
  const std::vector<std::int64_t> bwd_strides =
      DomainStrides(rank, output_embed, output_stride, fft_length,
                    /*conjugate_even=*/conjugate_even);

  switch (type) {
    case fft::Type::kC2CForward:
      return ConfigureDescriptor<dft::precision::SINGLE, dft::domain::COMPLEX>(
          stream, std::move(lengths), fwd_strides, bwd_strides, fwd_distance,
          bwd_distance, batch_count);
    case fft::Type::kZ2ZForward:
      return ConfigureDescriptor<dft::precision::DOUBLE, dft::domain::COMPLEX>(
          stream, std::move(lengths), fwd_strides, bwd_strides, fwd_distance,
          bwd_distance, batch_count);
    case fft::Type::kR2C:
      return ConfigureDescriptor<dft::precision::SINGLE, dft::domain::REAL>(
          stream, std::move(lengths), fwd_strides, bwd_strides, fwd_distance,
          bwd_distance, batch_count);
    case fft::Type::kD2Z:
      return ConfigureDescriptor<dft::precision::DOUBLE, dft::domain::REAL>(
          stream, std::move(lengths), fwd_strides, bwd_strides, fwd_distance,
          bwd_distance, batch_count);
    default:
      return absl::InvalidArgumentError("Invalid value of fft::Type.");
  }
}

}  // namespace

std::unique_ptr<fft::Plan> SyclFft::CreateBatchedPlanWithScratchAllocator(
    Stream* stream, int rank, std::uint64_t* elem_count,
    std::uint64_t* input_embed, std::uint64_t input_stride,
    std::uint64_t input_distance, std::uint64_t* output_embed,
    std::uint64_t output_stride, std::uint64_t output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator* scratch_allocator) {
  auto fft_plan = std::make_unique<SyclFftPlan>();
  // oneMKL manages the work area of a descriptor itself, so the scratch
  // allocator is not used.
  absl::Status status =
      fft_plan->Initialize(stream, rank, elem_count, input_embed, input_stride,
                           input_distance, output_embed, output_stride,
                           output_distance, type, in_place_fft, batch_count);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize batched oneMKL DFT plan: "
               << status.message() << ". Initialize params: rank: " << rank
               << " elem_count: " << *elem_count
               << " input_stride: " << input_stride
               << " input_distance: " << input_distance
               << " output_stride: " << output_stride
               << " output_distance: " << output_distance
               << " batch_count: " << batch_count;
    return nullptr;
  }
  return fft_plan;
}

void SyclFft::UpdatePlanWithScratchAllocator(
    Stream* stream, fft::Plan* plan, ScratchAllocator* scratch_allocator) {
  // Currently, oneMKL allocates the workspace of a descriptor itself.
  // TODO(intel-tf): Implement and evaluate allocation with scratch_allocator.
}

template <typename InputT, typename OutputT>
bool SyclFft::DoFftInternal(Stream* stream, fft::Plan* plan,
                            const DeviceAddress<InputT>& input,
                            DeviceAddress<OutputT>* output) {
  SyclFftPlan* sycl_fft_plan = dynamic_cast<SyclFftPlan*>(plan);
  if (sycl_fft_plan == nullptr) {
    LOG(ERROR) << "The passed-in plan is not a SyclFftPlan object.";
    return false;
  }

  using Traits = OneMklFftTraits<InputT, OutputT>;
  auto* descriptor =
      sycl_fft_plan->GetDescriptor<Traits::kPrecision, Traits::kDomain>();
  if (descriptor == nullptr) {
    LOG(ERROR) << "The oneMKL DFT plan was created for a different precision "
                  "or domain than the one the transform is run with.";
    return false;
  }

  if (absl::Status status = sycl_fft_plan->CommitOn(SyclQueue(stream));
      !status.ok()) {
    LOG(ERROR) << status.message();
    return false;
  }

  // The queues created by SyclStream are in-order, so the transform is ordered
  // against the rest of the stream without tracking events explicitly.
  auto* in = const_cast<InputT*>(GpuMemory(input));
  auto* out = GpuMemoryMutable(output);
  absl::StatusOr<::sycl::event> event =
      ExecMklFunc([&] { return dft::compute_forward(*descriptor, in, out); });
  if (!event.ok()) {
    LOG(ERROR) << "Failed to run the oneMKL DFT routine: "
               << event.status().message();
    return false;
  }
  return true;
}

#define STREAM_EXECUTOR_SYCL_DEFINE_FFT(__type)                         \
  bool SyclFft::DoFft(Stream* stream, fft::Plan* plan,                  \
                      const DeviceAddress<std::complex<__type>>& input, \
                      DeviceAddress<std::complex<__type>>* output) {    \
    return DoFftInternal(stream, plan, input, output);                  \
  }                                                                     \
  bool SyclFft::DoFft(Stream* stream, fft::Plan* plan,                  \
                      const DeviceAddress<__type>& input,               \
                      DeviceAddress<std::complex<__type>>* output) {    \
    return DoFftInternal(stream, plan, input, output);                  \
  }                                                                     \
  bool SyclFft::DoFft(Stream* stream, fft::Plan* plan,                  \
                      const DeviceAddress<std::complex<__type>>& input, \
                      DeviceAddress<__type>* output) {                  \
    LOG(ERROR) << "Inverse real FFT is not implemented for the SYCL "   \
                  "platform.";                                          \
    return false;                                                       \
  }

STREAM_EXECUTOR_SYCL_DEFINE_FFT(float)
STREAM_EXECUTOR_SYCL_DEFINE_FFT(double)

#undef STREAM_EXECUTOR_SYCL_DEFINE_FFT

void initialize_onemkl_fft() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
          sycl::kSyclPlatformId, "oneMKL DFT",
          [](StreamExecutor* parent) -> fft::FftSupport* {
            return new sycl::SyclFft(parent);
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register oneMKL DFT factory: " << status.message();
  }
}

}  // namespace sycl
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_onemkl_fft, {
  stream_executor::sycl::initialize_onemkl_fft();
});
