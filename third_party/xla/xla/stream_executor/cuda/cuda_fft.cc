/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_fft.h"

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cufft.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_helpers.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

using cuda::CUDAComplex;

namespace {

// A helper function transforming gpu_fft arguments into cuFFT arguments.
cufftType CUDAFftType(fft::Type type) {
  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kC2CInverse:
      return CUFFT_C2C;
    case fft::Type::kC2R:
      return CUFFT_C2R;
    case fft::Type::kR2C:
      return CUFFT_R2C;
    case fft::Type::kZ2ZForward:
    case fft::Type::kZ2ZInverse:
      return CUFFT_Z2Z;
    case fft::Type::kZ2D:
      return CUFFT_Z2D;
    case fft::Type::kD2Z:
      return CUFFT_D2Z;
    default:
      LOG(FATAL) << "Invalid value of fft::Type.";
  }
}

// Associates the given stream with the given cuFFT plan.
bool SetStream(StreamExecutor *parent, cufftHandle plan, Stream *stream) {
  std::unique_ptr<ActivateContext> activation = parent->Activate();
  auto ret = cufftSetStream(plan, AsGpuStreamValue(stream));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "Failed to run cuFFT routine cufftSetStream: " << ret;
    return false;
  }
  return true;
}

// Populates array of 32b integers from 64b integers, or an error if the
// numbers don't fit in 32b (signed).
absl::StatusOr<std::array<int32_t, 3>> Downsize64bArray(
    std::array<long long, 3> source, int32_t rank) {  // NOLINT
  std::array<int32_t, 3> downsized = {0};
  for (int32_t i = 0; i < rank; ++i) {
    if (source[i] > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(absl::StrCat(
          source[i], " exceeds max 32b signed integer. Conversion failed."));
    }
    downsized[i] = static_cast<int32_t>(source[i]);
  }
  return downsized;
}

}  // namespace

absl::Status CUDAFftPlan::Initialize(
    StreamExecutor *parent, Stream *stream, int rank, uint64_t *elem_count,
    uint64_t *input_embed, uint64_t input_stride, uint64_t input_distance,
    uint64_t *output_embed, uint64_t output_stride, uint64_t output_distance,
    fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
  if (IsInitialized()) {
    return absl::InternalError("cuFFT is already initialized.");
  }
  is_initialized_ = true;
  scratch_allocator_ = scratch_allocator;
  std::unique_ptr<ActivateContext> activation = parent->Activate();
  // NOLINTBEGIN
  std::array<long long, 3> elem_count_ = {0};
  std::array<long long, 3> input_embed_ = {0};
  std::array<long long, 3> output_embed_ = {0};
  // NOLINTEND
  for (int32_t i = 0; i < rank; ++i) {
    elem_count_[i] = elem_count[i];
    if (input_embed) {
      input_embed_[i] = input_embed[i];
    }
    if (output_embed) {
      output_embed_[i] = output_embed[i];
    }
  }
  parent_ = parent;
  fft_type_ = type;
  if (batch_count == 1 && input_embed == nullptr && output_embed == nullptr) {
    cufftResult_t ret;
    if (scratch_allocator == nullptr) {
      switch (rank) {
        case 1:
          // cufftPlan1d
          ret = cufftPlan1d(&plan_, elem_count_[0], CUDAFftType(type),
                            1 /* = batch */);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to create cuFFT 1d plan: " << ret;
            return absl::InternalError("Failed to create cuFFT 1d plan.");
          }
          return absl::OkStatus();
        case 2:
          // cufftPlan2d
          ret = cufftPlan2d(&plan_, elem_count_[0], elem_count_[1],
                            CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to create cuFFT 2d plan: " << ret;
            return absl::InternalError("Failed to create cuFFT 2d plan.");
          }
          return absl::OkStatus();
        case 3:
          // cufftPlan3d
          ret = cufftPlan3d(&plan_, elem_count_[0], elem_count_[1],
                            elem_count_[2], CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to create cuFFT 3d plan: " << ret;
            return absl::InternalError("Failed to create cuFFT 3d plan.");
          }
          return absl::OkStatus();
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return absl::InvalidArgumentError(
              "cufftPlan only takes rank 1, 2, or 3.");
      }
    } else {
      ret = cufftCreate(&plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT plan: " << ret;
        return absl::InternalError("Failed to create cuFFT plan.");
      }
      ret = cufftSetAutoAllocation(plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to set auto allocation for cuFFT plan: " << ret;
        return absl::InternalError(
            "Failed to set auto allocation for cuFFT plan.");
      }
      switch (rank) {
        case 1:
          ret = cufftMakePlan1d(plan_, elem_count_[0], CUDAFftType(type),
                                /*batch=*/1, &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to make cuFFT 1d plan: " << ret;
            return absl::InternalError("Failed to make cuFFT 1d plan.");
          }
          break;
        case 2:
          ret = cufftMakePlan2d(plan_, elem_count_[0], elem_count_[1],
                                CUDAFftType(type), &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to make cuFFT 2d plan: " << ret;
            return absl::InternalError("Failed to make cuFFT 2d plan.");
          }
          break;
        case 3:
          ret = cufftMakePlan3d(plan_, elem_count_[0], elem_count_[1],
                                elem_count_[2], CUDAFftType(type),
                                &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to make cuFFT 3d plan: " << ret;
            return absl::InternalError("Failed to make cuFFT 3d plan.");
          }
          break;
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return absl::InvalidArgumentError(
              "cufftPlan only takes rank 1, 2, or 3.");
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  } else {
    // For either multiple batches or rank higher than 3, use cufft*PlanMany*().
    if (scratch_allocator == nullptr) {
      // Downsize 64b arrays to 32b as there's no 64b version of cufftPlanMany
      TF_ASSIGN_OR_RETURN(auto elem_count_32b_,
                          Downsize64bArray(elem_count_, rank));
      TF_ASSIGN_OR_RETURN(auto input_embed_32b_,
                          Downsize64bArray(input_embed_, rank));
      TF_ASSIGN_OR_RETURN(auto output_embed_32b_,
                          Downsize64bArray(output_embed_, rank));
      auto ret = cufftPlanMany(
          &plan_, rank, elem_count_32b_.data(),
          input_embed ? input_embed_32b_.data() : nullptr, input_stride,
          input_distance, output_embed ? output_embed_32b_.data() : nullptr,
          output_stride, output_distance, CUDAFftType(type), batch_count);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT batched plan: " << ret;
        return absl::InternalError("Failed to create cuFFT batched plan.");
      }
    } else {
      auto ret = cufftCreate(&plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT batched plan: " << ret;
        return absl::InternalError("Failed to create cuFFT batched plan.");
      }
      ret = cufftSetAutoAllocation(plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to set auto allocation for cuFFT batched plan: "
                   << ret;
        return absl::InternalError(
            "Failed to set auto allocation for cuFFT batched plan.");
      }
      ret = cufftMakePlanMany64(
          plan_, rank, elem_count_.data(),
          input_embed ? input_embed_.data() : nullptr, input_stride,
          input_distance, output_embed ? output_embed_.data() : nullptr,
          output_stride, output_distance, CUDAFftType(type), batch_count,
          &scratch_size_bytes_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to make cuFFT batched plan: " << ret;
        return absl::InternalError("Failed to make cuFFT batched plan.");
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  }
  return absl::OkStatus();
}

absl::Status CUDAFftPlan::UpdateScratchAllocator(
    Stream *stream, ScratchAllocator *scratch_allocator) {
  scratch_allocator_ = scratch_allocator;

  if (scratch_size_bytes_ != 0) {
    auto allocated = scratch_allocator->AllocateBytes(scratch_size_bytes_);
    if (!allocated.ok() || (scratch_ = allocated.value()) == nullptr) {
      LOG(ERROR) << "Failed to allocate work area.";
      return allocated.status();
    }
  }
  // Connect work area with allocated space.
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  cufftResult_t ret = cufftSetWorkArea(plan_, scratch_.opaque());
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "Failed to set work area for cuFFT plan: " << ret;
    return absl::InternalError("Failed to set work area for cuFFT plan.");
  }
  return absl::OkStatus();
}

CUDAFftPlan::~CUDAFftPlan() {
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  cufftDestroy(plan_);
}

int CUDAFftPlan::GetFftDirection() const {
  if (!IsInitialized()) {
    LOG(FATAL) << "Try to get fft direction before initialization.";
  } else {
    switch (fft_type_) {
      case fft::Type::kC2CForward:
      case fft::Type::kZ2ZForward:
      case fft::Type::kR2C:
      case fft::Type::kD2Z:
        return CUFFT_FORWARD;
      case fft::Type::kC2CInverse:
      case fft::Type::kZ2ZInverse:
      case fft::Type::kC2R:
      case fft::Type::kZ2D:
        return CUFFT_INVERSE;
      default:
        LOG(FATAL) << "Invalid value of fft::Type.";
    }
  }
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64_t *elem_count, uint64_t *input_embed,
    uint64_t input_stride, uint64_t input_distance, uint64_t *output_embed,
    uint64_t output_stride, uint64_t output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  absl::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Initialize Params: rank: " << rank
               << " elem_count: " << *elem_count
               << " input_embed: " << *input_embed
               << " input_stride: " << input_stride
               << " input_distance: " << input_distance
               << " output_embed: " << *output_embed
               << " output_stride: " << output_stride
               << " output_distance: " << output_distance
               << " batch_count: " << batch_count;
    LOG(ERROR)
        << "Failed to initialize batched cufft plan with customized allocator: "
        << status.message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

void CUDAFft::UpdatePlanWithScratchAllocator(
    Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);
  absl::Status status =
      cuda_fft_plan->UpdateScratchAllocator(stream, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to update custom allocator for cufft plan: "
               << status.message();
  }
}

template <typename FuncT, typename InputT, typename OutputT>
bool CUDAFft::DoFftInternal(Stream *stream, fft::Plan *plan, FuncT cufftExec,
                            const DeviceMemory<InputT> &input,
                            DeviceMemory<OutputT> *output) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);

  DeviceMemory<InputT> input_maybe_copy = input;

  if (cuda_fft_plan == nullptr) {
    LOG(ERROR) << "The passed-in plan is not a CUDAFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, cuda_fft_plan->GetPlan(), stream)) {
    return false;
  }

#if CUDA_VERSION >= 10010
  // Workaround a cuFFT bug, which mutates the input buffer when it shouldn't.
  // See b/155276727 and go/nvbugs/2959622.
  // TODO(b/155276727): refine the bounding condition.
  if (input.opaque() != output->opaque() &&
      (std::is_same<InputT, std::complex<float>>::value ||
       std::is_same<InputT, std::complex<double>>::value) &&
      (std::is_same<OutputT, float>::value ||
       std::is_same<OutputT, double>::value) &&
      input.size() > 0) {
    auto *allocator = cuda_fft_plan->GetScratchAllocator();
    if (allocator) {
      auto allocated = allocator->AllocateBytes(input.size());
      if (allocated.ok()) {
        if (stream->Memcpy(&allocated.value(), input, input.size()).ok()) {
          input_maybe_copy = DeviceMemory<InputT>(allocated.value());
        }
      }
      // Keep going even the workaround fails, since we don't have a good
      // bounding box. We don't want to give up on a potentially correct
      // execution just because the allocation for the incorrect case fails.
    }
  }
#endif

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  auto ret =
      cufftExec(cuda_fft_plan->GetPlan(),
                CUDAComplex(const_cast<InputT *>(GpuMemory(input_maybe_copy))),
                CUDAComplex(GpuMemoryMutable(output)));

  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "Failed to run cuFFT routine: " << ret;
    return false;
  }

  return true;
}

template <typename FuncT, typename InputT, typename OutputT>
bool CUDAFft::DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                         FuncT cufftExec,
                                         const DeviceMemory<InputT> &input,
                                         DeviceMemory<OutputT> *output) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);
  if (cuda_fft_plan == nullptr) {
    LOG(ERROR) << "The passed-in plan is not a CUDAFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, cuda_fft_plan->GetPlan(), stream)) {
    return false;
  }

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  auto ret = cufftExec(cuda_fft_plan->GetPlan(),
                       CUDAComplex(const_cast<InputT *>(GpuMemory(input))),
                       CUDAComplex(GpuMemoryMutable(output)),
                       cuda_fft_plan->GetFftDirection());

  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "Failed to run cuFFT routine: " << ret;
    return false;
  }

  return true;
}

#define STREAM_EXECUTOR_CUDA_DEFINE_FFT(__type, __fft_type1, __fft_type2,      \
                                        __fft_type3)                           \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<std::complex<__type>> &input,         \
                      DeviceMemory<std::complex<__type>> *output) {            \
    return DoFftWithDirectionInternal(stream, plan, cufftExec##__fft_type1,    \
                                      input, output);                          \
  }                                                                            \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<__type> &input,                       \
                      DeviceMemory<std::complex<__type>> *output) {            \
    return DoFftInternal(stream, plan, cufftExec##__fft_type2, input, output); \
  }                                                                            \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<std::complex<__type>> &input,         \
                      DeviceMemory<__type> *output) {                          \
    return DoFftInternal(stream, plan, cufftExec##__fft_type3, input, output); \
  }

STREAM_EXECUTOR_CUDA_DEFINE_FFT(float, C2C, R2C, C2R)
STREAM_EXECUTOR_CUDA_DEFINE_FFT(double, Z2Z, D2Z, Z2D)

#undef STREAM_EXECUTOR_CUDA_DEFINE_FFT

}  // namespace gpu

void initialize_cufft() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
          cuda::kCudaPlatformId, "cuFFT",
          [](StreamExecutor *parent) -> fft::FftSupport * {
            return new gpu::CUDAFft(parent);
          });
  if (!status.ok()) {
    LOG(INFO) << "Unable to register cuFFT factory: " << status.message();
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_cufft, {
  stream_executor::initialize_cufft();
});
