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

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.h"

#include <array>
#include <complex>
#include <limits>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/platform/initialize.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuFftPlugin);

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
bool SetStream(GpuExecutor *parent, cufftHandle plan, Stream *stream) {
  cuda::ScopedActivateExecutorContext sac(parent);
  auto ret = cufftSetStream(plan, AsGpuStreamValue(stream));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "Failed to run cuFFT routine cufftSetStream: " << ret;
    return false;
  }
  return true;
}

// Populates array of 32b integers from 64b integers, or an error if the
// numbers don't fit in 32b (signed).
tsl::StatusOr<std::array<int32_t, 3>> Downsize64bArray(
    std::array<long long, 3> source, int32_t rank) {  // NOLINT
  std::array<int32_t, 3> downsized = {0};
  for (int32_t i = 0; i < rank; ++i) {
    if (source[i] > std::numeric_limits<int32_t>::max()) {
      return tsl::errors::InvalidArgument(
          source[i], " exceeds max 32b signed integer. Conversion failed.");
    }
    downsized[i] = static_cast<int32_t>(source[i]);
  }
  return downsized;
}

}  // namespace

tsl::Status CUDAFftPlan::Initialize(
    GpuExecutor *parent, Stream *stream, int rank, uint64_t *elem_count,
    uint64_t *input_embed, uint64 input_stride, uint64 input_distance,
    uint64_t *output_embed, uint64 output_stride, uint64 output_distance,
    fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
  if (IsInitialized()) {
    return tsl::errors::Internal("cuFFT is already initialized.");
  }
  is_initialized_ = true;
  scratch_allocator_ = scratch_allocator;
  cuda::ScopedActivateExecutorContext sac(parent);
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
            return tsl::errors::Internal("Failed to create cuFFT 1d plan.");
          }
          return ::tsl::OkStatus();
        case 2:
          // cufftPlan2d
          ret = cufftPlan2d(&plan_, elem_count_[0], elem_count_[1],
                            CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to create cuFFT 2d plan: " << ret;
            return tsl::errors::Internal("Failed to create cuFFT 2d plan.");
          }
          return ::tsl::OkStatus();
        case 3:
          // cufftPlan3d
          ret = cufftPlan3d(&plan_, elem_count_[0], elem_count_[1],
                            elem_count_[2], CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to create cuFFT 3d plan: " << ret;
            return tsl::errors::Internal("Failed to create cuFFT 3d plan.");
          }
          return ::tsl::OkStatus();
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return tsl::errors::InvalidArgument(
              "cufftPlan only takes rank 1, 2, or 3.");
      }
    } else {
      ret = cufftCreate(&plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT plan: " << ret;
        return tsl::errors::Internal("Failed to create cuFFT plan.");
      }
      ret = cufftSetAutoAllocation(plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to set auto allocation for cuFFT plan: " << ret;
        return tsl::errors::Internal(
            "Failed to set auto allocation for cuFFT plan.");
      }
      switch (rank) {
        case 1:
          ret = cufftMakePlan1d(plan_, elem_count_[0], CUDAFftType(type),
                                /*batch=*/1, &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to make cuFFT 1d plan: " << ret;
            return tsl::errors::Internal("Failed to make cuFFT 1d plan.");
          }
          break;
        case 2:
          ret = cufftMakePlan2d(plan_, elem_count_[0], elem_count_[1],
                                CUDAFftType(type), &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to make cuFFT 2d plan: " << ret;
            return tsl::errors::Internal("Failed to make cuFFT 2d plan.");
          }
          break;
        case 3:
          ret = cufftMakePlan3d(plan_, elem_count_[0], elem_count_[1],
                                elem_count_[2], CUDAFftType(type),
                                &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "Failed to make cuFFT 3d plan: " << ret;
            return tsl::errors::Internal("Failed to make cuFFT 3d plan.");
          }
          break;
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return tsl::errors::InvalidArgument(
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
        return tsl::errors::Internal("Failed to create cuFFT batched plan.");
      }
    } else {
      auto ret = cufftCreate(&plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to create cuFFT batched plan: " << ret;
        return tsl::errors::Internal("Failed to create cuFFT batched plan.");
      }
      ret = cufftSetAutoAllocation(plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "Failed to set auto allocation for cuFFT batched plan: "
                   << ret;
        return tsl::errors::Internal(
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
        return tsl::errors::Internal("Failed to make cuFFT batched plan.");
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  }
  return ::tsl::OkStatus();
}

tsl::Status CUDAFftPlan::Initialize(GpuExecutor *parent, Stream *stream,
                                    int rank, uint64_t *elem_count,
                                    fft::Type type,
                                    ScratchAllocator *scratch_allocator) {
  return Initialize(parent_, stream, rank, elem_count,
                    /*input_embed=*/nullptr, /*input_stride=*/0,
                    /*input_distance=*/0,
                    /*output_embed=*/nullptr, /*output_stride=*/0,
                    /*output_distance=*/0, type, 1, scratch_allocator);
}

tsl::Status CUDAFftPlan::UpdateScratchAllocator(
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
  cuda::ScopedActivateExecutorContext sac(parent_);
  cufftResult_t ret = cufftSetWorkArea(plan_, scratch_.opaque());
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "Failed to set work area for cuFFT plan: " << ret;
    return tsl::errors::Internal("Failed to set work area for cuFFT plan.");
  }
  return ::tsl::OkStatus();
}

CUDAFftPlan::~CUDAFftPlan() {
  cuda::ScopedActivateExecutorContext sac(parent_);
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

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlan(Stream *stream, uint64_t num_x,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[1] = {num_x};
  tsl::Status status =
      fft_plan_ptr->Initialize(parent_, stream, 1, elem_count, type,
                               /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x;
    LOG(ERROR) << "Failed to initialize cufft 1d plan: "
               << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, fft::Type type, bool in_place_fft,
    ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[1] = {num_x};
  tsl::Status status = fft_plan_ptr->Initialize(parent_, stream, 1, elem_count,
                                                type, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x;
    LOG(ERROR)
        << "Failed to initialize cufft 1d plan with customized allocator: "
        << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlan(Stream *stream, uint64_t num_x,
                                                 uint64_t num_y, fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[2] = {num_x, num_y};
  tsl::Status status =
      fft_plan_ptr->Initialize(parent_, stream, 1, elem_count, type,
                               /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y;
    LOG(ERROR) << "Failed to initialize cufft 2d plan: "
               << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, uint64 num_y, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[2] = {num_x, num_y};
  tsl::Status status = fft_plan_ptr->Initialize(parent_, stream, 2, elem_count,
                                                type, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y;
    LOG(ERROR)
        << "Failed to initialize cufft 2d plan with customized allocator: "
        << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlan(Stream *stream, uint64_t num_x,
                                                 uint64_t num_y, uint64 num_z,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[3] = {num_x, num_y, num_z};
  tsl::Status status =
      fft_plan_ptr->Initialize(parent_, stream, 3, elem_count, type,
                               /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y
               << " num_z: " << num_z;
    LOG(ERROR) << "Failed to initialize cufft 3d plan: "
               << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, uint64 num_y, uint64 num_z, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[3] = {num_x, num_y, num_z};
  tsl::Status status = fft_plan_ptr->Initialize(parent_, stream, 3, elem_count,
                                                type, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y
               << " num_z: " << num_z;
    LOG(ERROR)
        << "Failed to initialize cufft 3d plan with customized allocator: "
        << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlan(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  tsl::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, /*scratch_allocator=*/nullptr);
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
    LOG(ERROR) << "Failed to initialize batched cufft plan: "
               << status.error_message();
    return nullptr;
  }

  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  tsl::Status status = fft_plan_ptr->Initialize(
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
        << status.error_message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

void CUDAFft::UpdatePlanWithScratchAllocator(
    Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);
  tsl::Status status =
      cuda_fft_plan->UpdateScratchAllocator(stream, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to update custom allocator for cufft plan: "
               << status.error_message();
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
        if (stream->ThenMemcpy(&allocated.value(), input, input.size()).ok()) {
          input_maybe_copy = DeviceMemory<InputT>(allocated.value());
        }
      }
      // Keep going even the workaround fails, since we don't have a good
      // bounding box. We don't want to give up on a potentially correct
      // execution just because the allocation for the incorrect case fails.
    }
  }
#endif

  cuda::ScopedActivateExecutorContext sac(parent_);
  auto ret =
      cufftExec(cuda_fft_plan->GetPlan(),
                GpuComplex(const_cast<InputT *>(GpuMemory(input_maybe_copy))),
                GpuComplex(GpuMemoryMutable(output)));

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

  cuda::ScopedActivateExecutorContext sac(parent_);
  auto ret = cufftExec(cuda_fft_plan->GetPlan(),
                       GpuComplex(const_cast<InputT *>(GpuMemory(input))),
                       GpuComplex(GpuMemoryMutable(output)),
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
  tsl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
          cuda::kCudaPlatformId, gpu::kCuFftPlugin, "cuFFT",
          [](internal::StreamExecutorInterface *parent) -> fft::FftSupport * {
            gpu::GpuExecutor *cuda_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the cuFFT "
                         << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            return new gpu::CUDAFft(cuda_executor);
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuFFT factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kFft, gpu::kCuFftPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cufft,
                            { stream_executor::initialize_cufft(); });
