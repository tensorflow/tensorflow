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

#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_fft.h"

#include <complex>

#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/platform/dso_loader.h"
#include "tensorflow/compiler/xla/stream_executor/platform/initialize.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"
#include "tensorflow/tsl/platform/env.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocFftPlugin);

namespace wrap {

#ifdef PLATFORM_GOOGLE
// This macro wraps a global identifier, given by __name, in a callable
// structure that loads the DLL symbol out of the DSO handle in a thread-safe
// manner on first use. This dynamic loading technique is used to avoid DSO
// dependencies on vendor libraries which may or may not be available in the
// deployed binary environment.
#define STREAM_EXECUTOR_ROCFFT_WRAP(__name)                      \
  struct WrapperShim__##__name {                                 \
    template <typename... Args>                                  \
    hipfftResult operator()(GpuExecutor *parent, Args... args) { \
      gpu::ScopedActivateExecutorContext sac{parent};            \
      return ::__name(args...);                                  \
    }                                                            \
  } __name;

#else

#define STREAM_EXECUTOR_ROCFFT_WRAP(__name)                        \
  struct DynLoadShim__##__name {                                   \
    static const char *kName;                                      \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;   \
    static void *GetDsoHandle() {                                  \
      auto s = internal::CachedDsoLoader::GetHipfftDsoHandle();    \
      return s.value();                                            \
    }                                                              \
    static FuncPtrT LoadOrDie() {                                  \
      void *f;                                                     \
      auto s = tsl::Env::Default()                                 \
          -> GetSymbolFromLibrary(GetDsoHandle(), kName, &f);      \
      CHECK(s.ok()) << "could not find " << kName                  \
                    << " in rocfft DSO; dlerror: " << s.message(); \
      return reinterpret_cast<FuncPtrT>(f);                        \
    }                                                              \
    static FuncPtrT DynLoad() {                                    \
      static FuncPtrT f = LoadOrDie();                             \
      return f;                                                    \
    }                                                              \
    template <typename... Args>                                    \
    hipfftResult operator()(GpuExecutor *parent, Args... args) {   \
      gpu::ScopedActivateExecutorContext sac{parent};              \
      return DynLoad()(args...);                                   \
    }                                                              \
  } __name;                                                        \
  const char *DynLoadShim__##__name::kName = #__name;

#endif

// clang-format off
#define ROCFFT_ROUTINE_EACH(__macro) \
  __macro(hipfftDestroy)             \
  __macro(hipfftSetStream)           \
  __macro(hipfftPlan1d)              \
  __macro(hipfftPlan2d)              \
  __macro(hipfftPlan3d)              \
  __macro(hipfftPlanMany)            \
  __macro(hipfftCreate)              \
  __macro(hipfftSetAutoAllocation)   \
  __macro(hipfftSetWorkArea)         \
  __macro(hipfftGetSize1d)           \
  __macro(hipfftMakePlan1d)          \
  __macro(hipfftGetSize2d)           \
  __macro(hipfftMakePlan2d)          \
  __macro(hipfftGetSize3d)           \
  __macro(hipfftMakePlan3d)          \
  __macro(hipfftGetSizeMany)         \
  __macro(hipfftMakePlanMany)        \
  __macro(hipfftExecD2Z)             \
  __macro(hipfftExecZ2D)             \
  __macro(hipfftExecC2C)             \
  __macro(hipfftExecC2R)             \
  __macro(hipfftExecZ2Z)             \
  __macro(hipfftExecR2C)

// clang-format on

ROCFFT_ROUTINE_EACH(STREAM_EXECUTOR_ROCFFT_WRAP)

}  // namespace wrap

namespace {

// A helper function transforming gpu_fft arguments into rocFFT arguments.
hipfftType ROCMFftType(fft::Type type) {
  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kC2CInverse:
      return HIPFFT_C2C;
    case fft::Type::kC2R:
      return HIPFFT_C2R;
    case fft::Type::kR2C:
      return HIPFFT_R2C;
    case fft::Type::kZ2ZForward:
    case fft::Type::kZ2ZInverse:
      return HIPFFT_Z2Z;
    case fft::Type::kZ2D:
      return HIPFFT_Z2D;
    case fft::Type::kD2Z:
      return HIPFFT_D2Z;
    default:
      LOG(FATAL) << "Invalid value of fft::Type.";
  }
}

// Associates the given stream with the given rocFFT plan.
bool SetStream(GpuExecutor *parent, hipfftHandle plan, Stream *stream) {
  auto ret = wrap::hipfftSetStream(parent, plan, AsGpuStreamValue(stream));
  if (ret != HIPFFT_SUCCESS) {
    LOG(ERROR) << "failed to run rocFFT routine hipfftSetStream: " << ret;
    return false;
  }
  return true;
}

}  // namespace

tsl::Status ROCMFftPlan::Initialize(
    GpuExecutor *parent, Stream *stream, int rank, uint64_t *elem_count,
    uint64_t *input_embed, uint64 input_stride, uint64 input_distance,
    uint64_t *output_embed, uint64 output_stride, uint64 output_distance,
    fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
  if (IsInitialized()) {
    LOG(FATAL) << "Try to repeatedly initialize.";
  }
  is_initialized_ = true;
  scratch_allocator_ = scratch_allocator;
  int elem_count_[3], input_embed_[3], output_embed_[3];
  for (int i = 0; i < rank; ++i) {
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
    hipfftResult_t ret;
    if (scratch_allocator == nullptr) {
      switch (rank) {
        case 1:
          // hipfftPlan1d
          ret = wrap::hipfftPlan1d(parent, &plan_, elem_count_[0],
                                   ROCMFftType(type), 1 /* = batch */);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to create rocFFT 1d plan:" << ret;
            return tsl::Status{absl::StatusCode::kInternal,
                               "Failed to create rocFFT 1d plan."};
          }
          return tsl::OkStatus();
        case 2:
          // hipfftPlan2d
          ret = wrap::hipfftPlan2d(parent, &plan_, elem_count_[0],
                                   elem_count_[1], ROCMFftType(type));
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to create rocFFT 2d plan:" << ret;
            return tsl::Status{absl::StatusCode::kInternal,
                               "Failed to create rocFFT 2d plan."};
          }
          return tsl::OkStatus();
        case 3:
          // hipfftPlan3d
          ret =
              wrap::hipfftPlan3d(parent, &plan_, elem_count_[0], elem_count_[1],
                                 elem_count_[2], ROCMFftType(type));
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to create rocFFT 3d plan:" << ret;
            return tsl::Status{absl::StatusCode::kInternal,
                               "Failed to create rocFFT 3d plan."};
          }
          return tsl::OkStatus();
        default:
          LOG(ERROR) << "Invalid rank value for hipfftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return tsl::Status{absl::StatusCode::kInvalidArgument,
                             "hipfftPlan only takes rank 1, 2, or 3."};
      }
    } else {
      ret = wrap::hipfftCreate(parent, &plan_);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to create rocFFT plan:" << ret;
        return tsl::Status{absl::StatusCode::kInternal,
                           "Failed to create rocFFT plan."};
      }
      ret = wrap::hipfftSetAutoAllocation(parent, plan_, 0);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for rocFFT plan:" << ret;
        return tsl::Status{absl::StatusCode::kInternal,
                           "Failed to set auto allocation for rocFFT plan."};
      }
      switch (rank) {
        case 1:
          ret = wrap::hipfftMakePlan1d(parent, plan_, elem_count_[0],
                                       ROCMFftType(type), /*batch=*/1,
                                       &scratch_size_bytes_);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to make rocFFT 1d plan:" << ret;
            return tsl::Status{absl::StatusCode::kInternal,
                               "Failed to make rocFFT 1d plan."};
          }
          break;
        case 2:
          ret = wrap::hipfftMakePlan2d(parent, plan_, elem_count_[0],
                                       elem_count_[1], ROCMFftType(type),
                                       &scratch_size_bytes_);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to make rocFFT 2d plan:" << ret;
            return tsl::Status{absl::StatusCode::kInternal,
                               "Failed to make rocFFT 2d plan."};
          }
          break;
        case 3:
          ret = wrap::hipfftMakePlan3d(parent, plan_, elem_count_[0],
                                       elem_count_[1], elem_count_[2],
                                       ROCMFftType(type), &scratch_size_bytes_);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to make rocFFT 3d plan:" << ret;
            return tsl::Status{absl::StatusCode::kInternal,
                               "Failed to make rocFFT 3d plan."};
          }
          break;
        default:
          LOG(ERROR) << "Invalid rank value for hipfftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return tsl::Status{absl::StatusCode::kInvalidArgument,
                             "hipfftPlan only takes rank 1, 2, or 3."};
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  } else {
    // For either multiple batches or rank higher than 3, use hipfftPlanMany().
    if (scratch_allocator == nullptr) {
      auto ret = wrap::hipfftPlanMany(
          parent, &plan_, rank, elem_count_,
          input_embed ? input_embed_ : nullptr, input_stride, input_distance,
          output_embed ? output_embed_ : nullptr, output_stride,
          output_distance, ROCMFftType(type), batch_count);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to create rocFFT batched plan:" << ret;
        return tsl::Status{absl::StatusCode::kInternal,
                           "Failed to create rocFFT batched plan."};
      }
    } else {
      auto ret = wrap::hipfftCreate(parent, &plan_);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to create rocFFT batched plan:" << ret;
        return tsl::Status{absl::StatusCode::kInternal,
                           "Failed to create rocFFT batched plan."};
      }
      ret = wrap::hipfftSetAutoAllocation(parent, plan_, 0);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for rocFFT batched plan:"
                   << ret;
        return tsl::Status{
            absl::StatusCode::kInternal,
            "Failed to set auto allocation for rocFFT batched plan."};
      }
      ret = wrap::hipfftMakePlanMany(
          parent, plan_, rank, elem_count_,
          input_embed ? input_embed_ : nullptr, input_stride, input_distance,
          output_embed ? output_embed_ : nullptr, output_stride,
          output_distance, ROCMFftType(type), batch_count,
          &scratch_size_bytes_);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to make rocFFT batched plan:" << ret;
        return tsl::Status{absl::StatusCode::kInternal,
                           "Failed to make rocFFT batched plan."};
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  }
  return tsl::OkStatus();
}

tsl::Status ROCMFftPlan::Initialize(GpuExecutor *parent, Stream *stream,
                                    int rank, uint64_t *elem_count,
                                    fft::Type type,
                                    ScratchAllocator *scratch_allocator) {
  return Initialize(parent_, stream, rank, elem_count,
                    /*input_embed=*/nullptr, /*input_stride=*/0,
                    /*input_distance=*/0,
                    /*output_embed=*/nullptr, /*output_stride=*/0,
                    /*output_distance=*/0, type, 1, scratch_allocator);
}

tsl::Status ROCMFftPlan::UpdateScratchAllocator(
    Stream *stream, ScratchAllocator *scratch_allocator) {
  scratch_allocator_ = scratch_allocator;
  if (scratch_size_bytes_ != 0) {
    auto allocated = scratch_allocator->AllocateBytes(scratch_size_bytes_);
    if (!allocated.ok() || (scratch_ = allocated.value()) == nullptr) {
      LOG(ERROR) << "failed to allocate work area.";
      return allocated.status();
    }
  }
  // Connect work area with allocated space.
  auto ret = wrap::hipfftSetWorkArea(parent_, plan_, scratch_.opaque());
  if (ret != HIPFFT_SUCCESS) {
    LOG(ERROR) << "failed to set work area for rocFFT plan:" << ret;
    return tsl::Status(absl::StatusCode::kInternal,
                       "Failed to set work area for rocFFT plan.");
  }
  return tsl::OkStatus();
}

ROCMFftPlan::~ROCMFftPlan() { wrap::hipfftDestroy(parent_, plan_); }

int ROCMFftPlan::GetFftDirection() const {
  if (!IsInitialized()) {
    LOG(FATAL) << "Try to get fft direction before initialization.";
  } else {
    switch (fft_type_) {
      case fft::Type::kC2CForward:
      case fft::Type::kZ2ZForward:
      case fft::Type::kR2C:
      case fft::Type::kD2Z:
        return HIPFFT_FORWARD;
      case fft::Type::kC2CInverse:
      case fft::Type::kZ2ZInverse:
      case fft::Type::kC2R:
      case fft::Type::kZ2D:
        return HIPFFT_BACKWARD;
      default:
        LOG(FATAL) << "Invalid value of fft::Type.";
    }
  }
}

std::unique_ptr<fft::Plan> ROCMFft::Create1dPlan(Stream *stream, uint64_t num_x,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64_t elem_count[1] = {num_x};
  tsl::Status status =
      fft_plan_ptr->Initialize(parent_, stream, 1, elem_count, type,
                               /*scratch_allocator=*/nullptr);
  // TODO(yangzihao): In the future, send error msg back to TensorFlow
  // so it can fail gracefully,
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize hipfft 1d plan: " << status.message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create1dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, fft::Type type, bool in_place_fft,
    ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64_t elem_count[1] = {num_x};
  tsl::Status status = fft_plan_ptr->Initialize(parent_, stream, 1, elem_count,
                                                type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize hipfft 1d plan with customized allocator: "
        << status.message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create2dPlan(Stream *stream, uint64_t num_x,
                                                 uint64_t num_y, fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64_t elem_count[2] = {num_x, num_y};
  tsl::Status status =
      fft_plan_ptr->Initialize(parent_, stream, 1, elem_count, type,
                               /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize hipfft 2d plan: " << status.message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create2dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, uint64 num_y, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64_t elem_count[2] = {num_x, num_y};
  tsl::Status status = fft_plan_ptr->Initialize(parent_, stream, 2, elem_count,
                                                type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize hipfft 2d plan with customized allocator: "
        << status.message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create3dPlan(Stream *stream, uint64_t num_x,
                                                 uint64_t num_y, uint64 num_z,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64_t elem_count[3] = {num_x, num_y, num_z};
  tsl::Status status =
      fft_plan_ptr->Initialize(parent_, stream, 3, elem_count, type,
                               /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize hipfft 3d plan: " << status.message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create3dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, uint64 num_y, uint64 num_z, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64_t elem_count[3] = {num_x, num_y, num_z};
  tsl::Status status = fft_plan_ptr->Initialize(parent_, stream, 3, elem_count,
                                                type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize hipfft 3d plan with customized allocator: "
        << status.message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::CreateBatchedPlan(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  tsl::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize batched hipfft plan: "
               << status.message();
  }

  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  tsl::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize batched hipfft plan with customized "
                  "allocator: "
               << status.message();
  }
  return std::move(fft_plan_ptr);
}

void ROCMFft::UpdatePlanWithScratchAllocator(
    Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
  ROCMFftPlan *rocm_fft_plan = dynamic_cast<ROCMFftPlan *>(plan);
  tsl::Status status =
      rocm_fft_plan->UpdateScratchAllocator(stream, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL) << "failed to update custom allocator for hipfft plan: "
               << status.message();
  }
}

template <typename FuncT, typename InputT, typename OutputT>
bool ROCMFft::DoFftInternal(Stream *stream, fft::Plan *plan, FuncT hipfftExec,
                            const DeviceMemory<InputT> &input,
                            DeviceMemory<OutputT> *output) {
  ROCMFftPlan *rocm_fft_plan = dynamic_cast<ROCMFftPlan *>(plan);
  if (rocm_fft_plan == nullptr) {
    LOG(ERROR) << "the passed-in plan is not a ROCMFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, rocm_fft_plan->GetPlan(), stream)) {
    return false;
  }

  // As per rocFFT documentation, input buffers may be overwritten during
  // execution of the C2R / D2Z transforms, even if the transform is not
  // in-place.
  // see rocFFT issue #298 for more info
  //
  // Same seems to apply for the R2C / Z2D transforms, as reported in
  // see ROCm TF issue # 1150
  //
  // Hence for all those transforms, copy the input buffer
  DeviceMemory<InputT> input_maybe_copy = input;
  if (input.opaque() != output->opaque() && (input.size() > 0)) {
    auto *allocator = rocm_fft_plan->GetScratchAllocator();
    if (allocator) {
      auto allocated = allocator->AllocateBytes(input.size());
      if (allocated.ok()) {
        if (stream->ThenMemcpy(&allocated.value(), input, input.size())
                .ok()) {
          input_maybe_copy = DeviceMemory<InputT>(allocated.value());
        } else {
          LOG(ERROR) << "failed to copy input buffer for rocFFT.";
        }
      }
    }
  }

  InputT *ip = const_cast<InputT *>(GpuMemory(input_maybe_copy));
  auto ret = hipfftExec(parent_, rocm_fft_plan->GetPlan(), GpuComplex(ip),
                        GpuComplex(GpuMemoryMutable(output)));

  if (ret != HIPFFT_SUCCESS) {
    LOG(ERROR) << "failed to run rocFFT routine: " << ret;
    return false;
  }

  return true;
}

template <typename FuncT, typename InputT, typename OutputT>
bool ROCMFft::DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                         FuncT hipfftExec,
                                         const DeviceMemory<InputT> &input,
                                         DeviceMemory<OutputT> *output) {
  ROCMFftPlan *rocm_fft_plan = dynamic_cast<ROCMFftPlan *>(plan);
  if (rocm_fft_plan == nullptr) {
    LOG(ERROR) << "the passed-in plan is not a ROCMFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, rocm_fft_plan->GetPlan(), stream)) {
    return false;
  }

  auto ret = hipfftExec(parent_, rocm_fft_plan->GetPlan(),
                        GpuComplex(const_cast<InputT *>(GpuMemory(input))),
                        GpuComplex(GpuMemoryMutable(output)),
                        rocm_fft_plan->GetFftDirection());

  if (ret != HIPFFT_SUCCESS) {
    LOG(ERROR) << "failed to run rocFFT routine: " << ret;
    return false;
  }

  return true;
}

#define STREAM_EXECUTOR_ROCM_DEFINE_FFT(__type, __fft_type1, __fft_type2,    \
                                        __fft_type3)                         \
  bool ROCMFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<std::complex<__type>> &input,       \
                      DeviceMemory<std::complex<__type>> *output) {          \
    return DoFftWithDirectionInternal(                                       \
        stream, plan, wrap::hipfftExec##__fft_type1, input, output);         \
  }                                                                          \
  bool ROCMFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<__type> &input,                     \
                      DeviceMemory<std::complex<__type>> *output) {          \
    return DoFftInternal(stream, plan, wrap::hipfftExec##__fft_type2, input, \
                         output);                                            \
  }                                                                          \
  bool ROCMFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<std::complex<__type>> &input,       \
                      DeviceMemory<__type> *output) {                        \
    return DoFftInternal(stream, plan, wrap::hipfftExec##__fft_type3, input, \
                         output);                                            \
  }

STREAM_EXECUTOR_ROCM_DEFINE_FFT(float, C2C, R2C, C2R)
STREAM_EXECUTOR_ROCM_DEFINE_FFT(double, Z2Z, D2Z, Z2D)

#undef STREAM_EXECUTOR_ROCM_DEFINE_FFT

}  // namespace gpu

void initialize_rocfft() {
  auto rocFftAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kFft, gpu::kRocFftPlugin);

  if (!rocFftAlreadyRegistered) {
    tsl::Status status =
        PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
            rocm::kROCmPlatformId, gpu::kRocFftPlugin, "rocFFT",
            [](internal::StreamExecutorInterface *parent) -> fft::FftSupport * {
              gpu::GpuExecutor *rocm_executor =
                  dynamic_cast<gpu::GpuExecutor *>(parent);
              if (rocm_executor == nullptr) {
                LOG(ERROR)
                    << "Attempting to initialize an instance of the rocFFT "
                    << "support library with a non-ROCM StreamExecutor";
                return nullptr;
              }

              return new gpu::ROCMFft(rocm_executor);
            });
    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocFFT factory: " << status.message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kFft, gpu::kRocFftPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocfft,
                            { stream_executor::initialize_rocfft(); });
