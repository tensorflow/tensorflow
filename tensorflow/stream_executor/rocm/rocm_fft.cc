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

#include "tensorflow/stream_executor/rocm/rocm_fft.h"

#include <complex>

#include "tensorflow/stream_executor/rocm/rocm_activation.h"
#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_helpers.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace rocm {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocFftPlugin);

namespace wrap {

// This macro wraps a global identifier, given by __name, in a callable
// structure that loads the DLL symbol out of the DSO handle in a thread-safe
// manner on first use. This dynamic loading technique is used to avoid DSO
// dependencies on vendor libraries which may or may not be available in the
// deployed binary environment.
#define PERFTOOLS_GPUTOOLS_HIPFFT_WRAP(__name)                    \
  struct WrapperShim__##__name {                                  \
    template <typename... Args>                                   \
    hipfftResult operator()(ROCMExecutor *parent, Args... args) { \
      rocm::ScopedActivateExecutorContext sac{parent};            \
      return ::__name(args...);                                   \
    }                                                             \
  } __name;

#define HIPFFT_ROUTINE_EACH(__macro) \
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
  __macro(hipfftExecR2C)             \

HIPFFT_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_HIPFFT_WRAP)

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
bool SetStream(ROCMExecutor *parent, hipfftHandle plan, Stream *stream) {
  auto ret = wrap::hipfftSetStream(parent, plan, AsROCMStreamValue(stream));
  if (ret != HIPFFT_SUCCESS) {
    LOG(ERROR) << "failed to run rocFFT routine hipfftSetStream: " << ret;
    return false;
  }
  return true;
}

}  // namespace

port::Status ROCMFftPlan::Initialize(
    ROCMExecutor *parent, Stream *stream, int rank, uint64 *elem_count,
    uint64 *input_embed, uint64 input_stride, uint64 input_distance,
    uint64 *output_embed, uint64 output_stride, uint64 output_distance,
    fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
  if (IsInitialized()) {
    LOG(FATAL) << "Try to repeatedly initialize.";
  }
  is_initialized_ = true;
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
            return port::Status{port::error::INTERNAL,
                                "Failed to create rocFFT 1d plan."};
          }
          return port::Status::OK();
        case 2:
          // hipfftPlan2d
          ret = wrap::hipfftPlan2d(parent, &plan_, elem_count_[0],
                                  elem_count_[1], ROCMFftType(type));
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to create rocFFT 2d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to create rocFFT 2d plan."};
          }
          return port::Status::OK();
        case 3:
          // hipfftPlan3d
          ret =
              wrap::hipfftPlan3d(parent, &plan_, elem_count_[0], elem_count_[1],
                                elem_count_[2], ROCMFftType(type));
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to create rocFFT 3d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to create rocFFT 3d plan."};
          }
          return port::Status::OK();
        default:
          LOG(ERROR) << "Invalid rank value for hipfftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return port::Status{port::error::INVALID_ARGUMENT,
                              "hipfftPlan only takes rank 1, 2, or 3."};
      }
    } else {
      ret = wrap::hipfftCreate(parent, &plan_);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to create rocFFT plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to create rocFFT plan."};
      }
      ret = wrap::hipfftSetAutoAllocation(parent, plan_, 0);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for rocFFT plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to set auto allocation for rocFFT plan."};
      }
      size_t size_in_bytes;
      switch (rank) {
        case 1:
          ret = wrap::hipfftMakePlan1d(parent, plan_, elem_count_[0],
                                      ROCMFftType(type), /*batch=*/1,
                                      &size_in_bytes);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to make rocFFT 1d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to make rocFFT 1d plan."};
          }
          break;
        case 2:
          ret = wrap::hipfftMakePlan2d(parent, plan_, elem_count_[0],
                                      elem_count_[1], ROCMFftType(type),
                                      &size_in_bytes);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to make rocFFT 2d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to make rocFFT 2d plan."};
          }
          break;
        case 3:
          ret = wrap::hipfftMakePlan3d(parent, plan_, elem_count_[0],
                                      elem_count_[1], elem_count_[2],
                                      ROCMFftType(type), &size_in_bytes);
          if (ret != HIPFFT_SUCCESS) {
            LOG(ERROR) << "failed to make rocFFT 3d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to make rocFFT 3d plan."};
          }
          break;
        default:
          LOG(ERROR) << "Invalid rank value for hipfftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return port::Status{port::error::INVALID_ARGUMENT,
                              "hipfftPlan only takes rank 1, 2, or 3."};
      }
      // TODO(yangzihao): refactor this code and the one with the same function
      // in the batch mode.
      if (size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (!allocated.ok() || (scratch_ = allocated.ValueOrDie()) == nullptr) {
          LOG(ERROR) << "failed to allocate work area.";
          return allocated.status();
        }
      }
      // Connect work area with allocated space.
      ret = wrap::hipfftSetWorkArea(parent, plan_, scratch_.opaque());
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to set work area for rocFFT plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to set work area for rocFFT plan."};
      }
      return port::Status::OK();
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
        return port::Status{port::error::INTERNAL,
                            "Failed to create rocFFT bacthed plan."};
      }
    } else {
      auto ret = wrap::hipfftCreate(parent, &plan_);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to create rocFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to create rocFFT bacthed plan."};
      }
      ret = wrap::hipfftSetAutoAllocation(parent, plan_, 0);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for rocFFT batched plan:"
                   << ret;
        return port::Status{
            port::error::INTERNAL,
            "Failed to set auto allocation for rocFFT bacthed plan."};
      }
      size_t size_in_bytes;
      ret = wrap::hipfftMakePlanMany(
          parent, plan_, rank, elem_count_,
          input_embed ? input_embed_ : nullptr, input_stride, input_distance,
          output_embed ? output_embed_ : nullptr, output_stride,
          output_distance, ROCMFftType(type), batch_count, &size_in_bytes);
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to make rocFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to make rocFFT bacthed plan."};
      }
      if (size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (!allocated.ok() || (scratch_ = allocated.ValueOrDie()) == nullptr) {
          LOG(ERROR) << "failed to allocate work area.";
          return allocated.status();
        }
      }
      // Connect work area with allocated space.
      ret = wrap::hipfftSetWorkArea(parent, plan_, scratch_.opaque());
      if (ret != HIPFFT_SUCCESS) {
        LOG(ERROR) << "failed to set work area for rocFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to set work area for rocFFT bacthed plan."};
      }
    }
  }
  return port::Status::OK();
}

port::Status ROCMFftPlan::Initialize(ROCMExecutor *parent, Stream *stream,
                                     int rank, uint64 *elem_count,
                                     fft::Type type,
                                     ScratchAllocator *scratch_allocator) {
  return Initialize(parent_, stream, rank, elem_count,
                    /*input_embed=*/nullptr, /*input_stride=*/0,
                    /*input_distance=*/0,
                    /*output_embed=*/nullptr, /*output_stride=*/0,
                    /*output_distance=*/0, type, 1, scratch_allocator);
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

std::unique_ptr<fft::Plan> ROCMFft::Create1dPlan(Stream *stream, uint64 num_x,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64 elem_count[1] = {num_x};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 1, elem_count, type, /*scratch_allocator=*/nullptr);
  // TODO(yangzihao): In the future, send error msg back to TensorFlow
  // so it can fail gracefully,
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize hipfft 1d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create1dPlanWithScratchAllocator(
    Stream *stream, uint64 num_x, fft::Type type, bool in_place_fft,
    ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64 elem_count[1] = {num_x};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 1, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize hipfft 1d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create2dPlan(Stream *stream, uint64 num_x,
                                                 uint64 num_y, fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64 elem_count[2] = {num_x, num_y};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 1, elem_count, type, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize hipfft 2d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create2dPlanWithScratchAllocator(
    Stream *stream, uint64 num_x, uint64 num_y, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64 elem_count[2] = {num_x, num_y};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 2, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize hipfft 2d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create3dPlan(Stream *stream, uint64 num_x,
                                                 uint64 num_y, uint64 num_z,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64 elem_count[3] = {num_x, num_y, num_z};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 3, elem_count, type, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize hipfft 3d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::Create3dPlanWithScratchAllocator(
    Stream *stream, uint64 num_x, uint64 num_y, uint64 num_z, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  uint64 elem_count[3] = {num_x, num_y, num_z};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 3, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize hipfft 3d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::CreateBatchedPlan(
    Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,
    uint64 input_stride, uint64 input_distance, uint64 *output_embed,
    uint64 output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize batched hipfft plan: "
               << status.error_message();
  }

  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> ROCMFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,
    uint64 input_stride, uint64 input_distance, uint64 *output_embed,
    uint64 output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<ROCMFftPlan> fft_plan_ptr{new ROCMFftPlan()};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize batched hipfft plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

void ROCMFft::UpdatePlanWithScratchAllocator(
    Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
  LOG(ERROR) << "update plan with scratch allocator not implemented";
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

  auto ret = hipfftExec(parent_, rocm_fft_plan->GetPlan(),
                       ROCMComplex(const_cast<InputT *>(ROCMMemory(input))),
                       ROCMComplex(ROCMMemoryMutable(output)));

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
                       ROCMComplex(const_cast<InputT *>(ROCMMemory(input))),
                       ROCMComplex(ROCMMemoryMutable(output)),
                       rocm_fft_plan->GetFftDirection());

  if (ret != HIPFFT_SUCCESS) {
    LOG(ERROR) << "failed to run rocFFT routine: " << ret;
    return false;
  }

  return true;
}

#define PERFTOOLS_GPUTOOLS_ROCM_DEFINE_FFT(__type, __fft_type1, __fft_type2, \
                                           __fft_type3)                      \
  bool ROCMFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<std::complex<__type>> &input,       \
                      DeviceMemory<std::complex<__type>> *output) {          \
    return DoFftWithDirectionInternal(                                       \
         stream, plan, wrap::hipfftExec##__fft_type1, input, output);        \
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

PERFTOOLS_GPUTOOLS_ROCM_DEFINE_FFT(float, C2C, R2C, C2R)
PERFTOOLS_GPUTOOLS_ROCM_DEFINE_FFT(double, Z2Z, D2Z, Z2D)

#undef PERFTOOLS_GPUTOOLS_ROCM_DEFINE_FFT

}  // namespace rocm
}  // namespace stream_executor

namespace gpu = ::stream_executor;

REGISTER_MODULE_INITIALIZER(register_hipfft, {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::FftFactory>(
              gpu::rocm::kROCmPlatformId, gpu::rocm::kRocFftPlugin, "rocFFT",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::fft::FftSupport * {
                gpu::rocm::ROCMExecutor *rocm_executor =
                    dynamic_cast<gpu::rocm::ROCMExecutor *>(parent);
                if (rocm_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the rocFFT "
                      << "support library with a non-ROCM StreamExecutor";
                  return nullptr;
                }

                return new gpu::rocm::ROCMFft(rocm_executor);
              });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register rocFFT factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::rocm::kROCmPlatformId,
                                                     gpu::PluginKind::kFft,
                                                     gpu::rocm::kRocFftPlugin);
});
