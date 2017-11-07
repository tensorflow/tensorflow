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

#include "tensorflow/stream_executor/cuda/cuda_fft.h"

#include <complex>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuFftPlugin);

namespace wrap {

// This macro wraps a global identifier, given by __name, in a callable
// structure that loads the DLL symbol out of the DSO handle in a thread-safe
// manner on first use. This dynamic loading technique is used to avoid DSO
// dependencies on vendor libraries which may or may not be available in the
// deployed binary environment.
#define PERFTOOLS_GPUTOOLS_CUFFT_WRAP(__name)                    \
  struct WrapperShim__##__name {                                 \
    template <typename... Args>                                  \
    cufftResult operator()(CUDAExecutor *parent, Args... args) { \
      cuda::ScopedActivateExecutorContext sac{parent};           \
      return ::__name(args...);                                  \
    }                                                            \
  } __name;

#define CUFFT_ROUTINE_EACH(__macro)                                            \
  __macro(cufftDestroy) __macro(cufftSetStream) __macro(cufftPlan1d)           \
      __macro(cufftPlan2d) __macro(cufftPlan3d) __macro(cufftPlanMany)         \
          __macro(cufftExecD2Z) __macro(cufftExecZ2D) __macro(cufftExecC2C)    \
              __macro(cufftExecC2R) __macro(cufftExecZ2Z)                      \
                  __macro(cufftExecR2C) __macro(cufftCreate)                   \
                      __macro(cufftSetAutoAllocation)                          \
                          __macro(cufftSetWorkArea) __macro(cufftGetSize1d)    \
                              __macro(cufftMakePlan1d) __macro(cufftGetSize2d) \
                                  __macro(cufftMakePlan2d)                     \
                                      __macro(cufftGetSize3d)                  \
                                          __macro(cufftMakePlan3d)             \
                                              __macro(cufftGetSizeMany)        \
                                                  __macro(cufftMakePlanMany)

CUFFT_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUFFT_WRAP)

}  // namespace wrap

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
bool SetStream(CUDAExecutor *parent, cufftHandle plan, Stream *stream) {
  auto ret = wrap::cufftSetStream(parent, plan, AsCUDAStreamValue(stream));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine cufftSetStream: " << ret;
    return false;
  }
  return true;
}

}  // namespace

port::Status CUDAFftPlan::Initialize(
    CUDAExecutor *parent, Stream *stream, int rank, uint64 *elem_count,
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
    cufftResult_t ret;
    if (scratch_allocator == nullptr) {
      switch (rank) {
        case 1:
          // cufftPlan1d
          ret = wrap::cufftPlan1d(parent, &plan_, elem_count_[0],
                                  CUDAFftType(type), 1 /* = batch */);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to create cuFFT 1d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to create cuFFT 1d plan."};
          }
          return port::Status::OK();
        case 2:
          // cufftPlan2d
          ret = wrap::cufftPlan2d(parent, &plan_, elem_count_[0],
                                  elem_count_[1], CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to create cuFFT 2d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to create cuFFT 2d plan."};
          }
          return port::Status::OK();
        case 3:
          // cufftPlan3d
          ret =
              wrap::cufftPlan3d(parent, &plan_, elem_count_[0], elem_count_[1],
                                elem_count_[2], CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to create cuFFT 3d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to create cuFFT 3d plan."};
          }
          return port::Status::OK();
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return port::Status{port::error::INVALID_ARGUMENT,
                              "cufftPlan only takes rank 1, 2, or 3."};
      }
    } else {
      ret = wrap::cufftCreate(parent, &plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to create cuFFT plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to create cuFFT plan."};
      }
      ret = wrap::cufftSetAutoAllocation(parent, plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for cuFFT plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to set auto allocation for cuFFT plan."};
      }
      size_t size_in_bytes;
      switch (rank) {
        case 1:
          ret = wrap::cufftMakePlan1d(parent, plan_, elem_count_[0],
                                      CUDAFftType(type), /*batch=*/1,
                                      &size_in_bytes);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to make cuFFT 1d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to make cuFFT 1d plan."};
          }
          break;
        case 2:
          ret = wrap::cufftMakePlan2d(parent, plan_, elem_count_[0],
                                      elem_count_[1], CUDAFftType(type),
                                      &size_in_bytes);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to make cuFFT 2d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to make cuFFT 2d plan."};
          }
          break;
        case 3:
          ret = wrap::cufftMakePlan3d(parent, plan_, elem_count_[0],
                                      elem_count_[1], elem_count_[2],
                                      CUDAFftType(type), &size_in_bytes);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to make cuFFT 3d plan:" << ret;
            return port::Status{port::error::INTERNAL,
                                "Failed to make cuFFT 3d plan."};
          }
          break;
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return port::Status{port::error::INVALID_ARGUMENT,
                              "cufftPlan only takes rank 1, 2, or 3."};
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
      ret = wrap::cufftSetWorkArea(parent, plan_, scratch_.opaque());
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to set work area for cuFFT plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to set work area for cuFFT plan."};
      }
      return port::Status::OK();
    }
  } else {
    // For either multiple batches or rank higher than 3, use cufftPlanMany().
    if (scratch_allocator == nullptr) {
      auto ret = wrap::cufftPlanMany(
          parent, &plan_, rank, elem_count_,
          input_embed ? input_embed_ : nullptr, input_stride, input_distance,
          output_embed ? output_embed_ : nullptr, output_stride,
          output_distance, CUDAFftType(type), batch_count);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to create cuFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to create cuFFT batched plan."};
      }
    } else {
      auto ret = wrap::cufftCreate(parent, &plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to create cuFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to create cuFFT batched plan."};
      }
      ret = wrap::cufftSetAutoAllocation(parent, plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for cuFFT batched plan:"
                   << ret;
        return port::Status{
            port::error::INTERNAL,
            "Failed to set auto allocation for cuFFT batched plan."};
      }
      size_t size_in_bytes;
      ret = wrap::cufftMakePlanMany(
          parent, plan_, rank, elem_count_,
          input_embed ? input_embed_ : nullptr, input_stride, input_distance,
          output_embed ? output_embed_ : nullptr, output_stride,
          output_distance, CUDAFftType(type), batch_count, &size_in_bytes);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to make cuFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to make cuFFT batched plan."};
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
      ret = wrap::cufftSetWorkArea(parent, plan_, scratch_.opaque());
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to set work area for cuFFT batched plan:" << ret;
        return port::Status{port::error::INTERNAL,
                            "Failed to set work area for cuFFT batched plan."};
      }
    }
  }
  return port::Status::OK();
}

port::Status CUDAFftPlan::Initialize(CUDAExecutor *parent, Stream *stream,
                                     int rank, uint64 *elem_count,
                                     fft::Type type,
                                     ScratchAllocator *scratch_allocator) {
  return Initialize(parent_, stream, rank, elem_count,
                    /*input_embed=*/nullptr, /*input_stride=*/0,
                    /*input_distance=*/0,
                    /*output_embed=*/nullptr, /*output_stride=*/0,
                    /*output_distance=*/0, type, 1, scratch_allocator);
}

CUDAFftPlan::~CUDAFftPlan() { wrap::cufftDestroy(parent_, plan_); }

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

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlan(Stream *stream, uint64 num_x,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64 elem_count[1] = {num_x};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 1, elem_count, type, /*scratch_allocator=*/nullptr);
  // TODO(yangzihao): In the future, send error msg back to TensorFlow
  // so it can fail gracefully,
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize cufft 1d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlanWithScratchAllocator(
    Stream *stream, uint64 num_x, fft::Type type, bool in_place_fft,
    ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64 elem_count[1] = {num_x};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 1, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize cufft 1d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlan(Stream *stream, uint64 num_x,
                                                 uint64 num_y, fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64 elem_count[2] = {num_x, num_y};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 1, elem_count, type, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize cufft 2d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlanWithScratchAllocator(
    Stream *stream, uint64 num_x, uint64 num_y, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64 elem_count[2] = {num_x, num_y};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 2, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize cufft 2d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlan(Stream *stream, uint64 num_x,
                                                 uint64 num_y, uint64 num_z,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64 elem_count[3] = {num_x, num_y, num_z};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 3, elem_count, type, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize cufft 3d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlanWithScratchAllocator(
    Stream *stream, uint64 num_x, uint64 num_y, uint64 num_z, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64 elem_count[3] = {num_x, num_y, num_z};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 3, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize cufft 3d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlan(
    Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,
    uint64 input_stride, uint64 input_distance, uint64 *output_embed,
    uint64 output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(FATAL) << "failed to initialize batched cufft plan: "
               << status.error_message();
  }

  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,
    uint64 input_stride, uint64 input_distance, uint64 *output_embed,
    uint64 output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL)
        << "failed to initialize batched cufft plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

template <typename FuncT, typename InputT, typename OutputT>
bool CUDAFft::DoFftInternal(Stream *stream, fft::Plan *plan, FuncT cufftExec,
                            const DeviceMemory<InputT> &input,
                            DeviceMemory<OutputT> *output) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);
  if (cuda_fft_plan == nullptr) {
    LOG(ERROR) << "the passed-in plan is not a CUDAFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, cuda_fft_plan->GetPlan(), stream)) {
    return false;
  }

  auto ret = cufftExec(parent_, cuda_fft_plan->GetPlan(),
                       CUDAComplex(const_cast<InputT *>(CUDAMemory(input))),
                       CUDAComplex(CUDAMemoryMutable(output)));

  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine: " << ret;
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
    LOG(ERROR) << "the passed-in plan is not a CUDAFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, cuda_fft_plan->GetPlan(), stream)) {
    return false;
  }

  auto ret = cufftExec(parent_, cuda_fft_plan->GetPlan(),
                       CUDAComplex(const_cast<InputT *>(CUDAMemory(input))),
                       CUDAComplex(CUDAMemoryMutable(output)),
                       cuda_fft_plan->GetFftDirection());

  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine: " << ret;
    return false;
  }

  return true;
}

#define PERFTOOLS_GPUTOOLS_CUDA_DEFINE_FFT(__type, __fft_type1, __fft_type2, \
                                           __fft_type3)                      \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<std::complex<__type>> &input,       \
                      DeviceMemory<std::complex<__type>> *output) {          \
    return DoFftWithDirectionInternal(                                       \
        stream, plan, wrap::cufftExec##__fft_type1, input, output);          \
  }                                                                          \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<__type> &input,                     \
                      DeviceMemory<std::complex<__type>> *output) {          \
    return DoFftInternal(stream, plan, wrap::cufftExec##__fft_type2, input,  \
                         output);                                            \
  }                                                                          \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                       \
                      const DeviceMemory<std::complex<__type>> &input,       \
                      DeviceMemory<__type> *output) {                        \
    return DoFftInternal(stream, plan, wrap::cufftExec##__fft_type3, input,  \
                         output);                                            \
  }

PERFTOOLS_GPUTOOLS_CUDA_DEFINE_FFT(float, C2C, R2C, C2R)
PERFTOOLS_GPUTOOLS_CUDA_DEFINE_FFT(double, Z2Z, D2Z, Z2D)

#undef PERFTOOLS_GPUTOOLS_CUDA_DEFINE_FFT

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

namespace gpu = ::perftools::gputools;

REGISTER_MODULE_INITIALIZER(register_cufft, {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::FftFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuFftPlugin, "cuFFT",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::fft::FftSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuFFT "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                return new gpu::cuda::CUDAFft(cuda_executor);
              });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuFFT factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kFft,
                                                     gpu::cuda::kCuFftPlugin);
});
