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

#include <dlfcn.h>

#include <complex>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dso_loader.h"
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

namespace dynload {

// This macro wraps a global identifier, given by __name, in a callable
// structure that loads the DLL symbol out of the DSO handle in a thread-safe
// manner on first use. This dynamic loading technique is used to avoid DSO
// dependencies on vendor libraries which may or may not be available in the
// deployed binary environment.
#define PERFTOOLS_GPUTOOLS_CUFFT_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                           \
    static const char *kName;                                              \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;       \
    static void *GetDsoHandle() {                                          \
      static auto status = internal::CachedDsoLoader::GetCufftDsoHandle(); \
      return status.ValueOrDie();                                          \
    }                                                                      \
    static FuncPointerT DynLoad() {                                        \
      static void *f = dlsym(GetDsoHandle(), kName);                       \
      CHECK(f != nullptr) << "could not find " << kName                    \
                          << " in cuFFT DSO; dlerror: " << dlerror();      \
      return reinterpret_cast<FuncPointerT>(f);                            \
    }                                                                      \
    template <typename... Args>                                            \
    cufftResult operator()(CUDAExecutor * parent, Args... args) {          \
      cuda::ScopedActivateExecutorContext sac{parent};                     \
      return DynLoad()(args...);                                           \
    }                                                                      \
  } __name;                                                                \
  const char *DynLoadShim__##__name::kName = #__name;

#define CUFFT_ROUTINE_EACH(__macro)                                         \
  __macro(cufftDestroy) __macro(cufftSetStream) __macro(cufftPlan1d)        \
      __macro(cufftPlan2d) __macro(cufftPlan3d) __macro(cufftPlanMany)      \
          __macro(cufftExecD2Z) __macro(cufftExecZ2D) __macro(cufftExecC2C) \
              __macro(cufftExecC2R) __macro(cufftExecZ2Z)                   \
                  __macro(cufftExecR2C)

CUFFT_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUFFT_WRAP)

}  // namespace dynload

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
  auto ret = dynload::cufftSetStream(parent, plan, AsCUDAStreamValue(stream));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine cufftSetStream: " << ret;
    return false;
  }
  return true;
}

}  // namespace

CUDAFftPlan::CUDAFftPlan(CUDAExecutor *parent, uint64 num_x, fft::Type type)
    : parent_(parent), fft_type_(type) {
  auto ret = dynload::cufftPlan1d(parent, &plan_, num_x, CUDAFftType(type),
                                  1 /* = batch */);
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to create cuFFT 1d plan:" << ret;
  }
}

CUDAFftPlan::CUDAFftPlan(CUDAExecutor *parent, uint64 num_x, uint64 num_y,
                         fft::Type type)
    : parent_(parent), fft_type_(type) {
  auto ret =
      dynload::cufftPlan2d(parent, &plan_, num_x, num_y, CUDAFftType(type));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to create cuFFT 2d plan:" << ret;
  }
}

CUDAFftPlan::CUDAFftPlan(CUDAExecutor *parent, uint64 num_x, uint64 num_y,
                         uint64 num_z, fft::Type type)
    : parent_(parent), fft_type_(type) {
  auto ret = dynload::cufftPlan3d(parent, &plan_, num_x, num_y, num_z,
                                  CUDAFftType(type));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to create cuFFT 3d plan:" << ret;
  }
}

CUDAFftPlan::CUDAFftPlan(CUDAExecutor *parent, int rank, uint64 *elem_count,
                         uint64 *input_embed, uint64 input_stride,
                         uint64 input_distance, uint64 *output_embed,
                         uint64 output_stride, uint64 output_distance,
                         fft::Type type, int batch_count)
    : parent_(parent), fft_type_(type) {
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
  auto ret = dynload::cufftPlanMany(
      parent, &plan_, rank, elem_count_, input_embed ? input_embed_ : nullptr,
      input_stride, input_distance, output_embed ? output_embed_ : nullptr,
      output_stride, output_distance, CUDAFftType(type), batch_count);
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to create cuFFT batched plan:" << ret;
  }
}

CUDAFftPlan::~CUDAFftPlan() { dynload::cufftDestroy(parent_, plan_); }

int CUDAFftPlan::GetFftDirection() const {
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

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlan(Stream *stream, uint64 num_x,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<fft::Plan> plan{new CUDAFftPlan(parent_, num_x, type)};
  return plan;
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlan(Stream *stream, uint64 num_x,
                                                 uint64 num_y, fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<fft::Plan> plan{new CUDAFftPlan(parent_, num_x, num_y, type)};
  return plan;
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlan(Stream *stream, uint64 num_x,
                                                 uint64 num_y, uint64 num_z,
                                                 fft::Type type,
                                                 bool in_place_fft) {
  std::unique_ptr<fft::Plan> plan{
      new CUDAFftPlan(parent_, num_x, num_y, num_z, type)};
  return plan;
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlan(
    Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,
    uint64 input_stride, uint64 input_distance, uint64 *output_embed,
    uint64 output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
  std::unique_ptr<fft::Plan> plan{new CUDAFftPlan(
      parent_, rank, elem_count, input_embed, input_stride, input_distance,
      output_embed, output_stride, output_distance, type, batch_count)};
  return plan;
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

#define PERFTOOLS_GPUTOOLS_CUDA_DEFINE_FFT(__type, __fft_type1, __fft_type2,   \
                                           __fft_type3)                        \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<std::complex<__type>> &input,         \
                      DeviceMemory<std::complex<__type>> *output) {            \
    return DoFftWithDirectionInternal(                                         \
        stream, plan, dynload::cufftExec##__fft_type1, input, output);         \
  }                                                                            \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<__type> &input,                       \
                      DeviceMemory<std::complex<__type>> *output) {            \
    return DoFftInternal(stream, plan, dynload::cufftExec##__fft_type2, input, \
                         output);                                              \
  }                                                                            \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<std::complex<__type>> &input,         \
                      DeviceMemory<__type> *output) {                          \
    return DoFftInternal(stream, plan, dynload::cufftExec##__fft_type3, input, \
                         output);                                              \
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

  // Prime the cuFFT DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCufftDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load cuFFT DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kFft,
                                                     gpu::cuda::kCuFftPlugin);
});
