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
#include "xla/tsl/platform/default/dso_loader.h"

#include <stdlib.h>

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/cuda_config.h"
#include "third_party/nccl/nccl_config.h"
#include "third_party/nvshmem/nvshmem_config.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "third_party/tensorrt/tensorrt_config.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace tsl {
namespace internal {

namespace {
absl::string_view GetCudaVersion() { return TF_CUDA_VERSION; }
absl::string_view GetCudaRtVersion() { return TF_CUDART_VERSION; }
absl::string_view GetCuptiVersion() { return TF_CUPTI_VERSION; }
absl::string_view GetCudnnVersion() { return TF_CUDNN_VERSION; }
absl::string_view GetCublasVersion() { return TF_CUBLAS_VERSION; }
absl::string_view GetCusolverVersion() { return TF_CUSOLVER_VERSION; }
absl::string_view GetCufftVersion() { return TF_CUFFT_VERSION; }
absl::string_view GetCusparseVersion() { return TF_CUSPARSE_VERSION; }
absl::string_view GetNcclVersion() { return TF_NCCL_VERSION; }
absl::string_view GetTensorRTVersion() { return TF_TENSORRT_VERSION; }
absl::string_view GetNvshmemVersion() { return XLA_NVSHMEM_VERSION; }
absl::string_view GetHipVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_HIPRUNTIME_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
absl::string_view GetRocblasVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_ROCBLAS_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}

std::string GetHipblasltVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_HIPBLASLT_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetMiopenVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_MIOPEN_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetHipfftVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_HIPFFT_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetRocsolverVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_ROCSOLVER_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetHipsparseVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_HIPSPARSE_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetRoctracerVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_ROCTRACER_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetHipsolverVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_HIPSOLVER_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}
std::string GetRocrandVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_ROCRAND_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}

absl::StatusOr<void*> GetDsoHandle(const std::string& name,
                                   absl::string_view version) {
  auto filename =
      tsl::internal::FormatLibraryFileName(name, std::string(version));
  void* dso_handle;
  absl::Status status =
      tsl::internal::LoadDynamicLibrary(filename.c_str(), &dso_handle);
  if (status.ok()) {
    VLOG(1) << "Successfully opened dynamic library " << filename;
    return dso_handle;
  }

  auto message = absl::StrCat("Could not load dynamic library '", filename,
                              "'; dlerror: ", status.message());
#if !defined(PLATFORM_WINDOWS)
  if (const char* ld_library_path = getenv("LD_LIBRARY_PATH")) {
    absl::StrAppend(&message, "; LD_LIBRARY_PATH: ", ld_library_path);
  }
#endif
  VLOG(1) << message;
  return absl::Status(absl::StatusCode::kFailedPrecondition, message);
}
}  // namespace

namespace DsoLoader {
absl::StatusOr<void*> GetCudaDriverDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvcuda", "");
#elif defined(__APPLE__)
  // On Mac OS X, CUDA sometimes installs libcuda.dylib instead of
  // libcuda.1.dylib.
  auto handle_or = GetDsoHandle("cuda", "");
  if (handle_or.ok()) {
    return handle_or;
  }
#endif
  return GetDsoHandle("cuda", "1");
}

absl::StatusOr<void*> GetNvmlDsoHandle() {
  return GetDsoHandle("nvidia-ml", "1");
}

absl::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  return GetDsoHandle("cudart", GetCudaRtVersion());
}

absl::StatusOr<void*> GetCublasDsoHandle() {
  return GetDsoHandle("cublas", GetCublasVersion());
}

absl::StatusOr<void*> GetCublasLtDsoHandle() {
  return GetDsoHandle("cublasLt", GetCublasVersion());
}

absl::StatusOr<void*> GetCufftDsoHandle() {
  return GetDsoHandle("cufft", GetCufftVersion());
}

absl::StatusOr<void*> GetCusolverDsoHandle() {
  return GetDsoHandle("cusolver", GetCusolverVersion());
}

absl::StatusOr<void*> GetCusparseDsoHandle() {
  return GetDsoHandle("cusparse", GetCusparseVersion());
}

absl::StatusOr<void*> GetCuptiDsoHandle() {
  // Load specific version of CUPTI this is built.
  auto status_or_handle = GetDsoHandle("cupti", GetCuptiVersion());
  if (status_or_handle.ok()) return status_or_handle;
  // Load whatever libcupti.so user specified.
  return GetDsoHandle("cupti", "");
}

absl::StatusOr<void*> GetCudnnDsoHandle() {
  return GetDsoHandle("cudnn", GetCudnnVersion());
}

absl::StatusOr<void*> GetNcclDsoHandle() {
  return GetDsoHandle("nccl", GetNcclVersion());
}

absl::StatusOr<void*> GetNvshmemDsoHandle() {
  return GetDsoHandle("nvshmem_host", GetNvshmemVersion());
}

absl::StatusOr<void*> GetNvInferDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer", "");
#else
  return GetDsoHandle("nvinfer", GetTensorRTVersion());
#endif
}

absl::StatusOr<void*> GetNvInferPluginDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer_plugin", "");
#else
  return GetDsoHandle("nvinfer_plugin", GetTensorRTVersion());
#endif
}

absl::StatusOr<void*> GetRocblasDsoHandle() {
  return GetDsoHandle("rocblas", GetRocblasVersion());
}

absl::StatusOr<void*> GetMiopenDsoHandle() {
  return GetDsoHandle("MIOpen", GetMiopenVersion());
}

absl::StatusOr<void*> GetHipfftDsoHandle() {
  return GetDsoHandle("hipfft", GetHipfftVersion());
}

absl::StatusOr<void*> GetRocrandDsoHandle() {
  return GetDsoHandle("rocrand", GetRocrandVersion());
}

absl::StatusOr<void*> GetRocsolverDsoHandle() {
  return GetDsoHandle("rocsolver", GetRocsolverVersion());
}

#if TF_ROCM_VERSION >= 40500
absl::StatusOr<void*> GetHipsolverDsoHandle() {
  return GetDsoHandle("hipsolver", GetHipsolverVersion());
}
#endif

absl::StatusOr<void*> GetRoctracerDsoHandle() {
  return GetDsoHandle("roctracer64", GetRoctracerVersion());
}

absl::StatusOr<void*> GetHipsparseDsoHandle() {
  return GetDsoHandle("hipsparse", GetHipsparseVersion());
}

absl::StatusOr<void*> GetHipblasltDsoHandle() {
  return GetDsoHandle("hipblaslt", GetHipblasltVersion());
}

absl::StatusOr<void*> GetHipDsoHandle() {
  return GetDsoHandle("amdhip64", GetHipVersion());
}

}  // namespace DsoLoader

namespace CachedDsoLoader {
absl::StatusOr<void*> GetCudaDriverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaDriverDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaRuntimeDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCublasDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCublasLtDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasLtDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCufftDsoHandle() {
  static auto result = new auto(DsoLoader::GetCufftDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCusolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusolverDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCusparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusparseDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCuptiDsoHandle() {
  static auto result = new auto(DsoLoader::GetCuptiDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCudnnDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudnnDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetRocblasDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocblasDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetMiopenDsoHandle() {
  static auto result = new auto(DsoLoader::GetMiopenDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetHipfftDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipfftDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetRocrandDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocrandDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetRoctracerDsoHandle() {
  static auto result = new auto(DsoLoader::GetRoctracerDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetRocsolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocsolverDsoHandle());
  return *result;
}

#if TF_ROCM_VERSION >= 40500
absl::StatusOr<void*> GetHipsolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipsolverDsoHandle());
  return *result;
}
#endif

absl::StatusOr<void*> GetHipsparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipsparseDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetHipblasltDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipblasltDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetHipDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipDsoHandle());
  return *result;
}

}  // namespace CachedDsoLoader
}  // namespace internal
}  // namespace tsl
