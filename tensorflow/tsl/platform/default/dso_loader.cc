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
#include "tensorflow/tsl/platform/default/dso_loader.h"

#include <stdlib.h>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/cuda_config.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/platform.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "third_party/tensorrt/tensorrt_config.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace tsl {
namespace internal {

namespace {
string GetCudaVersion() { return TF_CUDA_VERSION; }
string GetCudaRtVersion() { return TF_CUDART_VERSION; }
string GetCudnnVersion() { return TF_CUDNN_VERSION; }
string GetCublasVersion() { return TF_CUBLAS_VERSION; }
string GetCusolverVersion() { return TF_CUSOLVER_VERSION; }
string GetCurandVersion() { return TF_CURAND_VERSION; }
string GetCufftVersion() { return TF_CUFFT_VERSION; }
string GetCusparseVersion() { return TF_CUSPARSE_VERSION; }
string GetTensorRTVersion() { return TF_TENSORRT_VERSION; }

StatusOr<void*> GetDsoHandle(const string& name, const string& version) {
  auto filename = Env::Default()->FormatLibraryFileName(name, version);
  void* dso_handle;
  Status status =
      Env::Default()->LoadDynamicLibrary(filename.c_str(), &dso_handle);
  if (status.ok()) {
    VLOG(1) << "Successfully opened dynamic library " << filename;
    return dso_handle;
  }

  auto message = absl::StrCat("Could not load dynamic library '", filename,
                              "'; dlerror: ", status.error_message());
#if !defined(PLATFORM_WINDOWS)
  if (const char* ld_library_path = getenv("LD_LIBRARY_PATH")) {
    message += absl::StrCat("; LD_LIBRARY_PATH: ", ld_library_path);
  }
#endif
  LOG(WARNING) << message;
  return Status(error::FAILED_PRECONDITION, message);
}
}  // namespace

namespace DsoLoader {
StatusOr<void*> GetCudaDriverDsoHandle() {
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

StatusOr<void*> GetCudaRuntimeDsoHandle() {
  return GetDsoHandle("cudart", GetCudaRtVersion());
}

StatusOr<void*> GetCublasDsoHandle() {
  return GetDsoHandle("cublas", GetCublasVersion());
}

StatusOr<void*> GetCublasLtDsoHandle() {
  return GetDsoHandle("cublasLt", GetCublasVersion());
}

StatusOr<void*> GetCufftDsoHandle() {
  return GetDsoHandle("cufft", GetCufftVersion());
}

StatusOr<void*> GetCusolverDsoHandle() {
  return GetDsoHandle("cusolver", GetCusolverVersion());
}

StatusOr<void*> GetCusparseDsoHandle() {
  return GetDsoHandle("cusparse", GetCusparseVersion());
}

StatusOr<void*> GetCurandDsoHandle() {
  return GetDsoHandle("curand", GetCurandVersion());
}

StatusOr<void*> GetCuptiDsoHandle() {
  // Load specific version of CUPTI this is built.
  auto status_or_handle = GetDsoHandle("cupti", GetCudaVersion());
  if (status_or_handle.ok()) return status_or_handle;
  // Load whatever libcupti.so user specified.
  return GetDsoHandle("cupti", "");
}

StatusOr<void*> GetCudnnDsoHandle() {
  return GetDsoHandle("cudnn", GetCudnnVersion());
}

StatusOr<void*> GetNvInferDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer", "");
#else
  return GetDsoHandle("nvinfer", GetTensorRTVersion());
#endif
}

StatusOr<void*> GetNvInferPluginDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer_plugin", "");
#else
  return GetDsoHandle("nvinfer_plugin", GetTensorRTVersion());
#endif
}

StatusOr<void*> GetRocblasDsoHandle() { return GetDsoHandle("rocblas", ""); }

StatusOr<void*> GetMiopenDsoHandle() { return GetDsoHandle("MIOpen", ""); }

StatusOr<void*> GetHipfftDsoHandle() { return GetDsoHandle("hipfft", ""); }

StatusOr<void*> GetRocrandDsoHandle() { return GetDsoHandle("rocrand", ""); }

StatusOr<void*> GetRocsolverDsoHandle() {
  return GetDsoHandle("rocsolver", "");
}

#if TF_ROCM_VERSION >= 40500
StatusOr<void*> GetHipsolverDsoHandle() {
  return GetDsoHandle("hipsolver", "");
}
#endif

StatusOr<void*> GetRoctracerDsoHandle() {
  return GetDsoHandle("roctracer64", "");
}

StatusOr<void*> GetHipsparseDsoHandle() {
  return GetDsoHandle("hipsparse", "");
}

StatusOr<void*> GetHipDsoHandle() { return GetDsoHandle("amdhip64", ""); }

}  // namespace DsoLoader

namespace CachedDsoLoader {
StatusOr<void*> GetCudaDriverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaDriverDsoHandle());
  return *result;
}

StatusOr<void*> GetCudaRuntimeDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaRuntimeDsoHandle());
  return *result;
}

StatusOr<void*> GetCublasDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasDsoHandle());
  return *result;
}

StatusOr<void*> GetCublasLtDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasLtDsoHandle());
  return *result;
}

StatusOr<void*> GetCurandDsoHandle() {
  static auto result = new auto(DsoLoader::GetCurandDsoHandle());
  return *result;
}

StatusOr<void*> GetCufftDsoHandle() {
  static auto result = new auto(DsoLoader::GetCufftDsoHandle());
  return *result;
}

StatusOr<void*> GetCusolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusolverDsoHandle());
  return *result;
}

StatusOr<void*> GetCusparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusparseDsoHandle());
  return *result;
}

StatusOr<void*> GetCuptiDsoHandle() {
  static auto result = new auto(DsoLoader::GetCuptiDsoHandle());
  return *result;
}

StatusOr<void*> GetCudnnDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudnnDsoHandle());
  return *result;
}

StatusOr<void*> GetRocblasDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocblasDsoHandle());
  return *result;
}

StatusOr<void*> GetMiopenDsoHandle() {
  static auto result = new auto(DsoLoader::GetMiopenDsoHandle());
  return *result;
}

StatusOr<void*> GetHipfftDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipfftDsoHandle());
  return *result;
}

StatusOr<void*> GetRocrandDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocrandDsoHandle());
  return *result;
}

StatusOr<void*> GetRoctracerDsoHandle() {
  static auto result = new auto(DsoLoader::GetRoctracerDsoHandle());
  return *result;
}

StatusOr<void*> GetRocsolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocsolverDsoHandle());
  return *result;
}

#if TF_ROCM_VERSION >= 40500
StatusOr<void*> GetHipsolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipsolverDsoHandle());
  return *result;
}
#endif

StatusOr<void*> GetHipsparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipsparseDsoHandle());
  return *result;
}

StatusOr<void*> GetHipDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipDsoHandle());
  return *result;
}

}  // namespace CachedDsoLoader
}  // namespace internal
}  // namespace tsl
