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
#include "tensorflow/stream_executor/platform/default/dso_loader.h"

#include <stdlib.h>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/cuda_config.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "third_party/tensorrt/tensorrt_config.h"

namespace stream_executor {
namespace internal {

namespace {
string GetCudaVersion() { return TF_CUDA_VERSION; }
string GetCudnnVersion() { return TF_CUDNN_VERSION; }
string GetCublasVersion() { return TF_CUBLAS_VERSION; }
string GetCusolverVersion() { return TF_CUSOLVER_VERSION; }
string GetCurandVersion() { return TF_CURAND_VERSION; }
string GetCufftVersion() { return TF_CUFFT_VERSION; }
string GetCusparseVersion() { return TF_CUSPARSE_VERSION; }
string GetTensorRTVersion() { return TF_TENSORRT_VERSION; }

port::StatusOr<void*> GetDsoHandle(const string& name, const string& version) {
  auto filename = port::Env::Default()->FormatLibraryFileName(name, version);
  void* dso_handle;
  port::Status status =
      port::Env::Default()->LoadLibrary(filename.c_str(), &dso_handle);
  if (status.ok()) {
    LOG(INFO) << "Successfully opened dynamic library " << filename;
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
  return port::Status(port::error::FAILED_PRECONDITION, message);
}
}  // namespace

namespace DsoLoader {
port::StatusOr<void*> GetCudaDriverDsoHandle() {
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

port::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  return GetDsoHandle("cudart", GetCudaVersion());
}

port::StatusOr<void*> GetCublasDsoHandle() {
  return GetDsoHandle("cublas", GetCublasVersion());
}

port::StatusOr<void*> GetCufftDsoHandle() {
  return GetDsoHandle("cufft", GetCufftVersion());
}

port::StatusOr<void*> GetCusolverDsoHandle() {
  return GetDsoHandle("cusolver", GetCusolverVersion());
}

port::StatusOr<void*> GetCusparseDsoHandle() {
  return GetDsoHandle("cusparse", GetCusparseVersion());
}

port::StatusOr<void*> GetCurandDsoHandle() {
  return GetDsoHandle("curand", GetCurandVersion());
}

port::StatusOr<void*> GetCuptiDsoHandle() {
#if defined(ANDROID_TEGRA)
  // On Android devices the CUDA version number is not added to the library
  // name.
  return GetDsoHandle("cupti", "");
#else
  return GetDsoHandle("cupti", GetCudaVersion());
#endif
}

port::StatusOr<void*> GetCudnnDsoHandle() {
  return GetDsoHandle("cudnn", GetCudnnVersion());
}

port::StatusOr<void*> GetNvInferDsoHandle() {
  return GetDsoHandle("nvinfer", GetTensorRTVersion());
}

port::StatusOr<void*> GetNvInferPluginDsoHandle() {
  return GetDsoHandle("nvinfer_plugin", GetTensorRTVersion());
}

port::StatusOr<void*> GetRocblasDsoHandle() {
  return GetDsoHandle("rocblas", "");
}

port::StatusOr<void*> GetMiopenDsoHandle() {
  return GetDsoHandle("MIOpen", "");
}

port::StatusOr<void*> GetRocfftDsoHandle() {
  return GetDsoHandle("rocfft", "");
}

port::StatusOr<void*> GetRocrandDsoHandle() {
  return GetDsoHandle("rocrand", "");
}

port::StatusOr<void*> GetHipDsoHandle() { return GetDsoHandle("hip_hcc", ""); }

}  // namespace DsoLoader

namespace CachedDsoLoader {
port::StatusOr<void*> GetCudaDriverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaDriverDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaRuntimeDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCublasDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCurandDsoHandle() {
  static auto result = new auto(DsoLoader::GetCurandDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCufftDsoHandle() {
  static auto result = new auto(DsoLoader::GetCufftDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCusolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusolverDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCusparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusparseDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCuptiDsoHandle() {
  static auto result = new auto(DsoLoader::GetCuptiDsoHandle());
  return *result;
}

port::StatusOr<void*> GetCudnnDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudnnDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRocblasDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocblasDsoHandle());
  return *result;
}

port::StatusOr<void*> GetMiopenDsoHandle() {
  static auto result = new auto(DsoLoader::GetMiopenDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRocfftDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocfftDsoHandle());
  return *result;
}

port::StatusOr<void*> GetRocrandDsoHandle() {
  static auto result = new auto(DsoLoader::GetRocrandDsoHandle());
  return *result;
}

port::StatusOr<void*> GetHipDsoHandle() {
  static auto result = new auto(DsoLoader::GetHipDsoHandle());
  return *result;
}

}  // namespace CachedDsoLoader
}  // namespace internal
}  // namespace stream_executor
