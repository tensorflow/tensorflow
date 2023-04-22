/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {
namespace internal {
namespace DsoLoader {

port::Status TryDlopenCUDALibraries() {
  namespace CachedLoader = ::stream_executor::internal::CachedDsoLoader;
  auto cudart_status = CachedLoader::GetCudaRuntimeDsoHandle();
  auto cublas_status = CachedLoader::GetCublasDsoHandle();
  auto cublaslt_status = CachedLoader::GetCublasLtDsoHandle();
  auto cufft_status = CachedLoader::GetCufftDsoHandle();
  auto curand_status = CachedLoader::GetCurandDsoHandle();
  auto cusolver_status = CachedLoader::GetCusolverDsoHandle();
  auto cusparse_status = CachedLoader::GetCusparseDsoHandle();
  auto cudnn_status = CachedLoader::GetCudnnDsoHandle();

  if (!cudart_status.status().ok() || !cublas_status.status().ok() ||
      !cufft_status.status().ok() || !curand_status.status().ok() ||
      !cusolver_status.status().ok() || !cusparse_status.status().ok() ||
      !cudnn_status.status().ok() || !cublaslt_status.status().ok()) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("Cannot dlopen all CUDA libraries."));
  } else {
    return port::Status::OK();
  }
}

port::Status TryDlopenROCmLibraries() {
  auto rocblas_status = GetRocblasDsoHandle();
  auto miopen_status = GetMiopenDsoHandle();
  auto rocfft_status = GetHipfftDsoHandle();
  auto rocrand_status = GetRocrandDsoHandle();
  if (!rocblas_status.status().ok() || !miopen_status.status().ok() ||
      !rocfft_status.status().ok() || !rocrand_status.status().ok()) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("Cannot dlopen all ROCm libraries."));
  } else {
    return port::Status::OK();
  }
}

port::Status MaybeTryDlopenGPULibraries() {
#if GOOGLE_CUDA
  return TryDlopenCUDALibraries();
#elif TENSORFLOW_USE_ROCM
  return TryDlopenROCmLibraries();
#else
  LOG(INFO) << "Not built with GPU enabled. Skip GPU library dlopen check.";
  return port::Status::OK();
#endif
}

port::Status TryDlopenTensorRTLibraries() {
  auto nvinfer_status = GetNvInferDsoHandle();
  auto nvinferplugin_status = GetNvInferPluginDsoHandle();
  if (!nvinfer_status.status().ok() || !nvinferplugin_status.status().ok()) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("Cannot dlopen all TensorRT libraries."));
  } else {
    return port::Status::OK();
  }
}

}  // namespace DsoLoader
}  // namespace internal
}  // namespace stream_executor
