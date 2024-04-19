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

// Common DSO loading functionality: exposes callables that dlopen DSOs
// in either the runfiles directories

#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_DSO_LOADER_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_DSO_LOADER_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tsl {
namespace internal {

namespace DsoLoader {
// The following methods either load the DSO of interest and return a dlopen
// handle or error status.
absl::StatusOr<void*> GetCudaDriverDsoHandle();
absl::StatusOr<void*> GetCudaRuntimeDsoHandle();
absl::StatusOr<void*> GetCublasDsoHandle();
absl::StatusOr<void*> GetCublasLtDsoHandle();
absl::StatusOr<void*> GetCufftDsoHandle();
absl::StatusOr<void*> GetCusolverDsoHandle();
absl::StatusOr<void*> GetCusparseDsoHandle();
absl::StatusOr<void*> GetCuptiDsoHandle();
absl::StatusOr<void*> GetCudnnDsoHandle();
absl::StatusOr<void*> GetNcclDsoHandle();
absl::StatusOr<void*> GetNvInferDsoHandle();
absl::StatusOr<void*> GetNvInferPluginDsoHandle();

absl::StatusOr<void*> GetRocblasDsoHandle();
absl::StatusOr<void*> GetMiopenDsoHandle();
absl::StatusOr<void*> GetHipfftDsoHandle();
absl::StatusOr<void*> GetRocrandDsoHandle();
absl::StatusOr<void*> GetRoctracerDsoHandle();
absl::StatusOr<void*> GetRocsolverDsoHandle();
absl::StatusOr<void*> GetHipsolverDsoHandle();
absl::StatusOr<void*> GetHipsparseDsoHandle();
absl::StatusOr<void*> GetHipDsoHandle();

// The following method tries to dlopen all necessary GPU libraries for the GPU
// platform TF is built with (CUDA or ROCm) only when these libraries should be
// dynamically loaded. Error status is returned when any of the libraries cannot
// be dlopened.
absl::Status MaybeTryDlopenGPULibraries();

// The following method tries to dlopen all necessary TensorRT libraries when
// these libraries should be dynamically loaded. Error status is returned when
// any of the libraries cannot be dlopened.
absl::Status TryDlopenTensorRTLibraries();
}  // namespace DsoLoader

// Wrapper around the DsoLoader that prevents us from dlopen'ing any of the DSOs
// more than once.
namespace CachedDsoLoader {
// Cached versions of the corresponding DsoLoader methods above.
absl::StatusOr<void*> GetCudaDriverDsoHandle();
absl::StatusOr<void*> GetCudaRuntimeDsoHandle();
absl::StatusOr<void*> GetCublasDsoHandle();
absl::StatusOr<void*> GetCublasLtDsoHandle();
absl::StatusOr<void*> GetCufftDsoHandle();
absl::StatusOr<void*> GetCusolverDsoHandle();
absl::StatusOr<void*> GetCusparseDsoHandle();
absl::StatusOr<void*> GetCuptiDsoHandle();
absl::StatusOr<void*> GetCudnnDsoHandle();

absl::StatusOr<void*> GetRocblasDsoHandle();
absl::StatusOr<void*> GetMiopenDsoHandle();
absl::StatusOr<void*> GetHipfftDsoHandle();
absl::StatusOr<void*> GetRocrandDsoHandle();
absl::StatusOr<void*> GetRocsolverDsoHandle();
absl::StatusOr<void*> GetHipsolverDsoHandle();
absl::StatusOr<void*> GetRoctracerDsoHandle();
absl::StatusOr<void*> GetHipsparseDsoHandle();
absl::StatusOr<void*> GetHipblasltDsoHandle();
absl::StatusOr<void*> GetHipDsoHandle();
}  // namespace CachedDsoLoader

}  // namespace internal
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_DSO_LOADER_H_
