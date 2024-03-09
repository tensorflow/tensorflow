/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_DEFAULT_DSO_LOADER_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_DEFAULT_DSO_LOADER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor {
namespace internal {

namespace DsoLoader {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::internal::DsoLoader::GetCublasDsoHandle;
using tsl::internal::DsoLoader::GetCublasLtDsoHandle;
using tsl::internal::DsoLoader::GetCudaDriverDsoHandle;
using tsl::internal::DsoLoader::GetCudaRuntimeDsoHandle;
using tsl::internal::DsoLoader::GetCudnnDsoHandle;
using tsl::internal::DsoLoader::GetCufftDsoHandle;
using tsl::internal::DsoLoader::GetCuptiDsoHandle;
using tsl::internal::DsoLoader::GetCusolverDsoHandle;
using tsl::internal::DsoLoader::GetCusparseDsoHandle;
using tsl::internal::DsoLoader::GetHipDsoHandle;
using tsl::internal::DsoLoader::GetHipfftDsoHandle;
using tsl::internal::DsoLoader::GetHipsolverDsoHandle;
using tsl::internal::DsoLoader::GetHipsparseDsoHandle;
using tsl::internal::DsoLoader::GetMiopenDsoHandle;
using tsl::internal::DsoLoader::GetNvInferDsoHandle;
using tsl::internal::DsoLoader::GetNvInferPluginDsoHandle;
using tsl::internal::DsoLoader::GetRocblasDsoHandle;
using tsl::internal::DsoLoader::GetRocrandDsoHandle;
using tsl::internal::DsoLoader::GetRocsolverDsoHandle;
using tsl::internal::DsoLoader::GetRoctracerDsoHandle;
using tsl::internal::DsoLoader::MaybeTryDlopenGPULibraries;
using tsl::internal::DsoLoader::TryDlopenTensorRTLibraries;
// NOLINTEND(misc-unused-using-decls)
}  // namespace DsoLoader

namespace CachedDsoLoader {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::internal::CachedDsoLoader::GetCublasDsoHandle;
using tsl::internal::CachedDsoLoader::GetCublasLtDsoHandle;
using tsl::internal::CachedDsoLoader::GetCudaDriverDsoHandle;
using tsl::internal::CachedDsoLoader::GetCudaRuntimeDsoHandle;
using tsl::internal::CachedDsoLoader::GetCudnnDsoHandle;
using tsl::internal::CachedDsoLoader::GetCufftDsoHandle;
using tsl::internal::CachedDsoLoader::GetCuptiDsoHandle;
using tsl::internal::CachedDsoLoader::GetCusolverDsoHandle;
using tsl::internal::CachedDsoLoader::GetCusparseDsoHandle;
using tsl::internal::CachedDsoLoader::GetHipblasltDsoHandle;
using tsl::internal::CachedDsoLoader::GetHipDsoHandle;
using tsl::internal::CachedDsoLoader::GetHipfftDsoHandle;
using tsl::internal::CachedDsoLoader::GetHipsolverDsoHandle;
using tsl::internal::CachedDsoLoader::GetHipsparseDsoHandle;
using tsl::internal::CachedDsoLoader::GetMiopenDsoHandle;
using tsl::internal::CachedDsoLoader::GetRocblasDsoHandle;
using tsl::internal::CachedDsoLoader::GetRocrandDsoHandle;
using tsl::internal::CachedDsoLoader::GetRocsolverDsoHandle;
using tsl::internal::CachedDsoLoader::GetRoctracerDsoHandle;
// NOLINTEND(misc-unused-using-decls)
}  // namespace CachedDsoLoader

}  // namespace internal
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_DEFAULT_DSO_LOADER_H_
