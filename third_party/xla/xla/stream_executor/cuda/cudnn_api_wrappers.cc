/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cudnn_api_wrappers.h"

#include <sys/resource.h>

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/library_types.h"
#include "third_party/gpus/cudnn/cudnn_version.h"
#if CUDNN_VERSION >= 90000
#include "third_party/gpus/cudnn/cudnn_graph.h"
#else
#include "third_party/gpus/cudnn/cudnn_ops_infer.h"
#endif
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace cuda {

static std::string CudnnStatusToString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
    default:
      return absl::StrCat("<unknown cudnn status: ", static_cast<int>(status),
                          ">");
  }
}

absl::Status ToStatus(cudnnStatus_t status, absl::string_view detail) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrCat("cuDNN error: ", CudnnStatusToString(status),
                   detail.empty() ? "" : absl::StrCat(":", detail)));
}

// cudnnGetProperty is forward declared as a weak symbol such that XLA-internal
// targets can depend on this wrapper library without linking the
// cudnn_plugin. This is mainly relevant for unit tests that use the CUDA
// executor which depends on this wrapper but should not introduce a dependency
// on cudnn.
extern "C" [[gnu::weak]] cudnnStatus_t cudnnGetProperty(
    libraryPropertyType type, int* value);

static libraryPropertyType ToLibraryPropertyType(CudnnProperty type) {
  switch (type) {
    case CudnnProperty::kMajorVersion:
      return MAJOR_VERSION;
    case CudnnProperty::kMinorVersion:
      return MINOR_VERSION;
    case CudnnProperty::kPatchLevelVersion:
      return PATCH_LEVEL;
  }
}

absl::StatusOr<int> GetCudnnProperty(CudnnProperty type) {
  if (!cudnnGetProperty) {
    return absl::NotFoundError("cuDNN is not linked into the application.");
  }
  int value{};
  TF_RETURN_IF_ERROR(
      ToStatus(cudnnGetProperty(ToLibraryPropertyType(type), &value)));
  return value;
}

absl::StatusOr<SemanticVersion> GetLoadedCudnnVersion() {
  TF_ASSIGN_OR_RETURN(int major,
                      GetCudnnProperty(CudnnProperty::kMajorVersion));
  TF_ASSIGN_OR_RETURN(int minor,
                      GetCudnnProperty(CudnnProperty::kMinorVersion));
  TF_ASSIGN_OR_RETURN(int patch,
                      GetCudnnProperty(CudnnProperty::kPatchLevelVersion));
  return SemanticVersion(major, minor, patch);
}

}  // namespace cuda
}  // namespace stream_executor
