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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDNN_API_WRAPPERS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDNN_API_WRAPPERS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cudnn/cudnn_graph.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {
namespace cuda {

// This enum holds all the properties that can be queried from cuDNN.
enum class CudnnProperty {
  kMajorVersion,
  kMinorVersion,
  kPatchLevelVersion,
};

// Returns the value of the given cuDNN property - or an error if cuDNN is not
// loaded.
absl::StatusOr<int> GetCudnnProperty(CudnnProperty type);

// Returns the version of the loaded cuDNN library - or an error if cuDNN is not
// loaded.
absl::StatusOr<SemanticVersion> GetLoadedCudnnVersion();

// Converts a cuDNN status code to an absl::Status of type
// absl::StatusCode::kInternal. If `detail` is non-empty, it is appended to the
// error message.
absl::Status ToStatus(cudnnStatus_t status, absl::string_view detail = "");

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDNN_API_WRAPPERS_H_
