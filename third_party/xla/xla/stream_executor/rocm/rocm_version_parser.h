/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_VERSION_PARSER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_VERSION_PARSER_H_

#include "absl/status/statusor.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {

// Parses a ROCm version as returned by the `HIP_VERSION` macro, or
// API functions like `hipDriverGetVersion`, or `hipRuntimeGetVersion`.
absl::StatusOr<SemanticVersion> ParseRocmVersion(int rocm_version);
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_VERSION_PARSER_H_
