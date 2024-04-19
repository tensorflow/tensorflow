/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_COMPILATION_ENVIRONMENT_H_
#define XLA_SERVICE_GPU_COMPILATION_ENVIRONMENT_H_
#include <string>
#include <vector>

#include "xla/statusor.h"
#include "xla/xla.pb.h"

namespace xla {

absl::StatusOr<GpuCompilationEnvironment> CreateGpuCompEnvFromFlagStrings(
    std::vector<std::string>& flags, bool strict);

absl::StatusOr<GpuCompilationEnvironment> CreateGpuCompEnvFromEnvVar();

GpuCompilationEnvironment CreateGpuCompEnvWithDefaultValues();

// Returns non-OK status if XLA_FLAGS env var has malformed values or
// if it has conflict with the GpuCompilationEnvironment proto
Status InitializeMissingFieldsFromXLAFlags(GpuCompilationEnvironment& env);

}  // namespace xla
#endif  // XLA_SERVICE_GPU_COMPILATION_ENVIRONMENT_H_
