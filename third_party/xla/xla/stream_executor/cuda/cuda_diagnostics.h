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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DIAGNOSTICS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DIAGNOSTICS_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_diagnostics.h"

namespace stream_executor {
namespace cuda {

// e.g. DriverVersion{346, 3, 4}
using DriverVersion = gpu::DriverVersion;

// Converts a parsed driver version to string form.
std::string DriverVersionToString(DriverVersion version);

// Converts a parsed driver version or status value to natural string form.
std::string DriverVersionStatusToString(absl::StatusOr<DriverVersion> version);

// Converts a string of a form like "331.79" to a DriverVersion{331, 79}.
absl::StatusOr<DriverVersion> StringToDriverVersion(const std::string& value);

using Diagnostician = gpu::Diagnostician;

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DIAGNOSTICS_H_
