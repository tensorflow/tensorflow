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
#include <tuple>

#include "absl/status/statusor.h"

namespace stream_executor {
namespace cuda {

// e.g. DriverVersion{346, 3, 4}
using DriverVersion = std::tuple<int, int, int>;

// Converts a parsed driver version to string form.
std::string DriverVersionToString(DriverVersion version);

// Converts a parsed driver version or status value to natural string form.
std::string DriverVersionStatusToString(absl::StatusOr<DriverVersion> version);

// Converts a string of a form like "331.79" to a DriverVersion{331, 79}.
absl::StatusOr<DriverVersion> StringToDriverVersion(const std::string& value);

class Diagnostician {
 public:
  // Logs diagnostic information when CUDA appears to be misconfigured (e.g. is
  // not initializing).
  //
  // Note: if we're running on a machine that has no GPUs, we don't want to
  // produce very much log spew beyond saying, "looks like there's no CUDA
  // kernel
  // module running".
  //
  // Note: we use non-Google-File:: API here because we may be called before
  // InitGoogle has completed.
  static void LogDiagnosticInformation();

  // Given the driver version file contents, finds the kernel module version and
  // returns it as a string.
  //
  // This is solely used for more informative log messages when the user is
  // running on a machine that happens to have a libcuda/kernel driver mismatch.
  static absl::StatusOr<DriverVersion> FindKernelModuleVersion(
      const std::string& driver_version_file_contents);

  // Extracts the kernel driver version from the current host.
  static absl::StatusOr<DriverVersion> FindKernelDriverVersion();

  // Iterates through loaded DSOs with DlIteratePhdrCallback to find the
  // driver-interfacing DSO version number. Returns it as a string.
  static absl::StatusOr<DriverVersion> FindDsoVersion();

  // Logs information about the kernel driver version and userspace driver
  // library version.
  static void LogDriverVersionInformation();

 private:
  // Given the DSO version number and the driver version file contents, extracts
  // the driver version and compares, warning the user in the case of
  // incompatibility.
  //
  // This is solely used for more informative log messages when the user is
  // running on a machine that happens to have a libcuda/kernel driver mismatch.
  static void WarnOnDsoKernelMismatch(
      absl::StatusOr<DriverVersion> dso_version,
      absl::StatusOr<DriverVersion> kernel_version);

  static std::string GetDevNodePath(int dev_node_ordinal);

  Diagnostician(const Diagnostician&) = delete;
  void operator=(const Diagnostician&) = delete;
};

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DIAGNOSTICS_H_
