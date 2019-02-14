/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <dirent.h>

#include <limits.h>
#include <link.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/gpu/gpu_diagnostics.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {
namespace gpu {

string DriverVersionToString(DriverVersion version) {
  return absl::StrFormat("%d.%d.%d", std::get<0>(version), std::get<1>(version),
                         std::get<2>(version));
}

string DriverVersionStatusToString(port::StatusOr<DriverVersion> version) {
  if (!version.ok()) {
    return version.status().ToString();
  }

  return DriverVersionToString(version.ValueOrDie());
}

port::StatusOr<DriverVersion> StringToDriverVersion(const string& value) {
  std::vector<string> pieces = port::Split(value, '.');
  if (pieces.size() != 2 && pieces.size() != 3) {
    return port::Status{port::error::INVALID_ARGUMENT,
                        absl::StrFormat("expected %%d.%%d or %%d.%%d.%%d form "
                                        "for driver version; got \"%s\"",
                                        value.c_str())};
  }

  int major;
  int minor;
  int patch = 0;
  if (!port::safe_strto32(pieces[0], &major)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse major version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[0].c_str(), value.c_str())};
  }
  if (!port::safe_strto32(pieces[1], &minor)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse minor version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[1].c_str(), value.c_str())};
  }
  if (pieces.size() == 3 && !port::safe_strto32(pieces[2], &patch)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse patch version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[2].c_str(), value.c_str())};
  }

  DriverVersion result{major, minor, patch};
  VLOG(2) << "version string \"" << value << "\" made value "
          << DriverVersionToString(result);
  return result;
}

// -- class Diagnostician

string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
  return absl::StrCat("/dev/kfd", dev_node_ordinal);
}

void Diagnostician::LogDiagnosticInformation() {
  LOG(INFO) << "retrieving ROCM diagnostic information for host: "
            << port::Hostname();

  LogDriverVersionInformation();
}

/* static */ void Diagnostician::LogDriverVersionInformation() {
  LOG(INFO) << "hostname: " << port::Hostname();
  if (VLOG_IS_ON(1)) {
    const char* value = getenv("LD_LIBRARY_PATH");
    string library_path = value == nullptr ? "" : value;
    VLOG(1) << "LD_LIBRARY_PATH is: \"" << library_path << "\"";

    std::vector<string> pieces = port::Split(library_path, ':');
    for (const auto& piece : pieces) {
      if (piece.empty()) {
        continue;
      }
      DIR* dir = opendir(piece.c_str());
      if (dir == nullptr) {
        VLOG(1) << "could not open \"" << piece << "\"";
        continue;
      }
      while (dirent* entity = readdir(dir)) {
        VLOG(1) << piece << " :: " << entity->d_name;
      }
      closedir(dir);
    }
  }
  port::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "librocm reported version is: "
            << DriverVersionStatusToString(dso_version);

  port::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << DriverVersionStatusToString(kernel_version);

  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
}

// Iterates through loaded DSOs with DlIteratePhdrCallback to find the
// driver-interfacing DSO version number. Returns it as a string.
port::StatusOr<DriverVersion> Diagnostician::FindDsoVersion() {
  port::StatusOr<DriverVersion> result{port::Status{
      port::error::NOT_FOUND,
      "was unable to find librocm.so DSO loaded into this program"}};

  // Callback used when iterating through DSOs. Looks for the driver-interfacing
  // DSO and yields its version number into the callback data, when found.
  auto iterate_phdr = [](struct dl_phdr_info* info, size_t size,
                         void* data) -> int {
    if (strstr(info->dlpi_name, "librocm.so.1")) {
      VLOG(1) << "found DLL info with name: " << info->dlpi_name;
      char resolved_path[PATH_MAX] = {0};
      if (realpath(info->dlpi_name, resolved_path) == nullptr) {
        return 0;
      }
      VLOG(1) << "found DLL info with resolved path: " << resolved_path;
      const char* slash = rindex(resolved_path, '/');
      if (slash == nullptr) {
        return 0;
      }
      const char* so_suffix = ".so.";
      const char* dot = strstr(slash, so_suffix);
      if (dot == nullptr) {
        return 0;
      }
      string dso_version = dot + strlen(so_suffix);
      // TODO(b/22689637): Eliminate the explicit namespace if possible.
      auto stripped_dso_version = port::StripSuffixString(dso_version, ".ld64");
      auto result = static_cast<port::StatusOr<DriverVersion>*>(data);
      *result = StringToDriverVersion(stripped_dso_version);
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_phdr, &result);

  return result;
}

port::StatusOr<DriverVersion> Diagnostician::FindKernelModuleVersion(
    const string& driver_version_file_contents) {
  static const char* kDriverFilePrelude = "Kernel Module  ";
  size_t offset = driver_version_file_contents.find(kDriverFilePrelude);
  if (offset == string::npos) {
    return port::Status{
        port::error::NOT_FOUND,
        absl::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\"")};
  }

  string version_and_rest = driver_version_file_contents.substr(
      offset + strlen(kDriverFilePrelude), string::npos);
  size_t space_index = version_and_rest.find(" ");
  auto kernel_version = version_and_rest.substr(0, space_index);
  // TODO(b/22689637): Eliminate the explicit namespace if possible.
  auto stripped_kernel_version =
      port::StripSuffixString(kernel_version, ".ld64");
  return StringToDriverVersion(stripped_kernel_version);
}

void Diagnostician::WarnOnDsoKernelMismatch(
    port::StatusOr<DriverVersion> dso_version,
    port::StatusOr<DriverVersion> kernel_version) {
  if (kernel_version.ok() && dso_version.ok() &&
      dso_version.ValueOrDie() == kernel_version.ValueOrDie()) {
    LOG(INFO) << "kernel version seems to match DSO: "
              << DriverVersionToString(kernel_version.ValueOrDie());
  } else {
    LOG(ERROR) << "kernel version "
               << DriverVersionStatusToString(kernel_version)
               << " does not match DSO version "
               << DriverVersionStatusToString(dso_version)
               << " -- cannot find working devices in this configuration";
  }
}

port::StatusOr<DriverVersion> Diagnostician::FindKernelDriverVersion() {
  auto status = port::Status{port::error::UNIMPLEMENTED,
                             "kernel reported driver version not implemented"};
  return status;
}

}  // namespace gpu
}  // namespace stream_executor
