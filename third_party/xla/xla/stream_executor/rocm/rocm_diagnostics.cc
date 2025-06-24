/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_diagnostics.h"

#include <dirent.h>
#include <limits.h>
#include <link.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/host_info.h"

namespace stream_executor {
namespace rocm {

std::string DriverVersionToString(DriverVersion version) {
  return absl::StrFormat("%d.%d.%d", std::get<0>(version), std::get<1>(version),
                         std::get<2>(version));
}

std::string DriverVersionStatusToString(absl::StatusOr<DriverVersion> version) {
  if (!version.ok()) {
    return version.status().ToString();
  }

  return DriverVersionToString(version.value());
}

absl::StatusOr<DriverVersion> StringToDriverVersion(const std::string& value) {
  std::vector<std::string> pieces = absl::StrSplit(value, '.');
  if (pieces.size() != 2 && pieces.size() != 3) {
    return absl::Status{absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("expected %%d.%%d or %%d.%%d.%%d form "
                                        "for driver version; got \"%s\"",
                                        value.c_str())};
  }

  int major;
  int minor;
  int patch = 0;
  if (!absl::SimpleAtoi(pieces[0], &major)) {
    return absl::Status{
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse major version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[0].c_str(), value.c_str())};
  }
  if (!absl::SimpleAtoi(pieces[1], &minor)) {
    return absl::Status{
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse minor version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[1].c_str(), value.c_str())};
  }
  if (pieces.size() == 3 && !absl::SimpleAtoi(pieces[2], &patch)) {
    return absl::Status{
        absl::StatusCode::kInvalidArgument,
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

std::string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
  return absl::StrCat("/dev/kfd", dev_node_ordinal);
}

void Diagnostician::LogDiagnosticInformation() {
  LOG(INFO) << "retrieving ROCM diagnostic information for host: "
            << tsl::port::Hostname();

  LogDriverVersionInformation();
}

/* static */ void Diagnostician::LogDriverVersionInformation() {
  LOG(INFO) << "hostname: " << tsl::port::Hostname();
  if (VLOG_IS_ON(1)) {
    const char* value = getenv("LD_LIBRARY_PATH");
    std::string library_path = value == nullptr ? "" : value;
    VLOG(1) << "LD_LIBRARY_PATH is: \"" << library_path << "\"";

    std::vector<std::string> pieces = absl::StrSplit(library_path, ':');
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
  absl::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "librocm reported version is: "
            << rocm::DriverVersionStatusToString(dso_version);

  absl::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << rocm::DriverVersionStatusToString(kernel_version);

  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
}

// Iterates through loaded DSOs with DlIteratePhdrCallback to find the
// driver-interfacing DSO version number. Returns it as a string.
absl::StatusOr<DriverVersion> Diagnostician::FindDsoVersion() {
  absl::StatusOr<DriverVersion> result{absl::Status{
      absl::StatusCode::kNotFound,
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
      std::string dso_version = dot + strlen(so_suffix);
      // TODO(b/22689637): Eliminate the explicit namespace if possible.
      auto stripped_dso_version = absl::StripSuffix(dso_version, ".ld64");
      auto result = static_cast<absl::StatusOr<DriverVersion>*>(data);
      *result = rocm::StringToDriverVersion(std::string(stripped_dso_version));
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_phdr, &result);

  return result;
}

absl::StatusOr<DriverVersion> Diagnostician::FindKernelModuleVersion(
    const std::string& driver_version_file_contents) {
  static const char* kDriverFilePrelude = "Kernel Module  ";
  size_t offset = driver_version_file_contents.find(kDriverFilePrelude);
  if (offset == std::string::npos) {
    return absl::Status{
        absl::StatusCode::kNotFound,
        absl::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\"")};
  }

  std::string version_and_rest = driver_version_file_contents.substr(
      offset + strlen(kDriverFilePrelude), std::string::npos);
  size_t space_index = version_and_rest.find(' ');
  auto kernel_version = version_and_rest.substr(0, space_index);
  // TODO(b/22689637): Eliminate the explicit namespace if possible.
  auto stripped_kernel_version = absl::StripSuffix(kernel_version, ".ld64");
  return rocm::StringToDriverVersion(std::string(stripped_kernel_version));
}

void Diagnostician::WarnOnDsoKernelMismatch(
    absl::StatusOr<DriverVersion> dso_version,
    absl::StatusOr<DriverVersion> kernel_version) {
  if (kernel_version.ok() && dso_version.ok() &&
      dso_version.value() == kernel_version.value()) {
    LOG(INFO) << "kernel version seems to match DSO: "
              << rocm::DriverVersionToString(kernel_version.value());
  } else {
    LOG(ERROR) << "kernel version "
               << rocm::DriverVersionStatusToString(kernel_version)
               << " does not match DSO version "
               << rocm::DriverVersionStatusToString(dso_version)
               << " -- cannot find working devices in this configuration";
  }
}

absl::StatusOr<DriverVersion> Diagnostician::FindKernelDriverVersion() {
  auto status = absl::Status{absl::StatusCode::kUnimplemented,
                             "kernel reported driver version not implemented"};
  return status;
}

}  // namespace rocm
}  // namespace stream_executor
