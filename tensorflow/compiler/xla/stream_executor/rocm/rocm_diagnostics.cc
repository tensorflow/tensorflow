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

#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_diagnostics.h"

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
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/host_info.h"

namespace stream_executor {
namespace rocm {

string DriverVersionToString(DriverVersion version) {
  return absl::StrFormat("%d.%d.%d", std::get<0>(version), std::get<1>(version),
                         std::get<2>(version));
}

string DriverVersionStatusToString(tsl::StatusOr<DriverVersion> version) {
  if (!version.ok()) {
    return version.status().ToString();
  }

  return DriverVersionToString(version.value());
}

tsl::StatusOr<DriverVersion> StringToDriverVersion(const string& value) {
  std::vector<string> pieces = absl::StrSplit(value, '.');
  if (pieces.size() != 2 && pieces.size() != 3) {
<<<<<<< HEAD
    return tsl::Status(
                    absl::StatusCode::kInvalidArgument,
                    absl::StrFormat("expected %%d.%%d or %%d.%%d.%%d form "
=======
    return tsl::Status{absl::StatusCode::kInvalidArgument,
                       absl::StrFormat("expected %%d.%%d or %%d.%%d.%%d form "
>>>>>>> upstream/master
                                       "for driver version; got \"%s\"",
                                       value.c_str()));
  }

  int major;
  int minor;
  int patch = 0;
  if (!absl::SimpleAtoi(pieces[0], &major)) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse major version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[0].c_str(), value.c_str()));
  }
  if (!absl::SimpleAtoi(pieces[1], &minor)) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse minor version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[1].c_str(), value.c_str()));
  }
  if (pieces.size() == 3 && !absl::SimpleAtoi(pieces[2], &patch)) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse patch version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[2].c_str(), value.c_str()));
  }

  DriverVersion result{major, minor, patch};
  VLOG(2) << "version string \"" << value << "\" made value "
          << DriverVersionToString(result);
  return result;
}

}  // namespace rocm
}  // namespace stream_executor

namespace stream_executor {
namespace gpu {

// -- class Diagnostician

string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
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
    string library_path = value == nullptr ? "" : value;
    VLOG(1) << "LD_LIBRARY_PATH is: \"" << library_path << "\"";

    std::vector<string> pieces = absl::StrSplit(library_path, ':');
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
  tsl::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "librocm reported version is: "
            << rocm::DriverVersionStatusToString(dso_version);

  tsl::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << rocm::DriverVersionStatusToString(kernel_version);

  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
}

// Iterates through loaded DSOs with DlIteratePhdrCallback to find the
// driver-interfacing DSO version number. Returns it as a string.
tsl::StatusOr<DriverVersion> Diagnostician::FindDsoVersion() {
<<<<<<< HEAD
  tsl::StatusOr<DriverVersion> result(tsl::Status(
      absl::StatusCode::kNotFound,
      "was unable to find librocm.so DSO loaded into this program"));
=======
  tsl::StatusOr<DriverVersion> result{tsl::Status{
      absl::StatusCode::kNotFound,
      "was unable to find librocm.so DSO loaded into this program"}};
>>>>>>> upstream/master

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
      auto stripped_dso_version = absl::StripSuffix(dso_version, ".ld64");
      auto result = static_cast<tsl::StatusOr<DriverVersion>*>(data);
      *result = rocm::StringToDriverVersion(string(stripped_dso_version));
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_phdr, &result);

  return result;
}

tsl::StatusOr<DriverVersion> Diagnostician::FindKernelModuleVersion(
    const string& driver_version_file_contents) {
  static const char* kDriverFilePrelude = "Kernel Module  ";
  size_t offset = driver_version_file_contents.find(kDriverFilePrelude);
  if (offset == string::npos) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kNotFound,
        absl::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\""));
  }

  string version_and_rest = driver_version_file_contents.substr(
      offset + strlen(kDriverFilePrelude), string::npos);
  size_t space_index = version_and_rest.find(" ");
  auto kernel_version = version_and_rest.substr(0, space_index);
  // TODO(b/22689637): Eliminate the explicit namespace if possible.
  auto stripped_kernel_version = absl::StripSuffix(kernel_version, ".ld64");
  return rocm::StringToDriverVersion(string(stripped_kernel_version));
}

void Diagnostician::WarnOnDsoKernelMismatch(
    tsl::StatusOr<DriverVersion> dso_version,
    tsl::StatusOr<DriverVersion> kernel_version) {
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

tsl::StatusOr<DriverVersion> Diagnostician::FindKernelDriverVersion() {
<<<<<<< HEAD
  auto status = tsl::Status(absl::StatusCode::kUnimplemented,
                            "kernel reported driver version not implemented");
=======
  auto status = tsl::Status{absl::StatusCode::kUnimplemented,
                            "kernel reported driver version not implemented"};
>>>>>>> upstream/master
  return status;
}

}  // namespace gpu
}  // namespace stream_executor
