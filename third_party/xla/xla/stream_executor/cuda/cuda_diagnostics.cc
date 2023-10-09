/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/cuda/cuda_diagnostics.h"

#include <dirent.h>
#include <limits.h>
#include <link.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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
#include "tsl/platform/host_info.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace stream_executor {
namespace cuda {

std::string DriverVersionToString(DriverVersion version) {
  return absl::StrFormat("%d.%d.%d", std::get<0>(version), std::get<1>(version),
                         std::get<2>(version));
}

std::string DriverVersionStatusToString(tsl::StatusOr<DriverVersion> version) {
  if (!version.ok()) {
    return version.status().ToString();
  }

  return DriverVersionToString(version.value());
}

tsl::StatusOr<DriverVersion> StringToDriverVersion(const std::string &value) {
  std::vector<std::string> pieces = absl::StrSplit(value, '.');
  if (pieces.size() < 2 || pieces.size() > 4) {
    return tsl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "expected %%d.%%d, %%d.%%d.%%d, or %%d.%%d.%%d.%%d form "
            "for driver version; got \"%s\"",
            value.c_str()));
  }

  int major;
  int minor;
  int patch = 0;
  if (!absl::SimpleAtoi(pieces[0], &major)) {
    return tsl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse major version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[0], value));
  }
  if (!absl::SimpleAtoi(pieces[1], &minor)) {
    return tsl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse minor version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[1].c_str(), value.c_str()));
  }
  if (pieces.size() == 3 && !absl::SimpleAtoi(pieces[2], &patch)) {
    return tsl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("could not parse patch version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[2], value));
  }

  DriverVersion result{major, minor, patch};
  VLOG(2) << "version string \"" << value << "\" made value "
          << DriverVersionToString(result);
  return result;
}

}  // namespace cuda
}  // namespace stream_executor

namespace stream_executor {
namespace gpu {

static const char *kDriverVersionPath = "/proc/driver/nvidia/version";

// -- class Diagnostician

std::string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
  return absl::StrCat("/dev/nvidia", dev_node_ordinal);
}

void Diagnostician::LogDiagnosticInformation() {
  if (access(kDriverVersionPath, F_OK) != 0) {
    VLOG(1) << "kernel driver does not appear to be running on this host "
            << "(" << tsl::port::Hostname() << "): "
            << "/proc/driver/nvidia/version does not exist";
    return;
  }
  auto dev0_path = GetDevNodePath(0);
  if (access(dev0_path.c_str(), F_OK) != 0) {
    VLOG(1) << "no NVIDIA GPU device is present: " << dev0_path
            << " does not exist";
    return;
  }

  LOG(INFO) << "retrieving CUDA diagnostic information for host: "
            << tsl::port::Hostname();

  LogDriverVersionInformation();
}

/* static */ void Diagnostician::LogDriverVersionInformation() {
  LOG(INFO) << "hostname: " << tsl::port::Hostname();
  if (VLOG_IS_ON(1)) {
    const char *value = getenv("LD_LIBRARY_PATH");
    std::string library_path = value == nullptr ? "" : value;
    VLOG(1) << "LD_LIBRARY_PATH is: \"" << library_path << "\"";

    std::vector<std::string> pieces = absl::StrSplit(library_path, ':');
    for (const auto &piece : pieces) {
      if (piece.empty()) {
        continue;
      }
      DIR *dir = opendir(piece.c_str());
      if (dir == nullptr) {
        VLOG(1) << "could not open \"" << piece << "\"";
        continue;
      }
      while (dirent *entity = readdir(dir)) {
        VLOG(1) << piece << " :: " << entity->d_name;
      }
      closedir(dir);
    }
  }
  tsl::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "libcuda reported version is: "
            << cuda::DriverVersionStatusToString(dso_version);

  tsl::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << cuda::DriverVersionStatusToString(kernel_version);

  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
}

// Iterates through loaded DSOs with DlIteratePhdrCallback to find the
// driver-interfacing DSO version number. Returns it as a string.
tsl::StatusOr<DriverVersion> Diagnostician::FindDsoVersion() {
  tsl::StatusOr<DriverVersion> result(tsl::Status(
      absl::StatusCode::kNotFound,
      "was unable to find libcuda.so DSO loaded into this program"));

#if !defined(ANDROID_TEGRA)
  // Callback used when iterating through DSOs. Looks for the driver-interfacing
  // DSO and yields its version number into the callback data, when found.
  auto iterate_phdr = [](struct dl_phdr_info *info, size_t size,
                         void *data) -> int {
    if (strstr(info->dlpi_name, "libcuda.so.1")) {
      VLOG(1) << "found DLL info with name: " << info->dlpi_name;
      char resolved_path[PATH_MAX] = {0};
      if (realpath(info->dlpi_name, resolved_path) == nullptr) {
        return 0;
      }
      VLOG(1) << "found DLL info with resolved path: " << resolved_path;
      const char *slash = rindex(resolved_path, '/');
      if (slash == nullptr) {
        return 0;
      }
      const char *so_suffix = ".so.";
      const char *dot = strstr(slash, so_suffix);
      if (dot == nullptr) {
        return 0;
      }
      std::string dso_version = dot + strlen(so_suffix);
      // TODO(b/22689637): Eliminate the explicit namespace if possible.
      auto stripped_dso_version = absl::StripSuffix(dso_version, ".ld64");
      auto result = static_cast<tsl::StatusOr<DriverVersion> *>(data);
      *result = cuda::StringToDriverVersion(std::string(stripped_dso_version));
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_phdr, &result);
#endif

  return result;
}

tsl::StatusOr<DriverVersion> Diagnostician::FindKernelModuleVersion(
    const std::string &driver_version_file_contents) {
  static const char *kDriverFilePrelude = "Kernel Module  ";
  size_t offset = driver_version_file_contents.find(kDriverFilePrelude);
  if (offset == std::string::npos) {
    return tsl::Status(
        absl::StatusCode::kNotFound,
        absl::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\""));
  }

  std::string version_and_rest = driver_version_file_contents.substr(
      offset + strlen(kDriverFilePrelude), std::string::npos);
  size_t space_index = version_and_rest.find(' ');
  auto kernel_version = version_and_rest.substr(0, space_index);
  // TODO(b/22689637): Eliminate the explicit namespace if possible.
  auto stripped_kernel_version = absl::StripSuffix(kernel_version, ".ld64");
  return cuda::StringToDriverVersion(std::string(stripped_kernel_version));
}

void Diagnostician::WarnOnDsoKernelMismatch(
    tsl::StatusOr<DriverVersion> dso_version,
    tsl::StatusOr<DriverVersion> kernel_version) {
  if (kernel_version.ok() && dso_version.ok() &&
      dso_version.value() == kernel_version.value()) {
    LOG(INFO) << "kernel version seems to match DSO: "
              << cuda::DriverVersionToString(kernel_version.value());
  } else {
    LOG(ERROR) << "kernel version "
               << cuda::DriverVersionStatusToString(kernel_version)
               << " does not match DSO version "
               << cuda::DriverVersionStatusToString(dso_version)
               << " -- cannot find working devices in this configuration";
  }
}

tsl::StatusOr<DriverVersion> Diagnostician::FindKernelDriverVersion() {
  FILE *driver_version_file = fopen(kDriverVersionPath, "r");
  if (driver_version_file == nullptr) {
    return tsl::Status(
        absl::StatusCode::kPermissionDenied,
        absl::StrCat("could not open driver version path for reading: ",
                     kDriverVersionPath));
  }

  static const int kContentsSize = 1024;
  absl::InlinedVector<char, 4> contents(kContentsSize);
  size_t retcode =
      fread(contents.begin(), 1, kContentsSize - 2, driver_version_file);
  if (retcode < kContentsSize - 1) {
    contents[retcode] = '\0';
  }
  contents[kContentsSize - 1] = '\0';

  if (retcode != 0) {
    VLOG(1) << "driver version file contents: \"\"\"" << contents.begin()
            << "\"\"\"";
    fclose(driver_version_file);
    return FindKernelModuleVersion(contents.begin());
  }

  auto status = tsl::Status(
      absl::StatusCode::kInternal,
      absl::StrCat(
          "failed to read driver version file contents: ", kDriverVersionPath,
          "; ferror: ", ferror(driver_version_file)));
  fclose(driver_version_file);
  return status;
}

}  // namespace gpu
}  // namespace stream_executor
