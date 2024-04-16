/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_api.h"

#include <cstdlib>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#if !defined(PLATFORM_WINDOWS)
#include <dlfcn.h>
#endif

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace pjrt {

// This is the minimum supported PJRT API minor version to ensure a forward
// compatibility window of at least 12 weeks. Please see the changelog of PJRT C
// API https://github.com/openxla/xla/blob/main/xla/pjrt/c/CHANGELOG.md for the
// date of each version and PJRT C API compatibility policy
// (https://docs.google.com/document/d/1TKB5NyGtdzrpgw5mpyFjVAhJjpSNdF31T6pjPl_UT2o/edit).
// The forward compatibility is controlled by the ENABLE_PJRT_COMPATIBILITY env
// variable.
constexpr int kMinPjRtMinor = 29;

// The bool indicates whether this plugin has been initialized.
static auto* pjrt_apis =
    new absl::flat_hash_map<std::string, std::pair<const PJRT_Api*, bool>>{};

static std::string CanonicalizeDeviceType(absl::string_view device_type) {
  return absl::AsciiStrToLower(device_type);
}

absl::StatusOr<const PJRT_Api*> PjrtApi(absl::string_view device_type) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  auto iter = pjrt_apis->find(canonicalize_device_type);
  if (iter == pjrt_apis->end()) {
    return tsl::errors::NotFound("PJRT_Api not found for device type ",
                                 canonicalize_device_type);
  }
  return iter->second.first;
}

absl::Status SetPjrtApi(absl::string_view device_type, const PJRT_Api* api) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  if (auto iter = pjrt_apis->find(canonicalize_device_type);
      iter != pjrt_apis->end()) {
    return tsl::errors::AlreadyExists(
        "PJRT_Api already exists for device type ", canonicalize_device_type);
  }
  (*pjrt_apis)[canonicalize_device_type] =
      std::make_pair(api, /*is_initialized=*/false);
  LOG(INFO) << "PJRT_Api is set for device type " << canonicalize_device_type;
  return absl::OkStatus();
}

typedef const PJRT_Api* (*PjrtApiInitFn)();
absl::StatusOr<const PJRT_Api*> LoadPjrtPlugin(absl::string_view device_type,
                                               absl::string_view library_path) {
#ifdef PLATFORM_WINDOWS
  return tsl::errors::Unimplemented(
      "LoadPjrtPlugin is not implemented on windows yet.");
#else
  void* library = dlopen(library_path.data(), RTLD_LAZY);
  if (library == nullptr) {
    return tsl::errors::Internal("Failed to open ", library_path, ": ",
                                 dlerror());
  }
  PjrtApiInitFn init_fn;
  *reinterpret_cast<void**>(&init_fn) = dlsym(library, "GetPjrtApi");
  if (init_fn == nullptr) {
    return tsl::errors::NotFound("GetPjrtApi not found in ", library_path);
  }
  LOG(INFO) << "GetPjrtApi was found for " << device_type << " at "
            << library_path;
  const PJRT_Api* api = init_fn();
  TF_RETURN_IF_ERROR(SetPjrtApi(device_type, api));
  return api;
#endif
}

absl::StatusOr<bool> IsPjrtPluginInitialized(absl::string_view device_type) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  auto iter = pjrt_apis->find(canonicalize_device_type);
  if (iter == pjrt_apis->end()) {
    return absl::NotFoundError(absl::StrCat(
        "PJRT_Api not found for device type ", canonicalize_device_type,
        ". Call SetPjrtApi before calling IsPjrtPluginInitialized."));
  }
  return iter->second.second;
}

static bool IsPjRtCompatibilityEnabled() {
  const char* val = getenv("ENABLE_PJRT_COMPATIBILITY");
  if (val == nullptr) {
    return false;
  }
  bool enabled = false;
  if (!absl::SimpleAtob(val, &enabled)) {
    return false;
  }
  return enabled;
}

absl::Status InitializePjrtPlugin(absl::string_view device_type) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  auto iter = pjrt_apis->find(canonicalize_device_type);
  if (iter == pjrt_apis->end()) {
    return absl::NotFoundError(absl::StrCat(
        "PJRT_Api not found for device type ", canonicalize_device_type,
        ". Call SetPjrtApi before calling IsPjrtPluginInitialized."));
  }
  if (iter->second.second) {
    return absl::InvalidArgumentError(
        absl::StrCat("InitializePjrtPlugin requested to run on already "
                     "initialized plugin ",
                     canonicalize_device_type));
  }
  const PJRT_Api* pjrt_api = iter->second.first;
  LOG(INFO) << "The PJRT plugin has PJRT API version "
            << pjrt_api->pjrt_api_version.major_version << "."
            << pjrt_api->pjrt_api_version.minor_version
            << ". The framework PJRT API version is " << PJRT_API_MAJOR << "."
            << PJRT_API_MINOR << ".";
  // TODO(b/305096260): improve the error message to something that the user can
  // act upon.
  if (IsPjRtCompatibilityEnabled()) {
    if (pjrt_api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mismatched PJRT plugin PJRT API major version (",
          pjrt_api->pjrt_api_version.major_version,
          ") and framework PJRT API major version ", PJRT_API_MAJOR, ")."));
    }
    if (pjrt_api->pjrt_api_version.minor_version < kMinPjRtMinor) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Plugin PJRT API version ", pjrt_api->pjrt_api_version.major_version,
          ".", pjrt_api->pjrt_api_version.minor_version,
          " is older than the minimum supported version ", PJRT_API_MAJOR, ".",
          kMinPjRtMinor));
    }
  } else {
    if (pjrt_api->pjrt_api_version.major_version != PJRT_API_MAJOR ||
        pjrt_api->pjrt_api_version.minor_version != PJRT_API_MINOR) {
      return absl::InvalidArgumentError(
          absl::StrCat("Mismatched PJRT plugin PJRT API version (",
                       pjrt_api->pjrt_api_version.major_version, ".",
                       pjrt_api->pjrt_api_version.minor_version,
                       ") and framework PJRT API version ", PJRT_API_MAJOR, ".",
                       PJRT_API_MINOR, ")."));
    }
  }
  PJRT_Plugin_Initialize_Args args;
  args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(pjrt_api->PJRT_Plugin_Initialize(&args),
                              pjrt_api);
  iter->second.second = true;
  return absl::OkStatus();
}

}  // namespace pjrt
