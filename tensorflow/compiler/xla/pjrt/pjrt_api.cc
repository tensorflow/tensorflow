/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"

#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

#if !defined(PLATFORM_WINDOWS)
#include <dlfcn.h>
#endif

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/errors.h"

namespace pjrt {

// The bool indicates whether this plugin has been initialized.
static auto* pjrt_apis =
    new absl::flat_hash_map<std::string, std::pair<const PJRT_Api*, bool>>{};

static std::string CanonicalizeDeviceType(absl::string_view device_type) {
  return absl::AsciiStrToLower(device_type);
}

xla::StatusOr<const PJRT_Api*> PjrtApi(absl::string_view device_type) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  auto iter = pjrt_apis->find(canonicalize_device_type);
  if (iter == pjrt_apis->end()) {
    return tsl::errors::NotFound("PJRT_Api not found for device type ",
                                 canonicalize_device_type);
  }
  return iter->second.first;
}

xla::Status SetPjrtApi(absl::string_view device_type, const PJRT_Api* api) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  if (auto iter = pjrt_apis->find(canonicalize_device_type);
      iter != pjrt_apis->end()) {
    return tsl::errors::AlreadyExists(
        "PJRT_Api already exists for device type ", canonicalize_device_type);
  }
  (*pjrt_apis)[canonicalize_device_type] =
      std::make_pair(api, /*is_initialized=*/false);
  LOG(INFO) << "PJRT_Api is set for device type " << canonicalize_device_type;
  // TODO(jieying): 592 is the size of PJRT_Api right after PJRT_Api_Version is
  // added. Remove this check after PJRT C API is stable and we assume all
  // plugins uses PJRT C API with PJRT_Api_Version.
  if (api->struct_size >= 592) {
    LOG(INFO) << "PJRT plugin for " << device_type << " has PJRT API version "
              << api->pjrt_api_version.major_version << "."
              << api->pjrt_api_version.minor_version
              << ". The framework PJRT API version is " << PJRT_API_MAJOR << "."
              << PJRT_API_MINOR << ".";
  }
  return tsl::OkStatus();
}

typedef const PJRT_Api* (*PjrtApiInitFn)();
xla::Status LoadPjrtPlugin(absl::string_view device_type,
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
  return SetPjrtApi(device_type, init_fn());
#endif
}

xla::StatusOr<bool> IsPjrtPluginInitialized(absl::string_view device_type) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  auto iter = pjrt_apis->find(canonicalize_device_type);
  if (iter == pjrt_apis->end()) {
    return absl::NotFoundError(absl::StrCat(
        "PJRT_Api not found for device type ", canonicalize_device_type,
        ". Call SetPjrtApi before calling IsPjrtPluginInitialized."));
  }
  return iter->second.second;
}

xla::Status InitializePjrtPlugin(absl::string_view device_type) {
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
  TF_RETURN_IF_ERROR(pjrt::CheckMatchingStructSizes(
      "PJRT_Api", PJRT_Api_STRUCT_SIZE, pjrt_api->struct_size));
  if (pjrt_api->struct_size >= 592 &&
      (pjrt_api->pjrt_api_version.major_version > 0 ||
       pjrt_api->pjrt_api_version.minor_version >= 13)) {
    PJRT_Plugin_Initialize_Args args;
    args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    args.priv = nullptr;
    RETURN_STATUS_IF_PJRT_ERROR(pjrt_api->PJRT_Plugin_Initialize(&args),
                                pjrt_api);
  }
  iter->second.second = true;
  return absl::OkStatus();
}

}  // namespace pjrt
