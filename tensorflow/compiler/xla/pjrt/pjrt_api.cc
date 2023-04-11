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

static auto* pjrt_apis =
    new absl::flat_hash_map<std::string, const PJRT_Api*>{};

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
  return iter->second;
}

xla::Status SetPjrtApi(absl::string_view device_type, const PJRT_Api* api) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  if (auto iter = pjrt_apis->find(canonicalize_device_type);
      iter != pjrt_apis->end()) {
    return tsl::errors::AlreadyExists(
        "PJRT_Api already exists for device type ", canonicalize_device_type);
  }
  (*pjrt_apis)[canonicalize_device_type] = api;
  LOG(INFO) << "PJRT_Api is set for device type " << canonicalize_device_type;
  return tsl::OkStatus();
}

xla::Status InitPjrtPlugin(PjrtApiInitFn init_fn,
                           absl::string_view device_type) {
  const PJRT_Api* pjrt_api = init_fn();
  TF_RETURN_IF_ERROR(pjrt::CheckMatchingStructSizes(
      "PJRT_Api", PJRT_Api_STRUCT_SIZE, pjrt_api->struct_size));
  return SetPjrtApi(device_type, pjrt_api);
}

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
  return InitPjrtPlugin(init_fn, device_type);
#endif
}

}  // namespace pjrt
