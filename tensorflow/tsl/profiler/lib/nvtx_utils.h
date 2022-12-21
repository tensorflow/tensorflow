/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PROFILER_LIB_NVTX_UTILS_H_
#define TENSORFLOW_TSL_PROFILER_LIB_NVTX_UTILS_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/macros.h"

#if GOOGLE_CUDA
#include "nvtx3/nvToolsExt.h"
#endif

namespace tsl {
namespace profiler {
namespace nvtx {

// Some typedef to help build without NVTX.
#if !GOOGLE_CUDA
typedef void* nvtxEventAttributes_t;
typedef void* nvtxDomainHandle_t;
#endif

// A helper function that return the domains to use if NVTX profiling
// is enabled.
inline std::optional<nvtxDomainHandle_t> GetNVTXDomain() {
#if GOOGLE_CUDA
  static nvtxDomainHandle_t domain;
  static bool is_enabled = [] {
    bool _is_enabled = false;
    // Force NVTX marker if a tool triggered the profiler.
    domain = nvtxDomainCreateA("TSL");
    if (domain) {
      _is_enabled = true;
    }
    VLOG(1) << "Is NVTX marker enabled? " << _is_enabled;
    return _is_enabled;
  }();
  if (is_enabled) return domain;
#endif
  return {};
}

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
inline bool RangesEnabled() {
#if GOOGLE_CUDA
  return GetNVTXDomain().has_value();
#else
  return false;
#endif
}

// Note: The memory backing msg must persist until the result of this function
// has been consumed by an NVTX API.
inline void MakeAttributes(const char* msg, nvtxEventAttributes_t* result) {
  *result = {0};
#if GOOGLE_CUDA
  result->version = NVTX_VERSION;
  result->size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  result->messageType = NVTX_MESSAGE_TYPE_ASCII;
  result->message.ascii = msg;
#endif
}

}  // namespace nvtx
}  // namespace profiler
}  // namespace tsl
#endif  // TENSORFLOW_TSL_PROFILER_LIB_NVTX_UTILS_H_
