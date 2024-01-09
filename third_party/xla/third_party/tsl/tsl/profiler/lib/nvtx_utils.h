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
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"

#if GOOGLE_CUDA
#include "nvtx3/nvToolsExt.h"
#else
// Some typedef to help build without NVTX.
typedef void* nvtxEventAttributes_t;
typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;
#endif

namespace tsl {
namespace profiler {
namespace nvtx {

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

// Two types of NVTX range annotation are supported, the older/simpler option
// is to use std::string and have the NVTX implementation copy a C-style
// string every time. The other option is to pass a struct implementing two
// methods:
//
//   std::string_view Title() const;
//   nvtxStringHandle_t NvtxRegisteredTitle() const;
//
// in which case NvtxRegisteredTitle() will be used when starting NVTX ranges,
// avoiding this string copy.
// The Title() method is needed because AnnotationStack::PushAnnotation(...) is
// the backend for some annotations when NVTX is not enabled, and it does not
// recognise registered strings. has_annotation_api_v<AnnotationType>
// distinguishes between the two types of annotation.
template <typename AnnotationType>
inline constexpr bool has_annotation_api_v =
    !std::is_same_v<AnnotationType, std::string>;

template <typename AnnotationType>
void RangePush(nvtxDomainHandle_t domain, const AnnotationType& annotation) {
#if GOOGLE_CUDA
  nvtxEventAttributes_t attrs{};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  if constexpr (has_annotation_api_v<std::decay_t<AnnotationType>>) {
    attrs.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attrs.message.registered = annotation.NvtxRegisteredTitle();
  } else {
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.message.ascii = annotation.c_str();
  }
  ::nvtxDomainRangePushEx(domain, &attrs);
#endif
}

}  // namespace nvtx
}  // namespace profiler
}  // namespace tsl
#endif  // TENSORFLOW_TSL_PROFILER_LIB_NVTX_UTILS_H_
