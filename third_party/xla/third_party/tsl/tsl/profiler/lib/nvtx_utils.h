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
#include <string>

#if GOOGLE_CUDA
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtPayload.h"
#else
// Some typedef to help build without NVTX.
typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;
#endif

namespace tsl {
namespace profiler {

// A helper function that return the domains to use if NVTX profiling
// is enabled.
inline std::optional<nvtxDomainHandle_t> GetNVTXDomain() {
#if GOOGLE_CUDA
  static nvtxDomainHandle_t domain = nvtxDomainCreateA("TSL");
  if (domain != nullptr) return domain;
#endif
  return std::nullopt;
}

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
inline bool RangesEnabled() {
#if GOOGLE_CUDA
  return GetNVTXDomain().has_value();
#else
  return false;
#endif
}

// Older/simpler version; NVTX implementation copies a C-style string each time
inline void RangePush(nvtxDomainHandle_t domain, const char* ascii) {
#if GOOGLE_CUDA
  nvtxEventAttributes_t attrs{};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attrs.message.ascii = ascii;
  ::nvtxDomainRangePushEx(domain, &attrs);
#endif
}
inline void RangePush(nvtxDomainHandle_t domain, const std::string& str) {
  RangePush(domain, str.c_str());
}

// More powerful version: pass a registered string instead of a C-style string,
// and attach a generic payload. The Annotation type must implement a method
// called NvtxSchemaId() that allows the NVTX backend to interpret the payload.
template <typename Annotation>
void RangePush(nvtxDomainHandle_t domain, nvtxStringHandle_t handle,
               const Annotation& annotation) {
#if GOOGLE_CUDA
  nvtxEventAttributes_t attrs{};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attrs.message.registered = handle;
  NVTX_PAYLOAD_EVTATTR_SET(attrs, annotation.NvtxSchemaId(), &annotation,
                           sizeof(Annotation));
  ::nvtxDomainRangePushEx(domain, &attrs);
#endif
}

}  // namespace profiler
}  // namespace tsl
#endif  // TENSORFLOW_TSL_PROFILER_LIB_NVTX_UTILS_H_
