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

#include <stddef.h>

#include <cstdint>
#include <string>

namespace tsl::profiler {
struct String;
// Opaque handle to a string that has been pre-registered with the profiler/NVTX
// implementation
using StringHandle = String*;

struct ProfilerDomain;
// Opaque handle to a domain in the profiler/NVTX implementation
using ProfilerDomainHandle = ProfilerDomain*;

// Get the "TSL" domain if NVTX profiling is enabled, otherwise null
ProfilerDomainHandle DefaultProfilerDomain();

// Register a string with the profiler/NVTX implementation for faster use
StringHandle RegisterString(ProfilerDomainHandle, const std::string&);

// End a range that was created on this thread by RangePush
void RangePop(ProfilerDomainHandle);

// Older/simpler version; NVTX implementation copies a C-style string each time
void RangePush(ProfilerDomainHandle domain, const char*);
inline void RangePush(ProfilerDomainHandle domain, const std::string& str) {
  RangePush(domain, str.c_str());
}

namespace detail {
void RangePush(ProfilerDomainHandle domain, StringHandle title,
               uint64_t schema_id, const void* payload, size_t payload_size);
}

// More powerful version: pass a registered string instead of a C-style
// string, and attach a generic payload. The Annotation type must implement a
// method called NvtxSchemaId() that allows the NVTX backend to interpret the
// payload.
template <typename Annotation>
void RangePush(ProfilerDomainHandle domain, StringHandle title,
               const Annotation& annotation) {
  return detail::RangePush(domain, title, annotation.NvtxSchemaId(),
                           &annotation, sizeof(Annotation));
}

// Register the schema of a custom payload type, for use with the more powerful
// version of RangePush
uint64_t RegisterSchema(ProfilerDomainHandle domain, const void* schemaAttr);
}  // namespace tsl::profiler
#endif  // TENSORFLOW_TSL_PROFILER_LIB_NVTX_UTILS_H_
