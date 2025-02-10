/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PROFILER_UTILS_TIMESTAMP_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_TIMESTAMP_UTILS_H_

#include <cstdint>

#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Add metadata regarding profile start_time and stop_time to xspace.
// This function won't have an effect if either of the timestamps is zero.
void SetSessionTimestamps(uint64_t start_walltime_ns, uint64_t stop_walltime_ns,
                          tensorflow::profiler::XSpace& space);
}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_TIMESTAMP_UTILS_H_
