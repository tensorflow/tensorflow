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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TRACE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TRACE_UTILS_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

// Constants used as trace_viewer PID (device_id in trace_events.proto).
// PID 0 is unused.
// Support up to 500 accelerator devices.
constexpr uint32 kFirstDeviceId = 1;
constexpr uint32 kLastDeviceId = 500;
// Host threads are shown as a single fake device.
constexpr uint32 kHostThreadsDeviceId = kLastDeviceId + 1;

// Constants used as trace_viewer TID (resource_id in trace_events.proto).
constexpr int kThreadIdDerivedMin = 0xdeadbeef;
constexpr int kThreadIdStepInfo = kThreadIdDerivedMin;
constexpr int kThreadIdKernelLaunch = kThreadIdDerivedMin + 1;
constexpr int kThreadIdTfNameScope = kThreadIdDerivedMin + 2;
constexpr int kThreadIdTfOp = kThreadIdDerivedMin + 3;
constexpr int kThreadIdHloModule = kThreadIdDerivedMin + 4;
constexpr int kThreadIdHloOp = kThreadIdDerivedMin + 5;
constexpr int kThreadIdOverhead = kThreadIdDerivedMin + 6;
constexpr int kThreadIdSource = kThreadIdDerivedMin + 7;
constexpr int kThreadIdDerivedMax = kThreadIdSource;

static inline bool IsDerivedThreadId(int thread_id) {
  return thread_id >= kThreadIdDerivedMin && thread_id <= kThreadIdDerivedMax;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TRACE_UTILS_H_
