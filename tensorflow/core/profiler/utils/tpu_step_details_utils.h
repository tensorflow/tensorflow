/* Copyright 2024 The OpenXLA Authors.

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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TPU_STEP_DETAILS_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TPU_STEP_DETAILS_UTILS_H_

#include <cstdint>

#include "tensorflow/core/profiler/protobuf/tpu_input_pipeline.pb.h"

namespace tensorflow {
namespace profiler {

inline double ComputeTimeMs(const PerTpuStepDetails& details) {
  return details.tc_compute_time_ms() + details.scv0_compute_time_ms();
}

inline double InfeedTimeMs(const PerTpuStepDetails& details) {
  return details.tc_infeed_time_ms() + details.scv0_infeed_time_ms();
}

inline double AllReduceTimeMs(const PerTpuStepDetails& details) {
  return details.all_reduce_compute_time_ms() +
         details.all_reduce_sync_time_ms();
}

inline double NonIdleTimeMs(const PerTpuStepDetails& details) {
  return ComputeTimeMs(details) + InfeedTimeMs(details) +
         AllReduceTimeMs(details) + details.tc_outfeed_time_ms();
}

// Time spent by a training step on TPU.
inline double StepTimeMs(const PerTpuStepDetails& details) {
  return NonIdleTimeMs(details) + details.tc_idle_time_ms();
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TPU_STEP_DETAILS_UTILS_H_
