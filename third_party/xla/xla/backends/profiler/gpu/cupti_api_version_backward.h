/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_API_VERSION_BACKWARD_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_API_VERSION_BACKWARD_H_

#include <cstdint>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"

// Following cupti API appears after some new CUDA version. This header filed
// declares the API for backward compatibility. And dummy implementation is
// provided in corresponding version_numbered .cc file(s).
extern "C" {

CUptiResult CUPTIAPI cuptiActivityEnableHWTrace(uint8_t enable);
}

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_API_VERSION_BACKWARD_H_
