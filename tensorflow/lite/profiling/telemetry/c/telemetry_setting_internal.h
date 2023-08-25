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
#ifndef TENSORFLOW_LITE_PROFILING_TELEMETRY_C_TELEMETRY_SETTING_INTERNAL_H_
#define TENSORFLOW_LITE_PROFILING_TELEMETRY_C_TELEMETRY_SETTING_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct TfLiteTelemetryConversionMetadata {
  std::vector<int32_t> model_optimization_modes;
};

struct TfLiteTelemetrySubgraphInfo {
  std::vector<TfLiteQuantization> quantizations;
};

struct TfLiteTelemetryInterpreterSettings {
  std::unique_ptr<TfLiteTelemetryConversionMetadata> conversion_metadata;
  std::vector<TfLiteTelemetrySubgraphInfo> subgraph_infos;
};

struct TfLiteTelemetryGpuDelegateSettings {
  // Reported by "GpuDelegate::DelegatePrepare" event.
  size_t num_nodes_delegated;

  // Reported by "GpuDelegateKernel::Prepare" event.
  enum Backend : int {
    UNKNOWN = 0,
    OPENCL = 1,
    OPENGL = 2,
  };
  Backend backend;
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_PROFILING_TELEMETRY_C_TELEMETRY_SETTING_INTERNAL_H_
