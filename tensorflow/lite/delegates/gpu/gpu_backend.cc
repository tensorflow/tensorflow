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

#include "tensorflow/lite/delegates/gpu/gpu_backend.h"

namespace tflite {
namespace gpu {

InferencePriority GpuBackend::ToPriority(int32_t priority) {
  switch (priority) {
    case TFLITE_GPU_INFERENCE_PRIORITY_AUTO:
      return InferencePriority::AUTO;
    case TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:
      return InferencePriority::MAX_PRECISION;
    case TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:
      return InferencePriority::MIN_LATENCY;
    case TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE:
      return InferencePriority::MIN_MEMORY_USAGE;
  }
  return InferencePriority::UNKNOWN;
}

InferenceUsage GpuBackend::ToUsage(int32_t usage) {
  switch (usage) {
    case TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return InferenceUsage::FAST_SINGLE_ANSWER;
    case TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return InferenceUsage::SUSTAINED_SPEED;
  }
  return InferenceUsage::UNKNOWN;
}

}  // namespace gpu
}  // namespace tflite
