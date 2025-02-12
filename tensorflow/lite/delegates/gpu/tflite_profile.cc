/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/gpu/tflite_profile.h"

#include "absl/time/time.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/gpu/common/task/profiling_info.h"

namespace tflite {
namespace gpu {

static void* s_profiler = nullptr;

bool IsTfLiteProfilerActive() { return s_profiler != nullptr; }

void SetTfLiteProfiler(void* profiler) { s_profiler = profiler; }

void* GetTfLiteProfiler() { return s_profiler; }

void AddTfLiteProfilerEvents(tflite::gpu::ProfilingInfo* profiling_info) {
  tflite::Profiler* profile =
      reinterpret_cast<tflite::Profiler*>(GetTfLiteProfiler());
  if (profile == nullptr) return;

  int node_index = 0;
  for (const auto& dispatch : profiling_info->dispatches) {
    profile->AddEvent(
        dispatch.label.c_str(),
        Profiler::EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT,
        absl::ToDoubleMicroseconds(dispatch.duration), node_index++);
  }
}

}  // namespace gpu
}  // namespace tflite
