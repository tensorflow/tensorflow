/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_CALIBRATION_UTILS_H_
#define TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_CALIBRATION_UTILS_H_

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {

struct Range {
  float min;
  float max;
};

// Invokes the subgraph at `subgraph_index` using the specified `interpreter`,
// and returns a map of min/max values for each non-constant tensor encountered
// during execution. Currently min/max collection supports only float32 and
// int32 tensors, all other tensors are ignored.
absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter = nullptr);

// Gets memory usage stats.
tflite::profiling::memory::MemoryUsage GetMemoryUsage();

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_CALIBRATION_UTILS_H_
