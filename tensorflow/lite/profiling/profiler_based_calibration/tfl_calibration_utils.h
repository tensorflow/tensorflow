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

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {

// Histogram of a tensor's values collected during a single invocation.
struct TensorHistogram {
  std::vector<int64_t> counts;
  double lower_bound = 0.0;
  double bin_width = 0.0;
};

struct Range {
  float min;
  float max;
  // Set only when `InvokeWithCalibration` is called with
  // `collect_histogram = true` and the tensor has at least one valid value.
  std::optional<TensorHistogram> histogram;
};

// Invokes the subgraph at `subgraph_index` using the specified `interpreter`,
// and returns a map of min/max values for each non-constant tensor encountered
// during execution. Currently min/max collection supports only float32 and
// int32 tensors, all other tensors are ignored.
//
// Float NaNs and values outside `(kValidFloatMin, kValidFloatMax)` are
// excluded from min/max unless every value is excluded, matching the Python
// `get_activation_min_max` used by preserve-all-tensors calibration.
//
// If `collect_histogram` is true, each returned `Range` also holds a histogram
// of the tensor's valid values for this invocation.
absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter = nullptr,
    bool collect_histogram = false);

// Gets memory usage stats.
tflite::profiling::memory::MemoryUsage GetMemoryUsage();

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_CALIBRATION_UTILS_H_
