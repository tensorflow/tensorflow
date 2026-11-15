/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Histogram of a float tensor's valid values collected during one invocation.
struct TensorHistogram {
  std::vector<int64_t> counts;
  double lower_bound = 0.0;
  double bin_width = 0.0;
};

struct Range {
  float min;
  float max;
  // Set only when `InvokeWithCalibration` is called with
  // `collect_histogram = true` on a float32 tensor with at least one value in
  // `(-3e38, 3e38)`.
  std::optional<TensorHistogram> histogram;
};

// Invokes the subgraph at `subgraph_index` using the specified `interpreter`,
// and returns a map of min/max values for each non-constant, non-empty, named
// tensor encountered during execution. Currently min/max collection supports
// only float32 and int32 tensors, all other tensors are ignored.
//
// Float NaNs and values outside `(-3e38, 3e38)` are excluded from min/max
// (similar to Python's `min_max_calibrate`). If a tensor has non-NaN values but
// none within `(-3e38, 3e38)`, its raw non-NaN min/max is used instead; all-NaN
// tensors are omitted.
//
// If `collect_histogram` is true, each returned `Range` for a float32 tensor
// with valid values also holds a histogram for this invocation.
absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter = nullptr,
    bool collect_histogram = false);

// Gets memory usage stats.
tflite::profiling::memory::MemoryUsage GetMemoryUsage();

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_CALIBRATION_UTILS_H_
