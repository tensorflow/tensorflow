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
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_dynamic_histogram.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {

struct Range {
  float min;
  float max;
};

struct CalibrationOptions {
  // Exclusive valid boundaries for floating-point values. Values outside
  // (valid_min, valid_max) or NaN are ignored unless all non-NaN values in the
  // tensor fall outside the valid boundary, in which case the unfiltered
  // min/max is used as fallback (matching Python get_activation_min_max).
  float valid_min = -3e38f;
  float valid_max = 3e38f;
  // Whether to accumulate dynamic histograms across invocations.
  bool collect_histogram = false;
  // Maximum number of bins per tensor histogram.
  int num_bins = 2048;
  // Optional allowlist of tensor names for which histograms should be
  // collected. If std::nullopt when `collect_histogram` is true, histograms are
  // collected for all non-constant tensors.
  std::optional<std::unordered_set<std::string>> histogram_tensor_names =
      std::nullopt;
};

struct TensorHistogram {
  std::vector<int64_t> counts;
  double lower_bound = 0.0;
  double bin_width = 0.0;
  float min = 0.0f;
  float max = 0.0f;
};

// Stateful calibration session that returns per-invocation min/max statistics
// and optionally accumulates dynamic histograms across multiple invocations.
class CalibrationSession {
 public:
  explicit CalibrationSession(CalibrationOptions options = {});

  // Invokes the subgraph at `subgraph_index` using the specified `interpreter`.
  // Returns a map of min/max values for each non-constant tensor encountered
  // during this invocation, and accumulates histograms internally when
  // `options_.collect_histogram` is enabled.
  absl::StatusOr<std::map<std::string, Range>> Invoke(
      tflite::Interpreter* interpreter, int subgraph_index,
      tflite::StatefulErrorReporter* reporter = nullptr);

  // Returns accumulated histograms for all initialized tensors.
  std::map<std::string, TensorHistogram> GetHistograms() const;

  // Clears all accumulated histograms.
  void Reset();

 private:
  bool ShouldCollectHistogram(const char* tensor_name) const;
  DynamicHistogram& GetOrCreateHistogram(const char* tensor_name);

  CalibrationOptions options_;
  absl::flat_hash_map<std::string, DynamicHistogram> histograms_;
};

// Invokes the subgraph at `subgraph_index` using the specified `interpreter`,
// and returns a map of min/max values for each non-constant tensor encountered
// during execution, using the default `CalibrationOptions` valid range.
// Supports float32, float16, bfloat16, and int32 tensors; all other tensors are
// ignored.
absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter = nullptr);

// Gets memory usage stats.
tflite::profiling::memory::MemoryUsage GetMemoryUsage();

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_CALIBRATION_UTILS_H_
