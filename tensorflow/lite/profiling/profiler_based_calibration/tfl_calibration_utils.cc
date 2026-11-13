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

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_calibration_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_dynamic_histogram.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_tensor_stats_profiler.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {
namespace {

// Exclusive bounds for valid floating-point values, matching the default
// `valid_range` of `min_max_calibrate` in Python.
constexpr float kValidFloatMin = -3e38f;
constexpr float kValidFloatMax = 3e38f;

// Filtered valid-range min/max for a single float tensor visit, returned by
// `UpdateFloatTensorStats` so histogram collection can skip a duplicate range
// scan.
struct FloatTensorRange {
  bool has_valid = false;
  float valid_min = 0.0f;
  float valid_max = 0.0f;
};

// Inserts `[current_min, current_max]` for `tensor_name`, or widens the
// existing `Range` when `tensor_name` has already been seen in this invocation.
void UpdateRange(absl::string_view tensor_name, float current_min,
                 float current_max,
                 std::map<std::string, Range>& tensor_stats) {
  auto [it, inserted] = tensor_stats.insert(
      {std::string(tensor_name), Range{current_min, current_max}});
  if (!inserted) {
    it->second.min = std::min(it->second.min, current_min);
    it->second.max = std::max(it->second.max, current_max);
  }
}

// Updates `tensor_stats` with the min/max of an int32 tensor buffer. No-op if
// `num_elements <= 0`.
void UpdateIntTensorStats(const int32_t* tensor_buffer,
                          Eigen::Index num_elements,
                          absl::string_view tensor_name,
                          std::map<std::string, Range>& tensor_stats) {
  if (num_elements <= 0) {
    return;
  }
  const Eigen::Map<const Eigen::VectorX<int32_t>> vec(tensor_buffer,
                                                      num_elements);
  UpdateRange(tensor_name, static_cast<float>(vec.minCoeff()),
              static_cast<float>(vec.maxCoeff()), tensor_stats);
}

// Updates `tensor_stats` with the min/max of a float32 tensor buffer and
// returns its filtered `[valid_min, valid_max]` when at least one element lies
// in `(kValidFloatMin, kValidFloatMax)`. No-op if `num_elements <= 0`. Skips
// NaNs and excludes values outside `(kValidFloatMin, kValidFloatMax)`, falling
// back to the raw non-NaN min/max if every non-NaN value falls outside that
// range. All-NaN tensors are omitted.
FloatTensorRange UpdateFloatTensorStats(
    const float* tensor_buffer, Eigen::Index num_elements,
    absl::string_view tensor_name, std::map<std::string, Range>& tensor_stats) {
  if (num_elements <= 0) {
    return {};
  }
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const Eigen::Map<const Eigen::ArrayXf> arr(tensor_buffer, num_elements);
  const auto in_range = (arr > kValidFloatMin) && (arr < kValidFloatMax);
  const float valid_min = in_range.select(arr, kInf).minCoeff();
  if (valid_min < kInf) {
    const float valid_max = in_range.select(arr, -kInf).maxCoeff();
    UpdateRange(tensor_name, valid_min, valid_max, tensor_stats);
    return {
        .has_valid = true,
        .valid_min = valid_min,
        .valid_max = valid_max,
    };
  }

  // Fallback when no element lies in `(kValidFloatMin, kValidFloatMax)`: use
  // the raw min/max over non-NaN values, or omit all-NaN tensors.
  float raw_min = kInf;
  float raw_max = -kInf;
  bool has_non_nan = false;
  for (Eigen::Index i = 0; i < num_elements; ++i) {
    const float val = tensor_buffer[i];
    if (!std::isnan(val)) {
      has_non_nan = true;
      raw_min = std::min(raw_min, val);
      raw_max = std::max(raw_max, val);
    }
  }
  if (has_non_nan) {
    UpdateRange(tensor_name, raw_min, raw_max, tensor_stats);
  }
  return {};
}

}  // namespace

absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter, bool collect_histogram) {
  // Use a callback to capture tensor stats during operator invocation. The
  // results are stored in a map of tensor name to min/max stat value.
  std::map<std::string, Range> tensor_stats;
  absl::flat_hash_map<std::string, DynamicHistogram> histograms;
  auto calc_tensor_stats = [&](const TfLiteTensor* tensor) {
    // Skip constant tensors.
    if (tensor->allocation_type == kTfLiteMmapRo ||
        tensor->allocation_type == kTfLitePersistentRo) {
      return;
    }
    // Skip unallocated or unnamed tensors. Tensors without a complete element
    // (e.g. `tensor->bytes == 0`) are skipped by the update helpers.
    if (tensor->data.raw == nullptr || tensor->name == nullptr) {
      return;
    }
    const absl::string_view tensor_name(tensor->name);
    if (tensor->type == kTfLiteFloat32) {
      const auto num_elements =
          static_cast<Eigen::Index>(tensor->bytes / sizeof(float));
      const FloatTensorRange float_range = UpdateFloatTensorStats(
          tensor->data.f, num_elements, tensor_name, tensor_stats);
      if (collect_histogram && float_range.has_valid) {
        histograms[tensor_name].Add(tensor->data.f, num_elements,
                                    float_range.valid_min,
                                    float_range.valid_max);
      }
    } else if (tensor->type == kTfLiteInt32) {
      UpdateIntTensorStats(
          tensor->data.i32,
          static_cast<Eigen::Index>(tensor->bytes / sizeof(int32_t)),
          tensor_name, tensor_stats);
    }
  };

  auto profiler = std::make_unique<odml::TensorStatsProfiler>(
      *interpreter, calc_tensor_stats);
  interpreter->SetProfiler(profiler.get());

  auto invoke_status = interpreter->subgraph(subgraph_index)->Invoke();

  // Reset the profiler to avoid dangling pointer issues when interpreter is
  // reused.
  interpreter->SetProfiler(nullptr);
  for (size_t i = 0; i < interpreter->subgraphs_size(); ++i) {
    interpreter->subgraph(i)->SetProfiler(nullptr, i);
  }

  if (invoke_status != kTfLiteOk) {
    return absl::InternalError("InvokeWithCalibration failed" +
                               (reporter ? ": " + reporter->message() : ""));
  }

  for (auto& [name, hist] : histograms) {
    const double lower_bound = hist.lower_bound();
    const double bin_width = hist.bin_width();
    tensor_stats.at(name).histogram = TensorHistogram{
        .counts = std::move(hist).counts(),
        .lower_bound = lower_bound,
        .bin_width = bin_width,
    };
  }
  return tensor_stats;
}

tflite::profiling::memory::MemoryUsage GetMemoryUsage() {
  return tflite::profiling::memory::GetMemoryUsage();
}
}  // namespace odml
