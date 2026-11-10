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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_tensor_stats_profiler.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {
namespace {

// Exclusive bounds for valid floating-point values, matching the default
// `valid_range` of `min_max_calibrate` in Python.
constexpr float kValidFloatMin = -3e38f;
constexpr float kValidFloatMax = 3e38f;

// Inserts `[current_min, current_max]` for `tensor_name`, or widens the
// existing `Range` when `tensor_name` has already been seen in this invocation.
void UpdateRange(const char* tensor_name, float current_min, float current_max,
                 std::map<std::string, Range>& tensor_stats) {
  auto [it, inserted] =
      tensor_stats.insert({tensor_name, Range{current_min, current_max}});
  if (!inserted) {
    it->second.min = std::min(it->second.min, current_min);
    it->second.max = std::max(it->second.max, current_max);
  }
}

// Updates `tensor_stats` with the min/max of an int32 tensor buffer. No-op if
// `num_elements <= 0`.
void UpdateIntTensorStats(const int32_t* tensor_buffer,
                          Eigen::Index num_elements, const char* tensor_name,
                          std::map<std::string, Range>& tensor_stats) {
  if (num_elements <= 0) {
    return;
  }
  const Eigen::Map<const Eigen::VectorX<int32_t>> vec(tensor_buffer,
                                                      num_elements);
  UpdateRange(tensor_name, static_cast<float>(vec.minCoeff()),
              static_cast<float>(vec.maxCoeff()), tensor_stats);
}

// Updates `tensor_stats` with the min/max of a float32 tensor buffer. No-op if
// `num_elements <= 0`. Skips NaNs and excludes values outside
// `(kValidFloatMin, kValidFloatMax)`, falling back to the raw non-NaN min/max
// if every non-NaN value falls outside that range. All-NaN tensors are omitted.
void UpdateFloatTensorStats(const float* tensor_buffer,
                            Eigen::Index num_elements, const char* tensor_name,
                            std::map<std::string, Range>& tensor_stats) {
  if (num_elements <= 0) {
    return;
  }
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const Eigen::Map<const Eigen::ArrayXf> arr(tensor_buffer, num_elements);
  const auto in_range = (arr > kValidFloatMin) && (arr < kValidFloatMax);
  const float valid_min = in_range.select(arr, kInf).minCoeff();
  if (valid_min < kInf) {
    const float valid_max = in_range.select(arr, -kInf).maxCoeff();
    UpdateRange(tensor_name, valid_min, valid_max, tensor_stats);
    return;
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
}

}  // namespace

absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter) {
  // Use a callback to capture tensor stats during operator invocation. The
  // results are stored in a map of tensor name to min/max stat value.
  std::map<std::string, Range> tensor_stats;
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
    if (tensor->type == kTfLiteFloat32) {
      UpdateFloatTensorStats(
          tensor->data.f,
          static_cast<Eigen::Index>(tensor->bytes / sizeof(float)),
          tensor->name, tensor_stats);
    } else if (tensor->type == kTfLiteInt32) {
      UpdateIntTensorStats(
          tensor->data.i32,
          static_cast<Eigen::Index>(tensor->bytes / sizeof(int32_t)),
          tensor->name, tensor_stats);
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

  return tensor_stats;
}

tflite::profiling::memory::MemoryUsage GetMemoryUsage() {
  return tflite::profiling::memory::GetMemoryUsage();
}
}  // namespace odml
