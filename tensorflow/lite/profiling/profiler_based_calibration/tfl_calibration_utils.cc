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

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_calibration_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_dynamic_histogram.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_tensor_stats_profiler.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {
namespace {

void UpdateRange(const char* tensor_name, float current_min, float current_max,
                 std::map<std::string, Range>& tensor_stats) {
  auto [it, inserted] =
      tensor_stats.insert({tensor_name, Range{current_min, current_max}});
  if (!inserted) {
    it->second.min = std::min(it->second.min, current_min);
    it->second.max = std::max(it->second.max, current_max);
  }
}

void UpdateIntTensorStats(const int32_t* tensor_buffer, int num_elements,
                          const char* tensor_name,
                          std::map<std::string, Range>& tensor_stats) {
  const Eigen::Map<const Eigen::VectorX<int32_t>> vec(tensor_buffer,
                                                      num_elements);
  UpdateRange(tensor_name, static_cast<float>(vec.minCoeff()),
              static_cast<float>(vec.maxCoeff()), tensor_stats);
}

void UpdateFloatTensorStats(const float* tensor_buffer, int num_elements,
                            const char* tensor_name,
                            std::map<std::string, Range>& tensor_stats) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float valid_min = kInf;
  float valid_max = -kInf;
  float raw_min = kInf;
  float raw_max = -kInf;
  bool has_non_nan = false;
  for (int i = 0; i < num_elements; ++i) {
    const float val = tensor_buffer[i];
    if (std::isnan(val)) {
      continue;
    }
    has_non_nan = true;
    raw_min = std::min(raw_min, val);
    raw_max = std::max(raw_max, val);
    if (val > kValidFloatMin) {
      valid_min = std::min(valid_min, val);
    }
    if (val < kValidFloatMax) {
      valid_max = std::max(valid_max, val);
    }
  }
  if (!has_non_nan) {
    return;
  }
  UpdateRange(tensor_name, valid_min == kInf ? raw_min : valid_min,
              valid_max == -kInf ? raw_max : valid_max, tensor_stats);
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
    // Skip empty, unallocated or unnamed tensors.
    if (tensor->data.raw == nullptr || tensor->bytes == 0 ||
        tensor->name == nullptr) {
      return;
    }
    if (tensor->type == kTfLiteFloat32) {
      const int num_elements = tensor->bytes / sizeof(float);
      UpdateFloatTensorStats(tensor->data.f, num_elements, tensor->name,
                             tensor_stats);
      if (collect_histogram) {
        histograms[tensor->name].Add(tensor->data.f, num_elements);
      }
    } else if (tensor->type == kTfLiteInt32) {
      const int num_elements = tensor->bytes / sizeof(int32_t);
      UpdateIntTensorStats(tensor->data.i32, num_elements, tensor->name,
                           tensor_stats);
      if (collect_histogram) {
        histograms[tensor->name].Add(tensor->data.i32, num_elements);
      }
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

  for (const auto& [name, hist] : histograms) {
    auto it = tensor_stats.find(name);
    if (!hist.initialized() || it == tensor_stats.end()) {
      continue;
    }
    it->second.histogram = TensorHistogram{
        .counts = hist.counts(),
        .lower_bound = hist.lower_bound(),
        .bin_width = hist.bin_width(),
    };
  }
  return tensor_stats;
}

tflite::profiling::memory::MemoryUsage GetMemoryUsage() {
  return tflite::profiling::memory::GetMemoryUsage();
}
}  // namespace odml
