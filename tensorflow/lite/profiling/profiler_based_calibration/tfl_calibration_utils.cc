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
#include <cstddef>
#include <cstdint>
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

template <typename T>
void UpdateTensorStats(const T* tensor_buffer, int num_elements,
                       const char* tensor_name,
                       std::map<std::string, Range>& tensor_stats) {
  if (num_elements <= 0) {
    return;
  }
  const Eigen::Map<const Eigen::VectorX<T>> vec(tensor_buffer, num_elements);
  const float current_min = static_cast<float>(vec.minCoeff());
  const float current_max = static_cast<float>(vec.maxCoeff());
  auto [it, inserted] =
      tensor_stats.insert({tensor_name, Range{current_min, current_max}});
  if (!inserted) {
    it->second.min = std::min(it->second.min, current_min);
    it->second.max = std::max(it->second.max, current_max);
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
    // Skip empty or unallocated tensors.
    if (tensor->data.raw == nullptr) {
      return;
    }
    if (tensor->type == kTfLiteFloat32) {
      UpdateTensorStats<float>(tensor->data.f, tensor->bytes / sizeof(float),
                               tensor->name, tensor_stats);
    } else if (tensor->type == kTfLiteInt32) {
      UpdateTensorStats<int32_t>(tensor->data.i32,
                                 tensor->bytes / sizeof(int32_t), tensor->name,
                                 tensor_stats);
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
