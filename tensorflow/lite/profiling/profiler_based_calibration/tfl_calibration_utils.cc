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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
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

template <typename T>
void UpdateFloatTensorStats(const T* tensor_buffer, int num_elements,
                            const char* tensor_name,
                            const CalibrationOptions& options,
                            std::map<std::string, Range>& tensor_stats) {
  if (num_elements <= 0) {
    return;
  }
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float filtered_min = kInf;
  float filtered_max = -kInf;
  float raw_min = kInf;
  float raw_max = -kInf;
  bool has_non_nan = false;

  for (int i = 0; i < num_elements; ++i) {
    const float val = static_cast<float>(tensor_buffer[i]);
    if (std::isnan(val)) {
      continue;
    }
    has_non_nan = true;
    raw_min = std::min(raw_min, val);
    raw_max = std::max(raw_max, val);
    if (val > options.valid_min) {
      filtered_min = std::min(filtered_min, val);
    }
    if (val < options.valid_max) {
      filtered_max = std::max(filtered_max, val);
    }
  }
  if (!has_non_nan) {
    return;
  }
  const float current_min = (filtered_min == kInf) ? raw_min : filtered_min;
  const float current_max = (filtered_max == -kInf) ? raw_max : filtered_max;
  auto [it, inserted] =
      tensor_stats.insert({tensor_name, Range{current_min, current_max}});
  if (!inserted) {
    it->second.min = std::min(it->second.min, current_min);
    it->second.max = std::max(it->second.max, current_max);
  }
}

template <typename T>
void UpdateIntTensorStats(const T* tensor_buffer, int num_elements,
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

CalibrationSession::CalibrationSession(CalibrationOptions options)
    : options_(std::move(options)) {}

bool CalibrationSession::ShouldCollectHistogram(const char* tensor_name) const {
  if (!options_.collect_histogram) {
    return false;
  }
  if (!options_.histogram_tensor_names.has_value()) {
    return true;
  }
  return options_.histogram_tensor_names->count(tensor_name) > 0;
}

DynamicHistogram& CalibrationSession::GetOrCreateHistogram(
    const char* tensor_name) {
  return histograms_.try_emplace(tensor_name, options_.num_bins).first->second;
}

absl::StatusOr<std::map<std::string, Range>> CalibrationSession::Invoke(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter) {
  // Detect active token count from a 2D [1, seq_len] mask input tensor if
  // present so that token-wise activation statistics and histograms exclude
  // mask == 0 right-padded positions.
  int mask_seq_len = 0;
  int valid_seq_len = 0;
  auto* main_subgraph = interpreter->subgraph(subgraph_index);
  if (main_subgraph != nullptr) {
    for (int inp_idx : main_subgraph->inputs()) {
      const TfLiteTensor* inp = main_subgraph->tensor(inp_idx);
      if (inp == nullptr || inp->name == nullptr || inp->data.raw == nullptr ||
          inp->dims == nullptr || inp->dims->size != 2 ||
          inp->dims->data[0] != 1 || inp->dims->data[1] <= 1) {
        continue;
      }
      const absl::string_view inp_name(inp->name);
      if (!absl::StrContains(inp_name, "mask")) {
        continue;
      }
      const int seq_len = inp->dims->data[1];
      int valid = 0;
      if (inp->type == kTfLiteFloat32 && inp->data.f != nullptr) {
        for (int i = 0; i < seq_len; ++i) {
          if (inp->data.f[i] != 0.0f) {
            ++valid;
          }
        }
      } else if (inp->type == kTfLiteInt32 && inp->data.i32 != nullptr) {
        for (int i = 0; i < seq_len; ++i) {
          if (inp->data.i32[i] != 0) {
            ++valid;
          }
        }
      } else {
        continue;
      }
      if (valid > 0 && valid <= seq_len) {
        mask_seq_len = seq_len;
        valid_seq_len = valid;
        break;
      }
    }
  }

  auto get_effective_num_elements = [&](const TfLiteTensor* tensor,
                                        int total_elements) -> int {
    if (valid_seq_len <= 0 || valid_seq_len >= mask_seq_len ||
        tensor->dims == nullptr || tensor->dims->size < 3 ||
        tensor->dims->data[0] != 1 || tensor->dims->data[1] != mask_seq_len ||
        total_elements % mask_seq_len != 0) {
      return total_elements;
    }
    const absl::string_view name(tensor->name);
    if (absl::StrContains(
            name, "_compute_encoder_final_activations_from_embeddings") ||
        absl::StrContains(name, "heads_layer")) {
      return total_elements;
    }
    return (total_elements / mask_seq_len) * valid_seq_len;
  };

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
    if (tensor->data.raw == nullptr || tensor->name == nullptr) {
      return;
    }
    const bool collect_hist = ShouldCollectHistogram(tensor->name);
    auto update_float = [&](const auto* data) {
      const int total_n = tensor->bytes / sizeof(*data);
      const int n = get_effective_num_elements(tensor, total_n);
      UpdateFloatTensorStats(data, n, tensor->name, options_, tensor_stats);
      if (collect_hist) {
        GetOrCreateHistogram(tensor->name)
            .Add(data, n, options_.valid_min, options_.valid_max);
      }
    };
    switch (tensor->type) {
      case kTfLiteFloat32:
        update_float(tensor->data.f);
        break;
      case kTfLiteFloat16:
        update_float(reinterpret_cast<const Eigen::half*>(tensor->data.f16));
        break;
      case kTfLiteBFloat16:
        update_float(
            reinterpret_cast<const Eigen::bfloat16*>(tensor->data.bf16));
        break;
      case kTfLiteInt32: {
        const int total_n = tensor->bytes / sizeof(int32_t);
        const int n = get_effective_num_elements(tensor, total_n);
        UpdateIntTensorStats<int32_t>(tensor->data.i32, n, tensor->name,
                                      tensor_stats);
        if (collect_hist) {
          GetOrCreateHistogram(tensor->name).Add(tensor->data.i32, n);
        }
        break;
      }
      default:
        break;
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

std::map<std::string, TensorHistogram> CalibrationSession::GetHistograms()
    const {
  std::map<std::string, TensorHistogram> result;
  for (const auto& [name, hist] : histograms_) {
    if (!hist.initialized()) {
      continue;
    }
    result.emplace(name, TensorHistogram{
                             .counts = hist.counts(),
                             .lower_bound = hist.lower_bound(),
                             .bin_width = hist.bin_width(),
                             .min = hist.global_min(),
                             .max = hist.global_max(),
                         });
  }
  return result;
}

void CalibrationSession::Reset() { histograms_.clear(); }

absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter) {
  CalibrationSession session;
  return session.Invoke(interpreter, subgraph_index, reporter);
}

tflite::profiling::memory::MemoryUsage GetMemoryUsage() {
  return tflite::profiling::memory::GetMemoryUsage();
}
}  // namespace odml
