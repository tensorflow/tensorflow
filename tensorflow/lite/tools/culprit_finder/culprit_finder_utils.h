/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_UTILS_H_

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite::tooling {
using ::tflite::tools::ToolParam;
using ::tflite::tools::ToolParams;

template <typename T>
tflite::Flag CreateFlag(const char* name, ToolParams* params,
                        const std::string& usage) {
  return tflite::Flag(
      name,
      [params, name](const T& val, int argv_position) {
        params->Set<T>(name, val, argv_position);
      },
      params->Get<T>(name), usage, tflite::Flag::kOptional);
}

template <class T, class U>
struct PairHash {
  std::size_t operator()(const std::pair<T, U>& p) const {
    // Combine hash values of the pair elements using XOR and bit shifting
    return std::hash<T>()(p.first) ^ (std::hash<U>()(p.second) << 1);
  }
};

// Stores error stats for a single OutputTensor.
struct TensorErrorStat {
  int tensor_index;
  float avg;
  float std_dev;
  float max;
  float min;
  bool delegate_output_is_nan = false;
  bool reference_output_is_nan = false;
};

// Stores error stats for a single inference comparison.
struct OverallStat {
  std::pair<int, int> delegated_node_range;
  std::vector<TensorErrorStat> output_error_stats;
  float min_error;
  float max_error;
  float total_error;
  std::vector<int> nan_output_indices;
  bool is_crash = false;
};

// Returns the error stats for a single OutputTensor.
template <typename T>
TensorErrorStat GetTensorErrorStat(int tensor_index, T* reference_output,
                                   T* test_output, int num_elements) {
  float sum = 0.0;
  float sum_sq = 0.0;
  float max = 0.0;
  float min = FLT_MAX;
  for (size_t i = 0; i < num_elements; ++i) {
    float reference_value = reference_output[i];
    float test_value = test_output[i];
    if (std::isnan(reference_value)) {
      std::cout << "### Reference value is NaN for tensor index: "
                << tensor_index << "\n";
      return {tensor_index, 0.0, 0.0, 0.0, 0.0, false, true};
    }
    if (std::isnan(test_value)) {
      return {tensor_index, 0.0, 0.0, 0.0, 0.0, true, false};
    }
    float diff = std::abs(test_value - reference_value);
    min = std::min(min, diff);
    max = std::max(max, diff);
    sum += diff;
    sum_sq += diff * diff;
  }

  float avg = sum / num_elements;
  float std_dev = std::sqrt(sum_sq / num_elements - avg * avg);
  return {tensor_index, avg, std_dev, max, min};
}

inline void GetOverallStat(int delegated_node_range_start,
                           int delegated_node_range_end,
                           tflite::Interpreter* reference_interpreter,
                           tflite::Interpreter* test_interpreter, bool is_crash,
                           OverallStat* overall_stat) {
  overall_stat->delegated_node_range = {delegated_node_range_start,
                                        delegated_node_range_end};
  overall_stat->min_error = FLT_MAX;
  overall_stat->max_error = 0.0;
  overall_stat->total_error = 0.0;
  if (is_crash) {
    overall_stat->is_crash = true;
    return;
  }

  for (size_t i = 0; i < test_interpreter->outputs().size(); ++i) {
    int tensor_index = test_interpreter->outputs()[i];
    TfLiteTensor* test_tensor = test_interpreter->tensor(tensor_index);
    TfLiteTensor* reference_tensor =
        reference_interpreter->tensor(tensor_index);

    TensorErrorStat tensor_error_stat;
    switch (test_tensor->type) {
      case kTfLiteBool:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index, static_cast<bool*>((void*)reference_tensor->data.raw),
            static_cast<bool*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(bool));
        break;
      case kTfLiteUInt8:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<uint8_t*>((void*)reference_tensor->data.raw),
            static_cast<uint8_t*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(uint8_t));
        break;
      case kTfLiteInt8:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<int8_t*>((void*)reference_tensor->data.raw),
            static_cast<int8_t*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(int8_t));
        break;
      case kTfLiteFloat16:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<uint16_t*>((void*)reference_tensor->data.raw),
            static_cast<uint16_t*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(uint16_t));
        break;
      case kTfLiteFloat32:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<float*>((void*)reference_tensor->data.raw),
            static_cast<float*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(float));
        break;
      case kTfLiteInt16:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<int16_t*>((void*)reference_tensor->data.raw),
            static_cast<int16_t*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(int16_t));
        break;
      case kTfLiteInt32:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<int32_t*>((void*)reference_tensor->data.raw),
            static_cast<int32_t*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(int32_t));
        break;
      case kTfLiteUInt32:
        tensor_error_stat = GetTensorErrorStat(
            tensor_index,
            static_cast<uint32_t*>((void*)reference_tensor->data.raw),
            static_cast<uint32_t*>((void*)test_tensor->data.raw),
            test_tensor->bytes / sizeof(uint32_t));
        break;
      default:
        TFLITE_LOG(ERROR) << "Unsupported tensor type: " << test_tensor->type;
    }
    overall_stat->output_error_stats.push_back(tensor_error_stat);
    overall_stat->min_error =
        std::min(overall_stat->min_error, tensor_error_stat.min);
    overall_stat->max_error =
        std::max(overall_stat->max_error, tensor_error_stat.max);
    overall_stat->total_error += tensor_error_stat.avg;
    if (tensor_error_stat.delegate_output_is_nan) {
      overall_stat->nan_output_indices.push_back(tensor_index);
    }
  }
}
}  // namespace tflite::tooling
#endif  // TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_UTILS_H_
