/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/profiling_info.h"

#include <map>
#include <string>

namespace tflite {
namespace gpu {

absl::Duration ProfilingInfo::GetTotalTime() const {
  absl::Duration total_time;
  for (const auto& dispatch : dispatches) {
    total_time += dispatch.duration;
  }
  return total_time;
}

std::string ProfilingInfo::GetDetailedReport() const {
  std::string result;
  struct OpStatistic {
    int count;
    double total_time;
  };
  std::map<std::string, OpStatistic> statistics;
  result +=
      "Per kernel timing(" + std::to_string(dispatches.size()) + " kernels):\n";
  for (const auto& dispatch : dispatches) {
    result += "  " + dispatch.label + " - " +
              std::to_string(absl::ToDoubleMilliseconds(dispatch.duration)) +
              " ms";
    const double times_per_sec =
        1000.0 / absl::ToDoubleMilliseconds(dispatch.duration);
    if (dispatch.read_mem_size && dispatch.write_mem_size) {
      const uint64_t total_size =
          dispatch.read_mem_size + dispatch.write_mem_size;
      const double giga_bytes = total_size / 1024.0 / 1024.0 / 1024.0;
      const double giga_bytes_per_sec = times_per_sec * giga_bytes;
      result += ", " + std::to_string(giga_bytes_per_sec) + " Gbyte/s";
    }
    if (dispatch.flops) {
      const double giga_flops = dispatch.flops / 1000.0 / 1000.0 / 1000.0;
      const double giga_flops_per_sec = times_per_sec * giga_flops;
      result += ", " + std::to_string(giga_flops_per_sec) + " Gflops";
    }
    result += "\n";
    auto name = dispatch.label.substr(0, dispatch.label.find(' '));
    if (statistics.find(name) != statistics.end()) {
      statistics[name].count++;
      statistics[name].total_time +=
          absl::ToDoubleMilliseconds(dispatch.duration);
    } else {
      statistics[name].count = 1;
      statistics[name].total_time =
          absl::ToDoubleMilliseconds(dispatch.duration);
    }
  }
  result += "--------------------\n";
  result += "Accumulated time per operation type:\n";
  for (auto& t : statistics) {
    auto stat = t.second;
    result += "  " + t.first + "(x" + std::to_string(stat.count) + ") - " +
              std::to_string(stat.total_time) + " ms\n";
  }
  result += "--------------------\n";
  result += "Ideal total time: " +
            std::to_string(absl::ToDoubleMilliseconds(GetTotalTime())) + "\n";
  result += "--------------------\n";
  return result;
}

}  // namespace gpu
}  // namespace tflite
