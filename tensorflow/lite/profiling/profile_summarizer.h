/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARIZER_H_
#define TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARIZER_H_

#include <functional>
#include <vector>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/profile_buffer.h"

namespace tflite {
namespace profiling {

// Creates a summary of operator invocations in the interpreter.
class ProfileSummarizer {
 public:
  ProfileSummarizer();
  virtual ~ProfileSummarizer() {}

  // Process profile events to update statistics for operator invocations.
  void ProcessProfiles(const std::vector<const ProfileEvent*>& profile_stats,
                       const tflite::Interpreter& interpreter);

  // Returns a string detailing the accumulated runtime stats in a tab-separated
  // format which can be pasted into a spreadsheet for further analysis.
  std::string GetOutputString() {
    return GenerateReport("profile", /*include_output_string*/ true);
  }

  std::string GetShortSummary() {
    return GenerateReport("summary", /*include_output_string*/ false);
  }

  tensorflow::StatsCalculator* GetStatsCalculator(uint32_t subgraph_index);

  bool HasProfiles() {
    for (auto& stats_calc : stats_calculator_map_) {
      auto subgraph_stats = stats_calc.second.get();
      if (subgraph_stats->num_runs() >= 1) return true;
    }
    return false;
  }

 private:
  // Map storing stats per subgraph.
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map_;

  // GenerateReport returns the report of subgraphs in a string format.
  std::string GenerateReport(std::string tag, bool include_output_string);
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARIZER_H_
