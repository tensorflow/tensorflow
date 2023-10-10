/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/profile_summary_formatter.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace tflite {
namespace profiling {

std::string ProfileSummaryDefaultFormatter::GetOutputString(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator) const {
  return GenerateReport("profile", /*include_output_string*/ true,
                        stats_calculator_map, delegate_stats_calculator);
}

std::string ProfileSummaryDefaultFormatter::GetShortSummary(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator) const {
  return GenerateReport("summary", /*include_output_string*/ false,
                        stats_calculator_map, delegate_stats_calculator);
}

std::string ProfileSummaryDefaultFormatter::GenerateReport(
    const std::string& tag, bool include_output_string,
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator) const {
  std::stringstream stream;
  bool has_non_primary_graph =
      (stats_calculator_map.size() - stats_calculator_map.count(0)) > 0;
  for (const auto& stats_calc : stats_calculator_map) {
    auto subgraph_index = stats_calc.first;
    auto subgraph_stats = stats_calc.second.get();
    if (has_non_primary_graph) {
      if (subgraph_index == 0) {
        stream << "Primary graph " << tag << ":" << std::endl;
      } else {
        stream << "Subgraph (index: " << subgraph_index << ") " << tag << ":"
               << std::endl;
      }
    }
    if (include_output_string) {
      stream << subgraph_stats->GetOutputString();
    }
    if (subgraph_index != 0) {
      stream << "Subgraph (index: " << subgraph_index << ") ";
    }
    stream << subgraph_stats->GetShortSummary() << std::endl;
  }

  if (delegate_stats_calculator.num_runs() > 0) {
    stream << "Delegate internal: " << std::endl;
    if (include_output_string) {
      stream << delegate_stats_calculator.GetOutputString();
    }
    stream << delegate_stats_calculator.GetShortSummary() << std::endl;
  }

  return stream.str();
}

tensorflow::StatSummarizerOptions
ProfileSummaryDefaultFormatter::GetStatSummarizerOptions() const {
  auto options = tensorflow::StatSummarizerOptions();
  // Summary will be manually handled per subgraphs in order to keep the
  // compatibility.
  options.show_summary = false;
  options.show_memory = false;
  return options;
}

tensorflow::StatSummarizerOptions
ProfileSummaryCSVFormatter::GetStatSummarizerOptions() const {
  auto options = ProfileSummaryDefaultFormatter::GetStatSummarizerOptions();
  options.format_as_csv = true;
  return options;
}

}  // namespace profiling
}  // namespace tflite
