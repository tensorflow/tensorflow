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
#ifndef TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARY_FORMATTER_H_
#define TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARY_FORMATTER_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/util/stat_summarizer_options.h"
#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/profiling/proto/profiling_info.pb.h"

namespace tflite {
namespace profiling {

// Formats the profile summary in a certain way.
class ProfileSummaryFormatter {
 public:
  ProfileSummaryFormatter() = default;
  virtual ~ProfileSummaryFormatter() {}
  // Returns a string detailing the accumulated runtime stats in StatsCalculator
  // of ProfileSummarizer.
  virtual std::string GetOutputString(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const = 0;
  // Returns a string detailing the short summary of the accumulated runtime
  // stats in StatsCalculator of ProfileSummarizer.
  virtual std::string GetShortSummary(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const = 0;
  virtual tensorflow::StatSummarizerOptions GetStatSummarizerOptions()
      const = 0;
  virtual void HandleOutput(const std::string& init_output,
                            const std::string& run_output,
                            std::string output_file_path) const = 0;
};

class ProfileSummaryDefaultFormatter : public ProfileSummaryFormatter {
 public:
  ProfileSummaryDefaultFormatter() = default;
  ~ProfileSummaryDefaultFormatter() override {}
  std::string GetOutputString(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const override;
  std::string GetShortSummary(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const override;
  tensorflow::StatSummarizerOptions GetStatSummarizerOptions() const override;
  void HandleOutput(const std::string& init_output,
                    const std::string& run_output,
                    std::string output_file_path) const override;

 private:
  std::string GenerateReport(
      const std::string& tag, bool include_output_string,
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const;
  void WriteOutput(const std::string& header, const std::string& data,
                   std::ostream* stream) const {
    (*stream) << header << std::endl;
    (*stream) << data << std::endl;
  }
};

class ProfileSummaryCSVFormatter : public ProfileSummaryDefaultFormatter {
 public:
  ProfileSummaryCSVFormatter() = default;
  tensorflow::StatSummarizerOptions GetStatSummarizerOptions() const override;
};

class ProfileSummaryProtoFormatter : public ProfileSummaryFormatter {
 public:
  std::string GetOutputString(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const override;
  std::string GetShortSummary(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const override;
  tensorflow::StatSummarizerOptions GetStatSummarizerOptions() const override;
  void HandleOutput(const std::string& init_output,
                    const std::string& run_output,
                    std::string output_file_path) const override;

 private:
  std::string GenerateReport(
      const std::string& tag, bool include_output_string,
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator,
      const std::map<uint32_t, std::string>& subgraph_name_map) const;
  void GenerateSubGraphProfilingData(
      const tensorflow::StatsCalculator* stats_calculator, int subgraph_index,
      const std::map<uint32_t, std::string>& subgraph_name_map,
      SubGraphProfilingData* sub_graph_profiling_data) const;

  void GenerateDelegateProfilingData(
      const tensorflow::StatsCalculator* stats_calculator,
      DelegateProfilingData* delegate_profiling_data) const;

  void GenerateOpProfileDataFromDetail(
      const tensorflow::StatsCalculator::Detail* detail,
      const tensorflow::StatsCalculator* stats_calculator,
      OpProfileData* op_profile_data) const;

  std::vector<tensorflow::StatsCalculator::Detail> GetDetailsSortedByRunOrder(
      const tensorflow::StatsCalculator* stats_calculator) const;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARY_FORMATTER_H_
