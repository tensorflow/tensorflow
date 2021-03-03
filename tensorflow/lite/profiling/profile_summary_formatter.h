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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/util/stats_calculator.h"

namespace tflite {
namespace profiling {

// Formats the profile summary in a certain way.
class ProfileSummaryFormatter {
 public:
  ProfileSummaryFormatter() {}
  virtual ~ProfileSummaryFormatter() {}
  // Returns a string detailing the accumulated runtime stats in StatsCalculator
  // of ProfileSummarizer.
  virtual std::string GetOutputString(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator) const = 0;
  // Returns a string detailing the short summary of the accumulated runtime
  // stats in StatsCalculator of ProfileSummarizer.
  virtual std::string GetShortSummary(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator) const = 0;
  virtual tensorflow::StatSummarizerOptions GetStatSummarizerOptions()
      const = 0;
};

class ProfileSummaryDefaultFormatter : public ProfileSummaryFormatter {
 public:
  ProfileSummaryDefaultFormatter() {}
  ~ProfileSummaryDefaultFormatter() override {}
  std::string GetOutputString(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator)
      const override;
  std::string GetShortSummary(
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator)
      const override;
  tensorflow::StatSummarizerOptions GetStatSummarizerOptions() const override;

 private:
  std::string GenerateReport(
      const std::string& tag, bool include_output_string,
      const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
          stats_calculator_map,
      const tensorflow::StatsCalculator& delegate_stats_calculator) const;
};

class ProfileSummaryCSVFormatter : public ProfileSummaryDefaultFormatter {
 public:
  ProfileSummaryCSVFormatter() {}
  tensorflow::StatSummarizerOptions GetStatSummarizerOptions() const override;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_PROFILE_SUMMARY_FORMATTER_H_
