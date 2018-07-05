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

#ifndef TENSORFLOW_CONTRIB_LITE_PROFILING_PROFILE_SUMMARIZER_H_
#define TENSORFLOW_CONTRIB_LITE_PROFILING_PROFILE_SUMMARIZER_H_

#include <vector>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/profiling/profiler.h"
#include "tensorflow/core/util/stats_calculator.h"

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
  std::string GetOutputString() const {
    return stats_calculator_->GetOutputString();
  }

  std::string GetShortSummary() const {
    return stats_calculator_->GetShortSummary();
  }

 private:
  std::unique_ptr<tensorflow::StatsCalculator> stats_calculator_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_PROFILING_PROFILE_SUMMARIZER_H_
