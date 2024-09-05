/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DCN_SLACK_ANALYSIS_COMBINER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DCN_SLACK_ANALYSIS_COMBINER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"

namespace tensorflow {
namespace profiler {

using tensorflow::profiler::DcnSlackAnalysis;
using tensorflow::profiler::DcnSlackSummary;

class DcnSlackAnalysisCombiner {
 private:
  absl::flat_hash_map<std::string, DcnSlackSummary> slack_summary_;

 public:
  // Combine the DCN Slack Summary in the DcnSlackAnalysis.
  // The DcnSlackAnalysis consists of average durations, The combine phase, the
  // summary consists of the total duration for all the occurrences. Finazile
  // must be called to get the accurate value.
  void Combine(const DcnSlackAnalysis& slack_analysis);

  // Finalize the DcnSlackSummary by converting total durations to averages.
  DcnSlackAnalysis Finalize();
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DCN_SLACK_ANALYSIS_COMBINER_H_
