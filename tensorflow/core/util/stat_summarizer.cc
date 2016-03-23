/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/util/stat_summarizer.h"

#include <iomanip>
#include <map>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

void StatSummarizer::ProcessStepStats(const StepStats& step_stats) {
  ++num_runs_;
  int64 curr_total = 0;
  for (const auto& ds : step_stats.dev_stats()) {
    for (const auto& ns : ds.node_stats()) {
      const string name = ns.node_name();
      const int64 curr_time = ns.all_end_rel_micros();
      curr_total += curr_time;
      int64 accum_time = timing_totals_[name];
      timing_totals_[name] = accum_time + curr_time;
    }
  }
  run_total_us_.UpdateStat(curr_total);

  timing_total_us_ += curr_total;
}

void StatSummarizer::PrintHeaders() {
  std::stringstream stream;
  stream << std::setw(40) << "[Name]"
         << "\t" << std::fixed << std::setprecision(2) << std::setw(7) << "[ms]"
         << "\t" << std::fixed << std::setprecision(2) << std::setw(6) << "[%]";
  LOG(INFO) << stream.str();
}

void StatSummarizer::PrintColumns(const char* name, const double time_ms,
                                  const double percentage) {
  std::stringstream stream;
  stream << std::setw(40) << name << "\t" << std::fixed << std::setprecision(2)
         << std::setw(7) << time_ms << "\t" << std::fixed
         << std::setprecision(2) << std::setw(6) << percentage;
  LOG(INFO) << stream.str();
}

void StatSummarizer::PrintStepStats() {
  const double avg_total_ms =
      timing_total_us_ / static_cast<double>(num_runs_) / 1000.0;

  LOG(INFO) << "Total time (us): " << run_total_us_;

  std::priority_queue<std::pair<double, string> > timings;
  LOG(INFO) << "========== Sorted by run order (ms) ==========";
  PrintHeaders();
  for (auto entry : timing_totals_) {
    const double avg_time_ms =
        entry.second / static_cast<double>(num_runs_) / 1000.0;

    const double overall_percentage = 100.0 * avg_time_ms / avg_total_ms;

    PrintColumns(entry.first.c_str(), avg_time_ms, overall_percentage);
    timings.push(std::pair<double, string>(avg_time_ms, entry.first));
  }
  LOG(INFO);

  LOG(INFO) << "============ Top by duration =================";
  PrintHeaders();
  int num_printed = 0;
  while (!timings.empty() && num_printed < 10) {
    auto entry = timings.top();
    timings.pop();

    const double overall_percentage = 100.0 * entry.first / avg_total_ms;
    PrintColumns(entry.second.c_str(), entry.first, overall_percentage);
    ++num_printed;
  }
  LOG(INFO);
}

}  // namespace tensorflow
