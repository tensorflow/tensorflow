/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
// This checker checks the accelerator's utilization.
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_ACCELERATOR_UTILIZATION_CHECKER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_ACCELERATOR_UTILIZATION_CHECKER_H_

#include "absl/strings/str_format.h"
#include "tensorflow/core/profiler/internal/advisor/checker.h"

namespace tensorflow {
namespace tfprof {

struct ExecStats {
 public:
  // Earliest start time of a step.
  int64 start_micros;
  // Latest finish time of a step.
  int64 end_micros;
  // The duration spent on running a kernel during a step.
  int64 exec_micros;
};

class AcceleratorUtilizationChecker : public Checker {
 public:
  string name() const override { return kCheckers[0]; }

 private:
  AdviceProto::Checker Check(const AdvisorOptionsProto::CheckerOption& options,
                             const TFStats* stats) override {
    if (!stats) {
      absl::FPrintF(
          stderr, "Missing profiles (e.g. graph, run_meta). Skip %s\n", name());
      return reports_;
    }
    for (const auto& n : stats->nodes()) {
      BuildExecStats(n.second.get());
    }
    return CheckInternal();
  }

  AdviceProto::Checker CheckInternal() {
    for (const auto& s : accelerator_exec_stats_) {
      const ExecStats& stat = s.second;
      int64 total_micros = stat.end_micros - stat.start_micros;
      if (total_micros <= 0) continue;
      double utilization = 1.0 * stat.exec_micros / total_micros;
      if (utilization >= 0.5) {
        reports_.add_reports(absl::StrFormat("device: %s utilization: %.2f",
                                             s.first, utilization));
      } else if (utilization < 0.5 && utilization > 0.2) {
        reports_.add_reports(absl::StrFormat("device: %s low utilization: %.2f",
                                             s.first, utilization));
      } else if (utilization <= 0.2) {
        reports_.add_reports(absl::StrFormat("device: %s low utilization: %.2f",
                                             s.first, utilization));
      }
    }
    return reports_;
  }

  void BuildExecStats(const TFGraphNode* node) {
    const auto& execs = node->all_op_execs();
    if (execs.empty()) {
      return;
    }
    if (!IsPlacedOnAccelerator(node->canonical_device())) {
      return;
    }

    if (accelerator_exec_stats_.find(node->canonical_device()) ==
        accelerator_exec_stats_.end()) {
      accelerator_exec_stats_.insert(
          std::pair<string, ExecStats>(node->canonical_device(), ExecStats()));
    }
    ExecStats& stats = accelerator_exec_stats_.at(node->canonical_device());

    // TODO(xpan): Use multiple steps?
    const ExecStep& exec = execs.rbegin()->second;

    if (stats.start_micros == 0) {
      stats.start_micros = exec.all_start_micros();
    } else if (exec.all_start_micros() != 0) {
      stats.start_micros =
          std::min(stats.start_micros, exec.all_start_micros());
    }
    stats.end_micros = std::max(stats.end_micros, exec.latest_end_micros());
    stats.exec_micros += exec.accelerator_exec_micros();
  }

  std::map<string, ExecStats> accelerator_exec_stats_;
  std::map<string, int64> ps_placement_;
  AdviceProto::Checker reports_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_ACCELERATOR_UTILIZATION_CHECKER_H_
