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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_TFPROF_ADVICE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_TFPROF_ADVICE_H_

#include "tensorflow/core/profiler/internal/advisor/accelerator_utilization_checker.h"
#include "tensorflow/core/profiler/internal/advisor/checker.h"
#include "tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h"
#include "tensorflow/core/profiler/internal/advisor/internal_checker_runner.h"
#include "tensorflow/core/profiler/internal/advisor/operation_checker.h"
#include "tensorflow/core/profiler/tfprof_options.pb.h"

namespace tensorflow {
namespace tfprof {

// The Advisor runs a list of Checkers, each checks a specific area.
class Advisor {
 public:
  Advisor(const TFStats* stats) : stats_(stats) {}

  static AdvisorOptionsProto DefaultOptions() {
    AdvisorOptionsProto options;
    std::vector<string> checkers(
        kCheckers, kCheckers + sizeof(kCheckers) / sizeof(*kCheckers));
    for (const string& checker : checkers) {
      (*options.mutable_checkers())[checker];
    }
    return options;
  }

  AdviceProto Advise(const AdvisorOptionsProto& options) {
    // Note: Release a checker's memory ASAP.
    AdviceProto ret = RunInternalCheckers(options, stats_);

    if (options.checkers().find(kCheckers[0]) != options.checkers().end()) {
      AcceleratorUtilizationChecker au_checker;
      (*ret.mutable_checkers())[kCheckers[0]].MergeFrom(
          au_checker.Run(options.checkers().at(kCheckers[0]), stats_));
    }
    if (options.checkers().find(kCheckers[1]) != options.checkers().end()) {
      OperationChecker op_checker;
      (*ret.mutable_checkers())[kCheckers[1]].MergeFrom(
          op_checker.Run(options.checkers().at(kCheckers[1]), stats_));
    }
    if (options.checkers().find(kCheckers[2]) != options.checkers().end()) {
      ExpensiveOperationChecker expensive_op_checker;
      (*ret.mutable_checkers())[kCheckers[2]].MergeFrom(
          expensive_op_checker.Run(options.checkers().at(kCheckers[2]),
                                   stats_));
    }
    for (const auto& checker : ret.checkers()) {
      fprintf(stdout, "\n%s:\n", checker.first.c_str());
      for (const string& r : checker.second.reports()) {
        fprintf(stdout, "%s\n", r.c_str());
      }
    }
    fflush(stdout);
    return ret;
  }

 private:
  const TFStats* stats_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_TFPROF_ADVICE_H_
