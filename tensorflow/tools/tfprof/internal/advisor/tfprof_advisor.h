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

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_TFPROF_ADVICE_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_TFPROF_ADVICE_H_

#include "tensorflow/tools/tfprof/internal/advisor/accelerator_utilization_checker.h"
#include "tensorflow/tools/tfprof/internal/advisor/checker.h"
#include "tensorflow/tools/tfprof/internal/advisor/internal_checker_runner.h"
#include "tensorflow/tools/tfprof/internal/advisor/operation_checker.h"

namespace tensorflow {
namespace tfprof {

// The Advisor runs a list of Checkers, each checks a specific area.
class Advisor {
 public:
  Advisor(const TFStats* stats) : stats_(stats) {}

  std::map<string, std::vector<string>> Advise() {
    // Note: Release a checker's memory ASAP.
    std::map<string, std::vector<string>> reports = RunInternalCheckers(stats_);
    // TODO(xpan): Think of a way to turn off/on specific checkers.
    AcceleratorUtilizationChecker au_checker;
    reports[au_checker.name()] = au_checker.Run(stats_);
    OperationChecker op_checker;
    reports[op_checker.name()] = op_checker.Run(stats_);

    for (const auto& checker_r : reports) {
      fprintf(stdout, "%s reports:\n", checker_r.first.c_str());
      for (const auto& r : checker_r.second) {
        fprintf(stdout, "%s\n", r.c_str());
      }
    }
    fflush(stdout);
    return reports;
  }

 private:
  const TFStats* stats_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_TFPROF_ADVICE_H_
