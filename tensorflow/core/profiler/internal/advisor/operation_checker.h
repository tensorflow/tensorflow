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
// This checker checks common wrong configurations of operations.
//
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_OPERATION_CHECKER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_OPERATION_CHECKER_H_

#include "absl/strings/str_format.h"
#include "tensorflow/core/profiler/internal/advisor/checker.h"

namespace tensorflow {
namespace tfprof {

class OperationChecker : public Checker {
 public:
  string name() const override { return kCheckers[1]; }

 private:
  AdviceProto::Checker Check(const AdvisorOptionsProto::CheckerOption& options,
                             const TFStats* stats) override {
    if (!stats) {
      absl::FPrintF(
          stderr, "Missing profiles (e.g. graph, run_meta). Skip %s\n", name());
      return reports_;
    }
    bool use_batch_norm = false;
    bool use_fused_batch_norm = false;
    bool recommend_nchw = false;
    for (const auto& n : stats->nodes()) {
      const TFGraphNode* node = n.second.get();
      if (node->name().find("BatchNorm") != node->name().npos) {
        use_batch_norm = true;
      }
      if (node->op_types().find("FusedBatchNorm") != node->op_types().end()) {
        use_fused_batch_norm = true;
      }

      const AttrValue* attr = node->op_attrs("data_format");
      if (attr) {
        if (attr->s() == "NHWC" &&
            IsPlacedOnAccelerator(node->canonical_device())) {
          recommend_nchw = true;
        }
      }
    }
    if (use_batch_norm && !use_fused_batch_norm) {
      reports_.add_reports(
          "Maybe use faster FusedBatchNorm instead of BatchNorm");
    }
    if (recommend_nchw) {
      // TODO(xpan): Maybe print which Op supports NCHW.
      reports_.add_reports(
          "Found operation using NHWC data_format on GPU. Maybe "
          "NCHW is faster.");
    }
    return reports_;
  }

 private:
  AdviceProto::Checker reports_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_OPERATION_CHECKER_H_
