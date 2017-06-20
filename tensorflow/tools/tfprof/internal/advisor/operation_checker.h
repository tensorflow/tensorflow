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
#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_OPERATION_CHECKER_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_OPERATION_CHECKER_H_

#include "tensorflow/tools/tfprof/internal/advisor/checker.h"

namespace tensorflow {
namespace tfprof {

class OperationChecker : public Checker {
 public:
  string name() override { return "OperationChecker"; }

 private:
  std::vector<string> Check(const TFStats* stats) override {
    if (!stats) {
      fprintf(stderr, "Missing profiles (e.g. graph, run_meta). Skip %s\n",
              name().c_str());
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
      if (node->op_attrs().find("data_format") != node->op_attrs().end()) {
        const AttrValue* attr_val = node->op_attrs().at("data_format");
        if (attr_val->s() == "NHWC" &&
            IsAcceleratorDevice(node->canonical_device())) {
          recommend_nchw = true;
        }
      }
    }
    if (use_batch_norm && !use_fused_batch_norm) {
      reports_.push_back(strings::Printf(
          "%s: Maybe use faster FusedBatchNorm instead of BatchNorm",
          kLevel[1]));
    }
    if (recommend_nchw) {
      // TODO(xpan): Maybe print which Op supports NCHW.
      reports_.push_back(strings::Printf(
          "%s: Found operation using NHWC data_format on GPU. Maybe "
          "NCHW is faster.",
          kLevel[1]));
    }
    return reports_;
  }

 private:
  std::vector<string> reports_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_OPERATION_CHECKER_H_
