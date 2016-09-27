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

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_node.h"

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"

namespace tensorflow {
namespace tfprof {
void TFNode::AddStepStat(const string& device, const NodeExecStats* step_stat) {
  if (!device.empty()) {
    // This might override device from GraphDef.
    device_ = device;
  }
  step_stat_ = step_stat;

  op_start_micros_ = step_stat_->all_start_micros();
  if (step_stat_->op_end_rel_micros() && step_stat_->op_start_rel_micros()) {
    op_exec_micros_ =
        step_stat_->op_end_rel_micros() - step_stat_->op_start_rel_micros();
  }
  all_spent_micros_ = step_stat_->all_end_rel_micros();

  for (const auto& output : step_stat_->output()) {
    if (output.has_tensor_description() &&
        output.tensor_description().has_allocation_description()) {
      requested_bytes_ += output.tensor_description()
                              .allocation_description()
                              .requested_bytes();
    }
  }
}
}  // namespace tfprof
}  // namespace tensorflow
