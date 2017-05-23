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

#include "tensorflow/tools/tfprof/internal/tfprof_node.h"

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"

namespace tensorflow {
namespace tfprof {
// Notes about start and end time from the NodeExecStats proto.
// For GPU, there is no difference between op_end_rel_micros and
// all_end_rel_micros. All are kernel times.
// For CPU, op_end_rel is the kernel time, while all_end_rel_micros includes
// some post-processing.
// Here, we only consider kernel time for simplicity.
void TFGraphNode::AddStepStat(const string& device,
                              const NodeExecStats* step_stat) {
  step_stat_ = step_stat;
  CHECK(step_stat_);

  string dev = str_util::Lowercase(device);

  devices_.insert(dev);
  op_kernel_execs_[dev].push_back(std::make_pair(
      step_stat_->all_start_micros(), step_stat_->op_end_rel_micros()));

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
