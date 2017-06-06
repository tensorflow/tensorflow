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
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {
// Notes about start and end time from the NodeExecStats proto.
// For GPU, there is no difference between op_end_rel_micros and
// all_end_rel_micros. All are kernel times.
// For CPU, op_end_rel is the kernel time, while all_end_rel_micros includes
// some post-processing.
// Here, we only consider kernel time for simplicity.
void TFGraphNode::AddStepStat(int64 step, const string& device,
                              const NodeExecStats& step_stat) {
  string dev = str_util::Lowercase(device);

  // TODO(xpan): Test it.
  if (RE2::FullMatch(dev, "/job:.*/replica:\\d+/task:\\d+/[a-z]+:\\d+")) {
    if (!canonical_device_.empty()) {
      if (canonical_device_ != dev) {
        fprintf(stderr, "Unexpected: graph node changed device: %s->%s.\n",
                canonical_device_.c_str(), dev.c_str());
        return;
      }
    } else {
      canonical_device_ = dev;
      // TODO(xpan): Support things other than gpu?
      host_device_ = StringReplace(dev, "gpu:\\d+", "cpu:0");
      AddOpType(canonical_device_);
    }
  }

  ExecStep& exec = execs_[step];
  exec.AddTimeStats(dev, step_stat);

  if (dev == canonical_device_) {
    exec.AddMemoryStats(dev, step_stat);
  }
}
}  // namespace tfprof
}  // namespace tensorflow
