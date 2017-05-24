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
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"

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
  // TODO(xpan): Test it.
  if (RE2::FullMatch(dev, "/job:.*/replica:\\d+/task:\\d+/[a-z]+:\\d+")) {
    canonical_device_ = dev;
    // TODO(xpan): Support things other than gpu?
    host_device_ = StringReplace(dev, "gpu:\\d+", "cpu:0");
  }

  devices_.insert(dev);
  if (step_stat_->all_start_micros() > 0) {
    if (all_start_micros_ > 0) {
      all_start_micros_ =
          std::min(all_start_micros_,
                   static_cast<int64>(step_stat_->all_start_micros()));
    } else {
      all_start_micros_ = step_stat_->all_start_micros();
    }
    int64 op_end_rel_micros = step_stat_->op_end_rel_micros();
    // Round quick execution to 1 micro to be semantically robust.
    if (op_end_rel_micros == 0) {
      ++op_end_rel_micros;
    }
    latest_end_rel_micros_ =
        std::max(latest_end_rel_micros_, op_end_rel_micros);

    op_execs_[dev].push_back(
        std::make_pair(step_stat_->all_start_micros(), op_end_rel_micros));

    if (dev.find("stream") != dev.npos && dev.find("stream:all") == dev.npos) {
      gpu_kernel_execs_[dev].push_back(
          std::make_pair(step_stat_->all_start_micros(), op_end_rel_micros));
    }
  }

  if (dev == canonical_device_) {
    for (const auto& mem : step_stat_->memory()) {
      // TODO(xpan): Fix this hack. Currently the allocator name seems quite
      // ad-hoc.
      if (mem.allocator_name().find("GPU") == mem.allocator_name().npos) {
        continue;
      }
      if (dev == canonical_device_) {
        allocator_bytes_in_use_ =
            std::max(allocator_bytes_in_use_,
                     static_cast<int64>(mem.allocator_bytes_in_use()));
      }
    }
    int64 total_output_bytes = 0;
    for (const auto& output : step_stat_->output()) {
      if (output.has_tensor_description() &&
          output.tensor_description().has_allocation_description()) {
        // TODO(xpan): Maybe allocated_bytes.
        int64 output_bytes = std::max(output.tensor_description()
                                          .allocation_description()
                                          .allocated_bytes(),
                                      output.tensor_description()
                                          .allocation_description()
                                          .requested_bytes());
        uint64 output_ptr =
            output.tensor_description().allocation_description().ptr();
        total_output_bytes += output_bytes;
        output_bytes_[output.slot()] = std::make_pair(output_bytes, output_ptr);
      }
    }
    if (step_stat_->has_memory_stats()) {
      host_temp_bytes_ += step_stat_->memory_stats().host_temp_memory_size();
      host_persistent_bytes_ +=
          step_stat_->memory_stats().host_persistent_memory_size();
      accelerator_temp_bytes_ +=
          step_stat_->memory_stats().device_temp_memory_size();
      accelerator_persistent_bytes_ +=
          step_stat_->memory_stats().device_persistent_memory_size();
    }
    requested_bytes_ = total_output_bytes;
  }
}
}  // namespace tfprof
}  // namespace tensorflow
