/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROFILE_HANDLER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROFILE_HANDLER_H_

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// A profile handler collects event stats from a running step.
class ProfileHandler {
 public:
  ProfileHandler() {}
  virtual ~ProfileHandler() {}

  // Records that a single Op was executed in the current step.
  //
  // Implementations of this method must be thread-safe.
  //
  // Args:
  // - device: Device on which the Op was executed.
  // - stats: Statistics of node execution timing.
  // - is_copy: True if the op was a copy, send or recv.
  // - label: Extra content for timeline click text.
  // - op_type: String name of the Op.
  // - details: Main content for timeline click text.
  virtual void RecordOneOp(const string& device, const NodeExecStats& stats,
                           bool is_copy, StringPiece label, StringPiece op_type,
                           StringPiece details) = 0;

  // Records that the current step finished.
  //
  // Implementations of this method need not be thread-safe.
  //
  // Args:
  // - start_time: The time at which the step started.
  // - finish_time: The time at which the step finished.
  // - cleanup_time: The time at which cleanup for the step finished.
  // - total_runops: The number of ops that ran during this step.
  // - final_status: The status that this step finished with.
  virtual void StepDone(Microseconds start_time, Microseconds finish_time,
                        Microseconds cleanup_time, int total_runops,
                        absl::Status final_status) = 0;

  // Returns true if the caller should collect rpc activity.
  virtual bool should_collect_rpcs() = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROFILE_HANDLER_H_
