/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_STEP_INTERSECTION_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_STEP_INTERSECTION_H_

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

// Description of how two step sequences are aligned.
struct StepsAlignment {
  uint32 begin_subordinate_idx;  // where the alignment begins on the
                                 // subordinate steps.
  uint32 begin_chief_idx;  // where the alignment begins on the chief steps.
  uint32 num_steps;        // aligned for how many steps.
};

class StepIntersection {
 public:
  StepIntersection(
      uint32 max_steps,
      const absl::flat_hash_map</*host_id=*/uint32, const StepDatabaseResult*>&
          perhost_stepdb);

  // Returns the number of steps in the intersection.
  uint32 NumSteps() const { return end_chief_idx_ - begin_chief_idx_; }

  // Returns the value of empty_intersect_ (see the explanation of
  // empty_intersect_ below).
  bool EmptyIntersect() const { return empty_intersect_; }

  // Returns the step numbers for the destination (i.e. the intersection
  // result).
  std::vector<uint32> DstStepNumbers() const;

  // Returns the index to the step in the given host that corresponds to the
  // first step in the intersection.
  uint32 FirstStepIndex(uint32 host_id) const;

  // Returns the number of steps dropped due to the max_steps constraint
  // specified in the constructor.
  uint32 StepsDropped() const { return steps_dropped_; }

  std::string DebugString() const;

 private:
  absl::flat_hash_map</*host_id=*/uint32, StepsAlignment> perhost_alignment_;
  uint32
      chief_host_id_;  // the host whose step sequence is selected as the chief.
  uint32 steps_dropped_;  // number of steps dropped.
  // If NumSteps() is 0, empty_intersect indicates one of two possible reasons:
  //   (i) At least one host has some steps, but the intersection over all hosts
  //   is empty. In this case, empty_intersect is true,
  //   (ii) None of the hosts has any steps. In this case, empty_intersect is
  //   false.
  // If NumSteps() > 0, empty_intersect is don't care.
  bool empty_intersect_;
  // The begin and end indices to the chief step sequence for this step
  // intersection. Note that the begin index is inclusive but the end index is
  // exclusive.
  uint32 begin_chief_idx_;
  uint32 end_chief_idx_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_STEP_INTERSECTION_H_
