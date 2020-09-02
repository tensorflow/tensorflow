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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_STEP_INTERVAL_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_STEP_INTERVAL_H_

#include <algorithm>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace profiler {

struct StepInterval {
  uint32 step_num_begin = 0;
  uint32 step_num_end = kuint32max;

  void Intersect(const StepInterval& other) {
    step_num_begin = std::max(step_num_begin, other.step_num_begin);
    step_num_end = std::min(step_num_end, other.step_num_end);
  }

  void Invalidate() {
    step_num_begin = kuint32max;
    step_num_end = 0;
  }

  // reduce the total number steps and return number of steps dropped.
  uint32 LimitNumSteps(uint32 max_steps) {
    uint32 old_steps = step_num_end - step_num_begin + 1;
    uint32 new_steps = std::min(old_steps, max_steps);
    step_num_end = step_num_begin + new_steps - 1;
    return old_steps - new_steps;
  }

  bool Valid() const { return step_num_begin <= step_num_end; }

  uint32 NumSteps() const {
    return Valid() ? (step_num_end - step_num_begin + 1) : 0;
  }

  bool Contains(uint32 step_num) const {
    return step_num_begin <= step_num && step_num <= step_num_end;
  }

  uint32 Index(uint32 step_num) const {
    DCHECK(Contains(step_num));
    return step_num - step_num_begin;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_STEP_INTERVAL_H_
