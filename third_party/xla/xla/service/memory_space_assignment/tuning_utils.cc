/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/tuning_utils.h"

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

namespace memory_space_assignment {

void CustomizeSortedBufferInterval(
    std::optional<std::vector<uint64_t>> memory_space_assignment_config,
    std::vector<BufferInterval>& sorted_buffer_intervals) {
  // A copy of the sorted buffer intervals to assist the creating of the
  // customized buffer intervals vector respecting the config.
  std::vector<BufferInterval> sorted_buffer_intervals_copy(
      sorted_buffer_intervals);

  std::vector<uint64_t> config;
  if (!memory_space_assignment_config.has_value()) {
    config.resize(sorted_buffer_intervals_copy.size());
    absl::c_iota(config, 0);
  } else {
    config = *memory_space_assignment_config;
  }

  CHECK_EQ(config.size(), sorted_buffer_intervals_copy.size());
  sorted_buffer_intervals.clear();
  for (int i = 0; i < config.size(); ++i) {
    sorted_buffer_intervals.push_back(sorted_buffer_intervals_copy[config[i]]);
  }

  if (!sorted_buffer_intervals.empty()) {
    HloModule* module =
        sorted_buffer_intervals[0].buffer->instruction()->GetModule();

    // Update the memory space assignment auto-tuning config of a module with a
    // given config.
    module->mutable_config().mutable_memory_space_assignment_config()->assign(
        config.begin(), config.end());
  }
}

}  // namespace memory_space_assignment
}  // namespace xla
