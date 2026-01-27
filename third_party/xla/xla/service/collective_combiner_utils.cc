/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/collective_combiner_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

int64_t FindMostFrequentGatherDim(
    absl::Span<HloInstruction* const> to_combine) {
  assert(!to_combine.empty());

  // Count frequencies.
  int64_t min_rank = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> frequency;
  for (const HloInstruction* it : to_combine) {
    int64_t dim = Cast<HloAllGatherInstruction>(it)->all_gather_dimension();
    frequency.resize(std::max(dim + 1, static_cast<int64_t>(frequency.size())),
                     0);
    ++frequency[dim];
    min_rank = std::min(min_rank,
                        static_cast<int64_t>(it->shape().dimensions().size()));
  }

  int64_t most_frequent_dim = std::distance(
      frequency.begin(), std::max_element(frequency.begin(), frequency.end()));
  return most_frequent_dim < min_rank ? most_frequent_dim : 0;
}

}  // namespace xla
