/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/spmd/sharding_format_picker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"

namespace xla {

class HloShardingTestHelper {
 public:
  static std::unique_ptr<HloSharding> CloneWithTileAssignment(
      const HloSharding& sharding, TileAssignment tile_assignment) {
    return std::unique_ptr<HloSharding>(
        new HloSharding(sharding, std::move(tile_assignment)));
  }
  static std::unique_ptr<HloSharding> Tuple(
      const std::vector<HloSharding>& sharding) {
    return std::unique_ptr<HloSharding>(new HloSharding(sharding));
  }
};

namespace test_only {
namespace {

bool PermuteDimsHelper(absl::Span<int64_t> dims, absl::Span<const int32_t> perm,
                       int start, const TileAssignment& tile_assignment,
                       TileAssignment* out) {
  if (start == dims.size() - 1) {
    TileAssignment v2(tile_assignment.dimensions(), dims, perm);
    if (v2 == tile_assignment) {
      *out = std::move(v2);
      return true;
    }
    return false;
  }
  using std::swap;
  if (PermuteDimsHelper(dims, perm, start + 1, tile_assignment, out)) {
    return true;
  }
  for (int i = start + 1; i < dims.size(); ++i) {
    if (dims[start] == dims[i]) {
      continue;
    }
    swap(dims[start], dims[i]);
    if (PermuteDimsHelper(dims, perm, start + 1, tile_assignment, out)) {
      return true;
    }
    swap(dims[start], dims[i]);
  }
  return false;
}

bool PermutePermHelper(absl::Span<int64_t> dims, absl::Span<int32_t> perm,
                       int start, const TileAssignment& tile_assignment,
                       TileAssignment* out) {
  if (start == dims.size() - 1) {
    return PermuteDimsHelper(dims, perm, 0, tile_assignment, out);
  }
  using std::swap;
  if (PermutePermHelper(dims, perm, start + 1, tile_assignment, out)) {
    return true;
  }
  for (int i = start + 1; i < perm.size(); ++i) {
    swap(perm[start], perm[i]);
    if (PermutePermHelper(dims, perm, start + 1, tile_assignment, out)) {
      return true;
    }
    swap(perm[start], perm[i]);
  }
  return false;
}

// Performs a brute force search to see if the sharding can be converted to V2.
// Returns the converted sharding if such transformation is possible and the
// sharding is not already V2.
std::unique_ptr<HloSharding> MaybeConvertToV2(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    std::vector<std::unique_ptr<HloSharding>> new_element_ptrs;
    new_element_ptrs.reserve(sharding.tuple_elements().size());
    bool changed = false;
    for (auto& element : sharding.tuple_elements()) {
      new_element_ptrs.push_back(MaybeConvertToV2(element));
      changed |= (new_element_ptrs.back() != nullptr);
    }
    if (!changed) return nullptr;
    std::vector<HloSharding> new_elements;
    new_elements.reserve(new_element_ptrs.size());
    for (int i = 0; i < new_element_ptrs.size(); ++i) {
      auto& ptr = new_element_ptrs[i];
      if (ptr) {
        new_elements.push_back(*ptr);
      } else {
        new_elements.push_back(sharding.tuple_elements()[i]);
      }
    }
    return HloShardingTestHelper::Tuple(new_elements);
  }
  auto& tile = sharding.tile_assignment();
  if (tile.iota() || sharding.IsReplicated() || sharding.IsTileMaximal() ||
      sharding.IsManual()) {
    return nullptr;
  }
  // Only brute force small number of devices.
  if (tile.num_elements() > 32 || tile.num_elements() < 2) return nullptr;
  const int32_t n = tile.num_elements();
  int32_t remain = n;
  std::vector<int64_t> prime_factors;
  for (int i = 2, r = std::max<int>(2, sqrt(n)); i <= r;) {
    if (remain % i == 0) {
      prime_factors.push_back(i);
      remain /= i;
      continue;
    }
    ++i;
  }
  if (remain > 1) {
    prime_factors.push_back(remain);
  }
  std::vector<int32_t> perm(prime_factors.size());
  absl::c_iota(perm, 0);
  TileAssignment new_tile;
  if (PermutePermHelper(absl::MakeSpan(prime_factors), absl::MakeSpan(perm), 0,
                        tile, &new_tile)) {
    return HloShardingTestHelper::CloneWithTileAssignment(sharding, new_tile);
  }
  return nullptr;
}

// Converts the sharding to V1 if it's not already V1, nullptr otherwise.
std::unique_ptr<HloSharding> MaybeConvertToV1(const HloSharding& sharding) {
  auto& tile = sharding.tile_assignment();
  if (!tile.iota()) {
    return nullptr;
  }
  return HloShardingTestHelper::CloneWithTileAssignment(
      sharding, TileAssignment(tile.shared_array()));
}

}  // namespace

absl::StatusOr<bool> ShardingFormatPicker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    for (HloInstruction* instruction : instructions) {
      if (!instruction->has_sharding()) {
        continue;
      }
      auto& sharding = instruction->sharding();
      std::unique_ptr<HloSharding> new_sharding;
      switch (sharding_type_) {
        case ShardingType::kV1:
          new_sharding = MaybeConvertToV1(sharding);
          break;
        case ShardingType::kBestEffortV2:
          new_sharding = MaybeConvertToV2(sharding);
          break;
      }
      if (new_sharding) {
        instruction->set_sharding(std::move(new_sharding));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace test_only
}  // namespace xla
