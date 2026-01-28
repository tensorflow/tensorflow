/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/tiling/tiling_specification.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<int64_t> TilingSpecification::ParameterIndex(
    const TilingSpecification::ParameterMapping& parameter_mapping,
    const HloInstruction* hlo, int64_t index) {
  int64_t offset = 0;
  for (const auto& [instruction, num_parameters] : parameter_mapping) {
    if (instruction == hlo) {
      if (index >= num_parameters) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Index ", index, " is out of bounds for instruction ",
            hlo->ToString(), " with num parameters ", num_parameters));
      }
      return offset + index;
    }

    offset += num_parameters;
  }
  return absl::NotFoundError(
      absl::StrCat("No tile sizes found for instruction: ", hlo->ToString()));
}

std::string TilingSpecification::ToString() const {
  std::string s = "TilingSpecification{\n";
  absl::StrAppend(&s, "  parameter_mapping={\n");
  for (const auto& [instruction, num_params] : parameter_mapping_) {
    absl::StrAppend(&s, "    ", instruction->name(), ": ", num_params, "\n");
  }
  absl::StrAppend(&s, "  }\n");
  absl::StrAppend(&s, "  constraints=", constraints_.ToString(), "\n");
  absl::StrAppend(&s, "}");
  return s;
}

absl::StatusOr<absl::Span<const int64_t>> Tiling::TileSizesForInstruction(
    const HloInstruction* hlo) const {
  if (auto it = tile_sizes_.find(hlo); it != tile_sizes_.end()) {
    return it->second;
  }

  return absl::NotFoundError(
      absl::StrCat("No tile sizes found for instruction: ", hlo->ToString()));
}

absl::StatusOr<FlatTiling> Tiling::Flatten(
    const TilingSpecification& tiling_specification) const {
  FlatTiling flat_tile_sizes;
  flat_tile_sizes.reserve(tiling_specification.num_parameters());
  for (const auto& mapping : tiling_specification.parameter_mapping()) {
    TF_ASSIGN_OR_RETURN(absl::Span<const int64_t> tile_sizes,
                        TileSizesForInstruction(mapping.instruction));
    if (tile_sizes.size() != mapping.num_tiling_parameters) {
      return absl::FailedPreconditionError(
          absl::StrCat("Instruction ", mapping.instruction->ToString(),
                       " was expected to have ", mapping.num_tiling_parameters,
                       " tile sizes but had ", tile_sizes.size(), "."));
    }
    flat_tile_sizes.insert(flat_tile_sizes.end(), tile_sizes.begin(),
                           tile_sizes.end());
  }

  return flat_tile_sizes;
}

/*static*/ absl::StatusOr<Tiling> Tiling::Unflatten(
    absl::Span<const int64_t> flat_tile_sizes,
    const TilingSpecification& tiling_specification) {
  if (flat_tile_sizes.size() != tiling_specification.num_parameters()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", tiling_specification.num_parameters(),
                     " tile sizes but got ", flat_tile_sizes.size(), "."));
  }

  TileMapping tile_mapping;
  int64_t offset = 0;
  for (const auto& [hlo, num_parameters] :
       tiling_specification.parameter_mapping()) {
    auto start_it = flat_tile_sizes.begin() + offset;
    auto end_it = start_it + num_parameters;
    tile_mapping[hlo] = {start_it, end_it};
    offset += num_parameters;
  }
  return Tiling(std::move(tile_mapping));
}

bool Tiling::ConformsTo(const TilingSpecification& tiling_specification) const {
  int64_t num_instructions = tile_sizes_.size();
  int64_t expected_num_instructions =
      tiling_specification.parameter_mapping().size();
  if (num_instructions != expected_num_instructions) {
    VLOG(1) << "Tiling tiles " << num_instructions << " instructions, but "
            << expected_num_instructions
            << " instructions were expected to be "
               "tiled.";
    return false;
  }

  // Linearization takes care of checking that we have the right number of
  // tile sizes specified for each instruction.
  absl::StatusOr<FlatTiling> flat_tile_sizes_or = Flatten(tiling_specification);
  if (!flat_tile_sizes_or.ok()) {
    return false;
  }

  return tiling_specification.constraints().IsSatisfiedBy(*flat_tile_sizes_or);
}

}  // namespace xla
