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
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/xla_data.pb.h"

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

FrontendAttributes MergeFrontendAttributes(
    absl::Span<HloInstruction* const> to_combine) {
  // Collect all unique values per key.
  absl::btree_map<std::string, absl::btree_set<std::string>> key_to_values;
  for (const HloInstruction* inst : to_combine) {
    if (!inst->has_frontend_attributes()) continue;
    for (const auto& [key, value] : inst->frontend_attributes().map()) {
      key_to_values[key].insert(value);
    }
  }

  // Build merged FrontendAttributes from the collected values.
  FrontendAttributes merged;
  for (const auto& [key, values] : key_to_values) {
    (*merged.mutable_map())[key] = absl::StrJoin(values, ",");
  }
  return merged;
}

namespace {

// Returns the longest common prefix of all strings, trimmed to the last '/'.
std::string CommonPrefix(absl::Span<const std::string> names) {
  if (names.empty()) return "";
  absl::string_view prefix = names.front();
  for (int64_t i = 1; i < names.size(); ++i) {
    prefix = absl::FindLongestCommonPrefix(prefix, names[i]);
  }
  // Trim to last '/' boundary so we don't split in the middle of a name.
  auto pos = prefix.rfind('/');
  if (pos != absl::string_view::npos) {
    return std::string(prefix.substr(0, pos + 1));
  }
  return "";
}

// Formats op_names as "common_prefix/(suffix1:suffix2:suffix3)".
// If all names are identical, returns the name as-is.
std::string MergeOpNames(absl::Span<const std::string> names) {
  if (names.empty()) return "";
  if (names.size() == 1) return names.front();

  std::string prefix = CommonPrefix(names);
  std::vector<std::string> suffixes;
  suffixes.reserve(names.size());
  for (const auto& name : names) {
    suffixes.push_back(name.substr(prefix.size()));
  }

  // If all suffixes are the same (all names identical), return as-is.
  bool all_same = true;
  for (int64_t i = 1; i < suffixes.size(); ++i) {
    if (suffixes[i] != suffixes[0]) {
      all_same = false;
      break;
    }
  }
  if (all_same) return names.front();

  return absl::StrCat(prefix, "(", absl::StrJoin(suffixes, ":"), ")");
}

}  // namespace

OpMetadata MergeMetadata(absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.empty()) return OpMetadata();

  // Start from the first instruction's metadata as base.
  OpMetadata merged = to_combine.front()->metadata();

  if (to_combine.size() == 1) return merged;

  // Collect source file:line pairs and op_names.
  std::vector<std::string> source_locations;
  std::vector<std::string> op_names;
  for (const HloInstruction* inst : to_combine) {
    const OpMetadata& md = inst->metadata();
    if (!md.source_file().empty()) {
      source_locations.push_back(
          absl::StrCat(md.source_file(), ":", md.source_line()));
    }
    if (!md.op_name().empty()) {
      op_names.push_back(md.op_name());
    }
  }

  // Concatenate source locations.
  if (!source_locations.empty()) {
    merged.set_source_file(absl::StrJoin(source_locations, ","));
    // Clear line numbers since they're embedded in the concatenated string.
    merged.set_source_line(0);
    merged.set_source_end_line(0);
  }

  // Merge op_names with common prefix extraction.
  if (!op_names.empty()) {
    merged.set_op_name(MergeOpNames(op_names));
  }

  return merged;
}

}  // namespace xla
