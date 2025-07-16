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

#include "xla/hlo/utils/hlo_longest_prefix.h"

#include <cstddef>
#include <optional>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/util.h"

namespace xla {
namespace hlo_longest_prefix {

namespace {

// Split `a` and `b` by `delim` into two lists of possibly-empty tokens, then
// rejoin the first N of those lists that match by `delim`. Note: it is
// unspecified which argument the return value points into.
absl::string_view LongestPrefix(absl::string_view a, absl::string_view b,
                                char delim = '/') {
  auto split_a = absl::StrSplit(a, delim);
  auto split_b = absl::StrSplit(b, delim);

  size_t common_prefix_len = 0;

  for (auto a_it = split_a.begin(), b_it = split_b.begin();
       a_it != split_a.end() && b_it != split_b.end(); ++a_it, ++b_it) {
    if (*a_it != *b_it) {
      break;
    }
    if (common_prefix_len) {
      ++common_prefix_len;  // account for delimiter
    }
    common_prefix_len += a_it->size();  // length of a matching token
  }

  return absl::string_view(a.data(), common_prefix_len);
}

// Find the longest prefix among instructions' op_name metadata
// Chunk this by delimiting slashes, i.e. given a/b/cat and a/b/cabbage, the
// longest prefix is a/b not a/b/ca
class OpNamePrefixVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit OpNamePrefixVisitor(bool ignore_malformed_op_names = false)
      : ignore_malformed_op_names_(ignore_malformed_op_names) {}

  absl::Status DefaultAction(const HloInstruction* inst) final {
    // Ignore empty op_name, op_name not contains a slash, or op_name starting
    // with a slash, those are generally not valid op_name.
    auto const& op_name = inst->metadata().op_name();
    if (IsValidOpName(op_name)) {
      auto const& original_prefix = prefix_.value_or("");
      prefix_ = prefix_ ? LongestPrefix(*prefix_, op_name) : op_name;

      VLOG(9) << "OpNamePrefixVisitor::DefaultAction for HloInstruction:"
              << inst->name() << ", metadata's op_name:" << op_name
              << ", original prefix:" << original_prefix
              << ", new prefix:" << prefix_.value_or("");
    }
    return absl::OkStatus();
  }

  absl::string_view longest_op_name_prefix() const {
    return prefix_.value_or("");
  }

 private:
  bool IsValidOpName(absl::string_view op_name) const {
    return !op_name.empty() &&
           (!ignore_malformed_op_names_ || (absl::StrContains(op_name, "/") &&
                                            !absl::StartsWith(op_name, "/")));
  }

  bool ignore_malformed_op_names_ = false;
  std::optional<absl::string_view> prefix_;
};

}  // namespace

absl::string_view GetLongestOpNamePrefix(const HloInstruction& inst,
                                         bool ignore_malformed_op_names) {
  OpNamePrefixVisitor visitor(ignore_malformed_op_names);
  if (!VisitInstAndCalledButNotOperands(visitor, inst).ok()) {
    return "";
  }
  return visitor.longest_op_name_prefix();
}

absl::string_view GetLongestOpNamePrefix(const HloModule& mod,
                                         bool ignore_malformed_op_names) {
  // In the presence of (at least) debug callbacks, calling Accept on the root
  // instruction of the module may not reach all instructions in the module.
  OpNamePrefixVisitor visitor(ignore_malformed_op_names);
  for (const HloComputation* computation : mod.computations()) {
    for (const HloInstruction* inst : computation->instructions()) {
      if (!visitor.DefaultAction(inst).ok()) {
        return {};
      }
    }
  }
  return visitor.longest_op_name_prefix();
}

}  // namespace hlo_longest_prefix
}  // namespace xla
