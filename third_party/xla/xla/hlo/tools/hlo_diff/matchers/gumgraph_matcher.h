/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_GUMGRAPH_MATCHER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_GUMGRAPH_MATCHER_H_

#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla::hlo_diff {

// Options allowing configuration of the instruction matching algorithm.
struct MatchOptions {
  bool use_top_down_matcher = true;
  bool debug_mode = false;

  // Threshold for "phase 0" matching in bipartite matcher.
  // If the lists of candidate instructions to match each contain a single
  // instruction, they are matched if their similarity is >= this threshold.
  //
  // How to pick a value:
  // - A higher value (e.g., 0.8) makes matching more conservative, requiring
  //   nodes to be highly similar (e.g., same shape, similar operands) to be
  //   matched.
  // - A lower value (e.g., 0.2) makes matching more aggressive, allowing
  //   nodes with different properties to be matched if they share the same
  //   opcode.
  //
  // A value of 0.5 is a sensible default because it strikes a balance between
  // avoiding false positives (matching unrelated nodes) and false negatives
  // (failing to match nodes that have undergone minor modifications).
  double phase_zero_threshold = 0.5;
};

// Base class for all node matchers. Each matcher implements a unique algorithm
// to match nodes between two HLO graphs. The base class standardizes input and
// output types, ensuring seamless integration and compatibility within any
// matcher sequence.
class HloGumgraphMatcher {
 public:
  virtual ~HloGumgraphMatcher() = default;
  virtual void Match(HloGumgraphMappings& mappings) const = 0;

 protected:
  HloGumgraphMatcher(MatcherType type, bool debug_mode)
      : type_(type), debug_mode_(debug_mode) {}
  const MatcherType type_;
  const bool debug_mode_;
};

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_GUMGRAPH_MATCHER_H_
