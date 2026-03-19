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
