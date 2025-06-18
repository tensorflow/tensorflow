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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_HLO_CALL_GRAPH_MATCHER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_HLO_CALL_GRAPH_MATCHER_H_

#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla::hlo_diff {

// Matches similar computations between two HloGumgraphs (HloModules).
// Computations with the semantically same input parameters and output
// parameters are matched as `kSignature` matches. If a kSignature match
// additionally has semantically identical instructions, then its classified as
// a `kExact` matches.

// The matcher does not match the instructions within the matched computations
// which is the responsibility of the subequent matchers.
void MatchCallGraphs(const HloGumgraph& left, const HloGumgraph& right,
                     HloGumgraphMappings& mappings);

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_MATCHERS_HLO_CALL_GRAPH_MATCHER_H_
