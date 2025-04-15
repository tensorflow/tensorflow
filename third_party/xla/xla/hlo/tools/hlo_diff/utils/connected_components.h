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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_UTILS_CONNECTED_COMPONENTS_H_
#define XLA_HLO_TOOLS_HLO_DIFF_UTILS_CONNECTED_COMPONENTS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla {
namespace hlo_diff {

// Finds the connected components in an undirected graph of HloComputations.
class ConnectedComponentsFinder {
 public:
  // Add an edge between two computations
  void AddEdge(const HloComputation* u, const HloComputation* v);

  // Find and return the connected components
  std::vector<std::vector<const HloComputation*>> FindConnectedComponents();

 private:
  // Find the representative of the set (with path compression)
  const HloComputation* Find(const HloComputation* i);

  // Union the sets containing a and b (by making one parent the other)
  void Union(const HloComputation* a, const HloComputation* b);

  absl::flat_hash_map<const HloComputation*, const HloComputation*> parent_;
  absl::flat_hash_set<const HloComputation*> nodes_;
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_UTILS_CONNECTED_COMPONENTS_H_
