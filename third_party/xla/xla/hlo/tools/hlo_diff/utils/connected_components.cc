// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/utils/connected_components.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla {
namespace hlo_diff {

// Find the representative of the set (with path compression)
const HloComputation* ConnectedComponentsFinder::Find(const HloComputation* i) {
  if (parent_.find(i) == parent_.end() || parent_[i] == i) {
    parent_[i] = i;
    return i;
  }
  return parent_[i] = Find(parent_[i]);  // Path compression
}

// Union the sets containing a and b (by making one parent the other)
void ConnectedComponentsFinder::Union(const HloComputation* a,
                                      const HloComputation* b) {
  const HloComputation* root_a = Find(a);
  const HloComputation* root_b = Find(b);
  if (root_a != root_b) {
    parent_[root_a] = root_b;
  }
}

// Add an edge between two computations
void ConnectedComponentsFinder::AddEdge(const HloComputation* u,
                                        const HloComputation* v) {
  nodes_.insert(u);
  nodes_.insert(v);
  Union(u, v);
}

// Find and return the connected components
std::vector<std::vector<const HloComputation*>>
ConnectedComponentsFinder::FindConnectedComponents() {
  absl::flat_hash_map<const HloComputation*, std::vector<const HloComputation*>>
      components;
  for (const auto& node : nodes_) {
    components[Find(node)].push_back(node);
  }

  std::vector<std::vector<const HloComputation*>> result;
  result.reserve(components.size());
  for (auto& [root, component_nodes] : components) {
    result.push_back(component_nodes);
  }
  return result;
}

}  // namespace hlo_diff
}  // namespace xla
