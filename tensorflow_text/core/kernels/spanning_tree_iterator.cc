// Copyright 2025 TF.Text Authors.
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

#include "tensorflow_text/core/kernels/spanning_tree_iterator.h"

namespace tensorflow {
namespace text {

SpanningTreeIterator::SpanningTreeIterator(bool forest) : forest_(forest) {}

bool SpanningTreeIterator::HasCycle(const SourceList &sources) {
  // Flags for whether each node has already been searched.
  searched_.assign(sources.size(), false);

  // Flags for whether the search is currently visiting each node.
  visiting_.assign(sources.size(), false);

  // Search upwards from each node to find cycles.
  for (uint32 initial_node = 0; initial_node < sources.size(); ++initial_node) {
    // Search upwards to try to find a cycle.
    uint32 current_node = initial_node;
    while (true) {
      if (searched_[current_node]) break;        // already searched
      if (visiting_[current_node]) return true;  // revisiting implies cycle
      visiting_[current_node] = true;  // mark as being currently visited
      const uint32 source_node = sources[current_node];
      if (source_node == current_node) break;  // self-loops are roots
      current_node = source_node;              // advance upwards
    }

    // No cycle; search upwards again to update flags.
    current_node = initial_node;
    while (true) {
      if (searched_[current_node]) break;  // already searched
      searched_[current_node] = true;
      visiting_[current_node] = false;
      const uint32 source_node = sources[current_node];
      if (source_node == current_node) break;  // self-loops are roots
      current_node = source_node;              // advance upwards
    }
  }

  return false;
}

uint32 SpanningTreeIterator::NumRoots(const SourceList &sources) {
  uint32 num_roots = 0;
  for (uint32 node = 0; node < sources.size(); ++node) {
    num_roots += (node == sources[node]);
  }
  return num_roots;
}

bool SpanningTreeIterator::NextSourceList(SourceList *sources) {
  const uint32 num_nodes = sources->size();
  for (uint32 i = 0; i < num_nodes; ++i) {
    const uint32 new_source = ++(*sources)[i];
    if (new_source < num_nodes) return true;  // absorbed in this digit
    (*sources)[i] = 0;  // overflowed this digit, carry to next digit
  }
  return false;  // overflowed the last digit
}

bool SpanningTreeIterator::NextTree(SourceList *sources) {
  // Iterate source lists, skipping non-trees.
  while (NextSourceList(sources)) {
    // Check the number of roots.
    const uint32 num_roots = NumRoots(*sources);
    if (forest_) {
      if (num_roots == 0) continue;
    } else {
      if (num_roots != 1) continue;
    }

    // Check for cycles.
    if (HasCycle(*sources)) continue;

    // Acyclic and rooted, therefore tree.
    return true;
  }
  return false;
}

}  // namespace text
}  // namespace tensorflow
