/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/scc.h"
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

// Data structure used to store data for Tarjan's Strongly Connected
// Components algorithm.
struct SCCNodeData {
  SCCNodeData()
      : node(nullptr),
        index(-1),
        lowlink(-1),
        onstack(false),
        caller(nullptr),
        caller_loop_location(-1) {}
  void ResetStack(int new_index, SCCNodeData* new_caller) {
    index = new_index;
    lowlink = new_index;
    onstack = true;
    caller = new_caller;
    caller_loop_location = 0;
  }
  const NodeDef* node;
  int index;
  int lowlink;
  bool onstack;
  std::vector<SCCNodeData*> children;
  // StrongConnect "call stack" storage.
  SCCNodeData* caller;       // Node calling StrongConnect
  int caller_loop_location;  // Index in parent StrongConnect for loop
};

// Core DFS step of Tarjan's Strongly Connected Component algorithm
// (implemented using iteration instead of recursion).
void StrongConnect(SCCNodeData* v, std::stack<SCCNodeData*>* stack, int* index,
                   std::unordered_map<const NodeDef*, int>* components,
                   int* scc_index) {
  // Iterative version of Tarjan's StrongConnect function.
  // The "call stack" state is composed of a SCCNodeData's caller and
  // caller_loop_location properties.
  v->ResetStack(*index /* index */, nullptr /* caller */);
  ++*index;
  stack->push(v);

  // No one put v on a StrongConnect call stack, reset caller values.
  v->caller = nullptr;
  v->caller_loop_location = 0;

  SCCNodeData* last = v;
  while (true) {
    if (last->caller_loop_location < last->children.size()) {
      // Recursive equivalent: Looping over the children of v (possibly
      // continuing at v->caller_loop_location after having finished a
      // recursive call.
      SCCNodeData* w = last->children[last->caller_loop_location];
      ++(last->caller_loop_location);  // For loop iterator increment
      if (w->index == -1) {
        w->ResetStack(*index /* index */, last /* caller */);
        ++*index;
        stack->push(w);
        last = w;
      } else if (w->onstack == true) {
        last->lowlink = std::min(last->lowlink, w->index);
      }
    } else {
      // At the end of v's children
      if (last->lowlink == last->index) {
        // v is the root of a strongly connected component
        SCCNodeData* top;
        while (true) {
          top = stack->top();
          stack->pop();
          top->onstack = false;
          (*components)[top->node] = *scc_index;
          if (top == last) {
            break;
          }
        }
        ++*scc_index;
      }

      // Go up the recursive call stack
      SCCNodeData* next_last = last->caller;
      if (next_last == nullptr) {
        // All nodes have been seen; finished.
        break;
      } else {
        next_last->lowlink = std::min(next_last->lowlink, last->lowlink);
        last = next_last;
      }
    }
  }
}

// This is an implementation of Tarjan's Strongly Connected Components
// DFS algorithm.  Most of the hard work is done in the function
// StrongConnect, which is an iterative reimplementation of the
// recursive version described here:
//   https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
//
// The edges for the purpose of this algorithm are directed from input
// to op (the reverse of the declarations of the NodeDef, which
// contain in-edges)
void StronglyConnectedComponents(
    const GraphDef& graph, std::unordered_map<const NodeDef*, int>* components,
    int* num_components) {
  std::stack<SCCNodeData*> stack;
  std::unordered_map<string, SCCNodeData*> name_to_data;
  std::vector<SCCNodeData> node_data_container;
  node_data_container.reserve(graph.node_size());
  std::unordered_map<const NodeDef*, SCCNodeData*> node_to_data;

  for (const NodeDef& node : graph.node()) {
    SCCNodeData node_data;
    node_data.node = &node;
    node_data_container.push_back(node_data);
    name_to_data[node.name()] = &(*node_data_container.rbegin());
    node_to_data[&node] = &(*node_data_container.rbegin());
  }

  // Create a list of top-level parents (add them to object queue)
  // Also create a mapping from nodes to their children.
  for (const NodeDef& node : graph.node()) {
    for (const string& input : node.input()) {
      name_to_data[NodeName(input)]->children.push_back(node_to_data[&node]);
    }
  }

  components->clear();
  *num_components = 0;
  int index = 0;
  for (auto& v : node_data_container) {
    if (v.index == -1) {
      // Node has not yet been visited.  Start a DFS at v.
      StrongConnect(&v, &stack, &index, components, num_components);
    }
  }

  std::vector<int> counts_per_component(*num_components, 0);
  for (auto& component : *components) {
    DCHECK(component.second >= 0);
    DCHECK(component.second < *num_components);
    counts_per_component[component.second]++;
  }
  for (auto& component : *components) {
    if (counts_per_component[component.second] == 1) {
      component.second = -1;
      (*num_components)--;
    }
  }
  (*num_components) += 1;
}

}  // namespace grappler
}  // namespace tensorflow
