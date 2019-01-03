/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_

#include <functional>

#include "tensorflow/core/grappler/graph_topology_view.h"

namespace tensorflow {
namespace grappler {

enum class TraversalDirection { kFollowInputs, kFollowOutputs };

// Encapsulate DFS callbacks that will be called during the graph traversal.
//
// If non-empty, the `pre_order` and `post_order` functors will be called on
// each reachable node (including the `from` nodes) in pre and post order. If
// loops are found, the `on_back_edge` functor will be called on the
// corresponding back edges. Moreover, the pre and post order will assume that
// these back edges will be cut.
struct DfsCallbacks {
  DfsCallbacks() = default;
  DfsCallbacks(std::function<void(const NodeDef*)> pre,
               std::function<void(const NodeDef*)> post,
               std::function<void(const NodeDef*, const NodeDef*)> back_edge)
      : pre_order(std::move(pre)),
        post_order(std::move(post)),
        on_back_edge(std::move(back_edge)) {}

  static DfsCallbacks PreOrder(std::function<void(const NodeDef*)> pre) {
    return DfsCallbacks(std::move(pre), nullptr, nullptr);
  }

  static DfsCallbacks PostOrder(std::function<void(const NodeDef*)> post) {
    return DfsCallbacks(nullptr, std::move(post), nullptr);
  }

  std::function<void(const NodeDef*)> pre_order;
  std::function<void(const NodeDef*)> post_order;
  std::function<void(const NodeDef*, const NodeDef*)> on_back_edge;
};

// Encapsulate DFS predicates for traversing the graph.
//
// The `enter` predicate decides if traversal should enter the node, and the
// `advance` predicate decides if the traversal should follow inputs/outputs
// from the node.
//
// If predicates are empty (default initialized), it's assumed that we can enter
// into any node and advance from any node respectively.
struct DfsPredicates {
  DfsPredicates() = default;
  DfsPredicates(std::function<bool(const NodeDef*)> enter,
                std::function<bool(const NodeDef*)> advance)
      : enter(std::move(enter)), advance(std::move(advance)) {}

  static DfsPredicates Enter(std::function<bool(const NodeDef*)> enter) {
    return DfsPredicates(std::move(enter), nullptr);
  }

  static DfsPredicates Advance(std::function<bool(const NodeDef*)> advance) {
    return DfsPredicates(nullptr, std::move(advance));
  }

  std::function<bool(const NodeDef*)> enter;
  std::function<bool(const NodeDef*)> advance;
};

// Traverse the graph in DFS order in the given direction, starting from the
// list of nodes specified in the `from` argument. Use `predicates` to decide if
// traversal should enter/advance to/from the graph node. These predicates also
// applied to the `from` nodes. Call corresponding callbacks for each visited
// node.
void DfsTraversal(const GraphTopologyView& graph_view,
                  absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsPredicates& predicates,
                  const DfsCallbacks& callbacks);

// Traverse the graph in DFS order in the given direction, starting from the
// list of nodes specified in the `from` argument. Call corresponding callbacks
// for each visited node.
void DfsTraversal(const GraphTopologyView& graph_view,
                  absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsCallbacks& callbacks);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_TRAVERSAL_H_
