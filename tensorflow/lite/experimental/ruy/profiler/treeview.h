/* Copyright 2020 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_TREEVIEW_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_TREEVIEW_H_

#ifdef RUY_PROFILER

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

namespace ruy {
namespace profiler {

// A tree view of a profile.
class TreeView {
 public:
  struct Node {
    std::vector<std::unique_ptr<Node>> children;
    Label label;
    int weight = 0;
  };

  void Populate(const std::vector<char>& samples_buf_);

  // Intentionally an *ordered* map so that threads are enumerated
  // in an order that's consistent and typically putting the 'main thread'
  // first.
  using ThreadRootsMap = std::map<std::uint32_t, std::unique_ptr<Node>>;

  const ThreadRootsMap& thread_roots() const { return thread_roots_; }
  ThreadRootsMap* mutable_thread_roots() { return &thread_roots_; }

 private:
  ThreadRootsMap thread_roots_;
};

/* Below are API functions for manipulating and printing treeviews. */

// Prints the treeview to stdout.
void Print(const TreeView& treeview);

// Prints the treeview below the given node on stdout.
void PrintTreeBelow(const TreeView::Node& node);

// Returns the tree depth below the given node.
int DepthOfTreeBelow(const TreeView::Node& node);

// Returns the sum of weights of nodes below the given node and filtered by
// the `match` predicate.
int WeightBelowNodeMatchingFunction(
    const TreeView::Node& node, const std::function<bool(const Label&)>& match);

// Returns the sum of weights of nodes below the given node and whose
// unformatted label (i.e. raw format string) matches the given `format` string.
//
// This allows to aggregate nodes whose labels differ only by parameter values.
int WeightBelowNodeMatchingUnformatted(const TreeView::Node& node,
                                       const std::string& format);

// Returns the sum of weights of nodes below the given node and whose formatted
// label matches the `formatted` string.
//
// In the case of nodes with parametrized labels, this allows to count only
// nodes with specific parameter values. For that purpose, one may also instead
// use WeightBelowNodeMatchingFunction directly, with a `match` predicate
// comparing raw integer parameter values directly, instead of going through
// formatted strings.
int WeightBelowNodeMatchingFormatted(const TreeView::Node& node,
                                     const std::string& formatted);

// Produces a `node_out` that is a copy of `node_in` but with tree depth below
// it clamped at `depth`, with further subtrees aggregated into single leaf
// nodes.
void CollapseNode(const TreeView::Node& node_in, int depth,
                  TreeView::Node* node_out);

// Calls CollapseNode with the given `depth` on every subnode filtered by the
// `match` predicate. Note that this does NOT limit the tree depth below
// `node_out` to `depth`, since each collapsed node below `node_out` may be
// arbitrarily far below it and `depth` is only used as the collapsing depth
// at that point.
void CollapseSubnodesMatchingFunction(
    const TreeView::Node& node_in, int depth,
    const std::function<bool(const Label&)>& match, TreeView::Node* node_out);

// Calls CollapseNode with the given `depth` on every node filtered by the
// `match` predicate. Note that this does NOT limit the tree depth below
// `node_out` to `depth`, since each collapsed node below `node_out` may be
// arbitrarily far below it and `depth` is only used as the collapsing depth
// at that point.
void CollapseNodesMatchingFunction(
    const TreeView& treeview_in, int depth,
    const std::function<bool(const Label&)>& match, TreeView* treeview_out);

// Special case of CollapseNodesMatchingFunction matching unformatted labels,
// i.e. raw format strings.
// See the comment on WeightBelowNodeMatchingUnformatted.
void CollapseNodesMatchingUnformatted(const TreeView& treeview_in, int depth,
                                      const std::string& format,
                                      TreeView* treeview_out);

// Special case of CollapseNodesMatchingFunction matching formatted labels.
// See the comment on WeightBelowNodeMatchingFormatted.
void CollapseNodesMatchingFormatted(const TreeView& treeview_in, int depth,
                                    const std::string& formatted,
                                    TreeView* treeview_out);

}  // namespace profiler
}  // namespace ruy

#endif  // RUY_PROFILER

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_TREEVIEW_H_
