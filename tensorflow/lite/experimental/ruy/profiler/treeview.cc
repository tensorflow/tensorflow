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

#ifdef RUY_PROFILER

#include "tensorflow/lite/experimental/ruy/profiler/treeview.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

namespace ruy {
namespace profiler {

namespace {

void SortNode(TreeView::Node* node) {
  using NodePtr = std::unique_ptr<TreeView::Node>;
  std::sort(node->children.begin(), node->children.end(),
            [](const NodePtr& n1, const NodePtr& n2) {
              return n1->weight > n2->weight;
            });
  for (const auto& child : node->children) {
    SortNode(child.get());
  }
}

// Records a stack i.e. a sample in a treeview, by incrementing the weights
// of matching existing nodes and/or by creating new nodes as needed,
// recursively, below the given node.
void AddStack(const detail::Stack& stack, TreeView::Node* node, int level) {
  node->weight++;
  if (stack.size == level) {
    return;
  }
  TreeView::Node* child_to_add_to = nullptr;
  for (const auto& child : node->children) {
    if (child->label == stack.labels[level]) {
      child_to_add_to = child.get();
      break;
    }
  }
  if (!child_to_add_to) {
    child_to_add_to = node->children.emplace_back(new TreeView::Node).get();
    child_to_add_to->label = stack.labels[level];
  }
  AddStack(stack, child_to_add_to, level + 1);
}

// Recursively populates the treeview below the given node with 'other'
// entries documenting for each node the difference between its weight and the
// sum of its children's weight.
void AddOther(TreeView::Node* node) {
  int top_level_children_weight = 0;
  for (const auto& child : node->children) {
    AddOther(child.get());
    top_level_children_weight += child->weight;
  }
  if (top_level_children_weight != 0 &&
      top_level_children_weight != node->weight) {
    const auto& new_child = node->children.emplace_back(new TreeView::Node);
    new_child->label = Label("[other]");
    new_child->weight = node->weight - top_level_children_weight;
  }
}

}  // namespace

void TreeView::Populate(const std::vector<char>& samples_buf_) {
  thread_roots_.clear();
  // Populate the treeview with regular nodes coming from samples.
  const char* buf_ptr = samples_buf_.data();
  const char* const buf_ptr_end = buf_ptr + samples_buf_.size();
  while (buf_ptr < buf_ptr_end) {
    detail::Stack stack;
    detail::ReadFromBuffer(buf_ptr, &stack);
    // Empty stacks should have been dropped during sampling.
    assert(stack.size > 0);
    buf_ptr += GetBufferSize(stack);
    const int id = stack.id;
    if (!thread_roots_[id]) {
      thread_roots_[id].reset(new Node);
    }
    AddStack(stack, thread_roots_[id].get(), 0);
  }
  // Populate the treeview with additional 'other' nodes, sort, and set
  // root labels.
  for (const auto& thread_root : thread_roots_) {
    std::uint32_t id = thread_root.first;
    Node* root = thread_root.second.get();
    AddOther(root);
    SortNode(root);
    root->label.Set("Thread %x (%d samples)", id, root->weight);
  }
}

// Recursively prints the treeview below the given node. The 'root' node
// argument is only needed to compute weights ratios, with the root ratio
// as denominator.
void PrintTreeBelow(const TreeView::Node& node, const TreeView::Node& root,
                    int level) {
  if (&node == &root) {
    printf("%s\n\n", node.label.Formatted().c_str());
  } else {
    for (int i = 1; i < level; i++) {
      printf("  ");
    }
    printf("* %.2f%% %s\n", 100.0f * node.weight / root.weight,
           node.label.Formatted().c_str());
  }
  for (const auto& child : node.children) {
    PrintTreeBelow(*child, root, level + 1);
  }
}

void Print(const TreeView& treeview) {
  printf("\n");
  printf("Profile (%d threads):\n\n",
         static_cast<int>(treeview.thread_roots().size()));
  for (const auto& thread_root : treeview.thread_roots()) {
    const TreeView::Node& root = *thread_root.second;
    PrintTreeBelow(root, root, 0);
    printf("\n");
  }
}

int DepthOfTreeBelow(const TreeView::Node& node) {
  if (node.children.empty()) {
    return 0;
  } else {
    int max_child_depth = 0;
    for (const auto& child : node.children) {
      max_child_depth = std::max(max_child_depth, DepthOfTreeBelow(*child));
    }
    return 1 + max_child_depth;
  }
}

int WeightBelowNodeMatchingFunction(
    const TreeView::Node& node,
    const std::function<bool(const Label&)>& match) {
  int weight = 0;
  if (match(node.label)) {
    weight += node.weight;
  }
  for (const auto& child : node.children) {
    weight += WeightBelowNodeMatchingFunction(*child, match);
  }
  return weight;
}

int WeightBelowNodeMatchingUnformatted(const TreeView::Node& node,
                                       const std::string& format) {
  return WeightBelowNodeMatchingFunction(
      node, [&format](const Label& label) { return label.format() == format; });
}

int WeightBelowNodeMatchingFormatted(const TreeView::Node& node,
                                     const std::string& formatted) {
  return WeightBelowNodeMatchingFunction(
      node, [&formatted](const Label& label) {
        return label.Formatted() == formatted;
      });
}

void CollapseNode(const TreeView::Node& node_in, int depth,
                  TreeView::Node* node_out) {
  node_out->label = node_in.label;
  node_out->weight = node_in.weight;
  node_out->children.clear();
  if (depth > 0) {
    for (const auto& child_in : node_in.children) {
      auto* child_out = new TreeView::Node;
      node_out->children.emplace_back(child_out);
      CollapseNode(*child_in, depth - 1, child_out);
    }
  }
}

void CollapseSubnodesMatchingFunction(
    const TreeView::Node& node_in, int depth,
    const std::function<bool(const Label&)>& match, TreeView::Node* node_out) {
  if (match(node_in.label)) {
    CollapseNode(node_in, depth, node_out);
  } else {
    node_out->label = node_in.label;
    node_out->weight = node_in.weight;
    node_out->children.clear();

    for (const auto& child_in : node_in.children) {
      auto* child_out = new TreeView::Node;
      node_out->children.emplace_back(child_out);
      CollapseSubnodesMatchingFunction(*child_in, depth, match, child_out);
    }
  }
}

void CollapseNodesMatchingFunction(
    const TreeView& treeview_in, int depth,
    const std::function<bool(const Label&)>& match, TreeView* treeview_out) {
  treeview_out->mutable_thread_roots()->clear();
  for (const auto& thread_root_in : treeview_in.thread_roots()) {
    std::uint32_t id = thread_root_in.first;
    const auto& root_in = *thread_root_in.second;
    auto* root_out = new TreeView::Node;
    treeview_out->mutable_thread_roots()->emplace(id, root_out);
    CollapseSubnodesMatchingFunction(root_in, depth, match, root_out);
  }
}

void CollapseNodesMatchingUnformatted(const TreeView& treeview_in, int depth,
                                      const std::string& format,
                                      TreeView* treeview_out) {
  CollapseNodesMatchingFunction(
      treeview_in, depth,
      [&format](const Label& label) { return label.format() == format; },
      treeview_out);
}

void CollapseNodesMatchingFormatted(const TreeView& treeview_in, int depth,
                                    const std::string& formatted,
                                    TreeView* treeview_out) {
  CollapseNodesMatchingFunction(
      treeview_in, depth,
      [&formatted](const Label& label) {
        return label.Formatted() == formatted;
      },
      treeview_out);
}

}  // namespace profiler
}  // namespace ruy

#endif  // RUY_PROFILER
