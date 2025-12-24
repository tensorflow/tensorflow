/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TUPLE_TREE_H_
#define XLA_TUPLE_TREE_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

// Forward declaration for friending.
template <typename T>
class TupleTree;

namespace internal {

// Index table for TupleTree.
class IndexTable {
 public:
  struct Entry {
    // Index in the TupleTree::nodes_ vector.
    size_t node_id;
    // Index of the first child of this node in the entries_ vector. -1 for
    // nodes with no children array (i.e., Node::IsLeaf() is true).
    std::make_signed_t<size_t> children_start_id = -1;
    // Number of children. This is necessary for bounds checking in GetEntry,
    // as TupleTree doesn't have a separate structure definition like Shape.
    size_t num_children = 0;
  };

  IndexTable() = default;

  template <typename T>
  static IndexTable Create(const typename TupleTree<T>::Node& root) {
    IndexTable table;
    table.Initialize<T>(root);
    return table;
  }

  explicit IndexTable(const Shape& shape);

  absl::StatusOr<const Entry*> GetEntry(ShapeIndexView index) const;

  const std::optional<absl::InlinedVector<Entry, 1>>& entries() const {
    return entries_;
  }

  static absl::StatusOr<IndexTable> CreateFromSubtree(
      const IndexTable& original_table, const ShapeIndex& index);

  // Checks if two subtrees rooted at the given entries are structurally
  // compatible.
  static absl::Status IsSubtreeCompatible(const IndexTable& other_table,
                                          const Entry* other_entry,
                                          const IndexTable& this_table,
                                          const Entry* this_entry);

  // Counts the number of nodes in the subtree rooted at root_entry.
  static size_t CountSubtreeNodes(const IndexTable& table,
                                  const Entry* root_entry);

 private:
  // Computes the total size of all nested tuples in the given tuple shape.
  static size_t IndexTableTuplesSize(const Shape& shape);

  // Initializes the index table in the given entries span. Span must point into
  // the appropriately sized entries storage.
  static void InitializeIndexTable(const Shape& shape,
                                   absl::Span<IndexTable::Entry> entries,
                                   size_t entry_index, size_t& next_node_id,
                                   size_t& next_children_start_index);

  template <typename U>
  static size_t CountNodes(const typename TupleTree<U>::Node& node) {
    size_t count = 1;
    if (!node.IsLeaf()) {
      for (const typename TupleTree<U>::Node& child : node.children()) {
        count += CountNodes<U>(child);
      }
    }
    return count;
  }

  template <typename U>
  void Initialize(const typename TupleTree<U>::Node& root) {
    size_t num_nodes = CountNodes<U>(root);
    entries_.emplace(num_nodes);

    size_t next_node_id = 0;
    size_t next_children_start_index = 1;
    BuildTable<U>(root, absl::MakeSpan(*entries_), 0, next_node_id,
                  next_children_start_index);
  }

  template <typename U>
  void BuildTable(const typename TupleTree<U>::Node& node,
                  absl::Span<Entry> entries, size_t current_entry_idx,
                  size_t& next_node_id, size_t& next_children_start_index) {
    Entry& entry = entries[current_entry_idx];
    entry.node_id = next_node_id++;

    if (node.IsLeaf()) {
      entry.children_start_id = -1;
      entry.num_children = 0;
      return;
    }

    // !node.IsLeaf(), so it's a tuple node (possibly empty).
    const std::vector<typename TupleTree<U>::Node>& children = node.children();
    entry.num_children = children.size();
    entry.children_start_id = next_children_start_index;

    size_t my_children_start = next_children_start_index;
    next_children_start_index += entry.num_children;

    for (size_t i = 0; i < children.size(); ++i) {
      BuildTable<U>(children[i], entries, my_children_start + i, next_node_id,
                    next_children_start_index);
    }
  }

  std::optional<absl::InlinedVector<Entry, 1>> entries_;
};

}  // namespace internal

// A TupleTree<T> is a tree data structure where each node, whether an
// internal node (tuple) or a leaf node, holds a value of type T. The structure
// is defined by the nesting of tuples.
//
// Key Characteristics:
// - Each node in the tree has an associated value of type T.
// - Nodes can be either internal nodes (tuples with children) or leaf nodes
//   (no children).
// - The structure is independent of XLA Shapes.
// - Internal nodes' values are distinct from their children's values.
//
// Non-obvious Behaviors:
// - Constructor from Span of Pairs: When constructing from a span of
//   {ShapeIndex, T} pairs, only the nodes at the specified indices are
//   initialized with the given values. Any necessary ancestor tuple nodes are
//   implicitly created and their values are constructed from the provided
//   arguments.
// - `element()` and `mutable_element()`: These methods can access the value
//   of *any* node in the tree, not just leaves, using its ShapeIndex.
// - `Map` and `MapWithStatus`: These functions apply the given function to the
//   values of *all* nodes in the tree, including internal tuple nodes.
// - Iterators:
//     - `begin()`/`end()` (and `nodes()`): Iterate over all nodes in the tree
//       in pre-order.
//     - `leaf_begin()`/`leaf_end()` (and `leaves()`): Iterate only over nodes
//       for which `IsLeaf()` is true (i.e., nodes with no children).
template <typename T>
class TupleTree {
 public:
  // Represents a node in the tree. It can be either a leaf value of type T
  // or a vector of subtrees (inner tuple).
  // This class is used for constructing TupleTree instances, defining the
  // structure and initial values.
  class Node {
   public:
    // Static factory for leaf nodes.
    static Node Leaf(const T& value) { return Node(value, std::nullopt); }
    static Node Leaf(T&& value) { return Node(std::move(value), std::nullopt); }

    // Static factories for tuple nodes.
    static Node Tuple() { return Node(T(), std::vector<Node>()); }
    static Node Tuple(const T& value) {
      return Node(value, std::vector<Node>());
    }
    static Node Tuple(T&& value) {
      return Node(std::move(value), std::vector<Node>());
    }
    static Node Tuple(absl::Span<const Node> children) {
      return Node(T(), std::vector<Node>(children.begin(), children.end()));
    }
    static Node Tuple(const T& value, absl::Span<const Node> children) {
      return Node(value, std::vector<Node>(children.begin(), children.end()));
    }
    static Node Tuple(T&& value, absl::Span<const Node> children) {
      return Node(std::move(value),
                  std::vector<Node>(children.begin(), children.end()));
    }

    // Default constructor creates an empty tuple.
    Node() : children_({}) {}

    // Copy constructor and assignment
    Node(const Node& other) = default;
    Node& operator=(const Node& other) = default;

    // Move constructor and assignment
    Node(Node&& other) noexcept = default;
    Node& operator=(Node&& other) noexcept = default;

    bool IsLeaf() const { return !children_.has_value(); }

    const T& value() const { return value_; }
    T* mutable_value() { return &value_; }

    const std::vector<Node>& children() const {
      CHECK(children_.has_value());
      return *children_;
    }
    std::vector<Node>* mutable_children() {
      CHECK(children_.has_value());
      return &*children_;
    }

    bool operator==(const Node& other) const {
      return value_ == other.value_ && children_ == other.children_;
    }
    bool operator!=(const Node& other) const { return !(*this == other); }

   private:
    // Primary constructors
    explicit Node(T value, std::optional<std::vector<Node>> children)
        : value_(std::move(value)), children_(std::move(children)) {}

    T value_;
    std::optional<std::vector<Node>> children_;
  };

  using NodePair = std::pair<ShapeIndex, T>;
  using NodePairs = absl::InlinedVector<NodePair, 1>;
  using IndexTable = internal::IndexTable;

  // Constructor for an empty tuple.
  TupleTree() { Initialize(Node::Tuple()); }

  // Constructor for a single leaf node.
  explicit TupleTree(const T& leaf_value) {
    Initialize(Node::Leaf(leaf_value));
  }
  explicit TupleTree(T&& leaf_value) {
    Initialize(Node::Leaf(std::move(leaf_value)));
  }

  // Constructor from an initializer list, creating a flat tuple of leaves.
  TupleTree(std::initializer_list<T> items) {
    std::vector<Node> children;
    children.reserve(items.size());
    for (const auto& item : items) {
      children.push_back(Node::Leaf(item));
    }
    Initialize(Node::Tuple(std::move(children)));
  }

  // Constructor from an initializer list of Nodes for nested structures.
  TupleTree(std::initializer_list<Node> items) {
    Initialize(Node::Tuple(items));
  }

  // Basic constructor taking the root node.
  explicit TupleTree(Node&& root) { Initialize(std::move(root)); }

  // Constructor from a list of shape indices and values.
  // U must be std::pair<ShapeIndex, T> or const std::pair<ShapeIndex, T>.
  // The tree structure is created based on the provided indices. Nodes at these
  // indices are initialized with the given values. Any implicitly created
  // parent nodes are constructed from the provided arguments.
  template <typename U, typename... Args>
  explicit TupleTree(absl::Span<U> node_pairs, Args&&... args) {
    static_assert(
        std::is_same_v<std::remove_const_t<U>, std::pair<ShapeIndex, T>>,
        "TupleTree constructor requires absl::Span of std::pair<ShapeIndex, T> "
        "or const std::pair<ShapeIndex, T>");

    Node root_node(Node::Tuple(T(args...)));
    for (auto& [index, value] : node_pairs) {
      Node* node = GetOrCreateNode(root_node, index,
                                   /*preserve_leaf_value=*/false, args...);
      // Forward the second element to enable move semantics if U is non-const.
      *node = Node::Leaf(std::forward<decltype(value)>(value));
    }
    Initialize(std::move(root_node));
  }

  // Constructor from a Shape and an initial value for all nodes constructed
  // from the given arguments.
  template <typename... Args>
  explicit TupleTree(const Shape& shape, Args&&... args) {
    index_table_ = IndexTable(shape);
    if (!shape.IsTuple()) {
      nodes_.emplace_back(ShapeIndex(), T(args...));
    } else {
      nodes_.reserve(ShapeUtil::SubshapeCount(shape));
      ShapeUtil::ForEachSubshape(shape,
                                 [&](const Shape&, const ShapeIndex& index) {
                                   nodes_.emplace_back(index, T(args...));
                                 });
    }
  }

  // Returns the data element at the given index. This works for any valid
  // index, whether it's an internal node or a leaf node.
  // CHECK-fails if the index is invalid.
  const T& element(ShapeIndexView index) const {
    absl::StatusOr<const internal::IndexTable::Entry*> entry_or =
        index_table_.GetEntry(index);
    CHECK_OK(entry_or.status());
    const internal::IndexTable::Entry* entry = entry_or.value();
    return nodes_[entry->node_id].second;
  }

  // Returns a pointer to the data element at the given index. This works for
  // any valid index, whether it's an internal node or a leaf node.
  // CHECK-fails if the index is invalid.
  T* mutable_element(ShapeIndexView index) {
    absl::StatusOr<const internal::IndexTable::Entry*> entry_or =
        index_table_.GetEntry(index);
    CHECK_OK(entry_or.status());
    const internal::IndexTable::Entry* entry = entry_or.value();
    return &nodes_[entry->node_id].second;
  }

  // Returns true if the node at the given index is a leaf node (has no
  // children).
  bool IsLeaf(ShapeIndexView index) const {
    absl::StatusOr<const internal::IndexTable::Entry*> entry_or =
        index_table_.GetEntry(index);
    if (!entry_or.ok()) {
      return false;
    }
    return entry_or.value()->children_start_id == -1;
  }

  // Checks if the structure of this TupleTree is compatible with the given
  // shape.
  bool IsStructurallyCompatible(const Shape& shape) const {
    internal::IndexTable shape_table(shape);
    auto shape_root_or = shape_table.GetEntry({});
    auto tree_root_or = index_table_.GetEntry({});
    if (!shape_root_or.ok() || !tree_root_or.ok()) {
      return false;
    }
    return internal::IndexTable::IsSubtreeCompatible(
               shape_table, shape_root_or.value(), index_table_,
               tree_root_or.value())
        .ok();
  }

  bool IsTuple() const { return nodes_.size() > 1; }

  absl::Status CopyCompatibleSubtreeFrom(const TupleTree<T>& other,
                                         const ShapeIndex& src_index,
                                         const ShapeIndex& dst_index) {
    TF_ASSIGN_OR_RETURN(const internal::IndexTable::Entry* src_entry,
                        other.index_table_.GetEntry(src_index));
    TF_ASSIGN_OR_RETURN(const internal::IndexTable::Entry* dst_entry,
                        this->index_table_.GetEntry(dst_index));

    TF_RETURN_IF_ERROR(internal::IndexTable::IsSubtreeCompatible(
        other.index_table_, src_entry, this->index_table_, dst_entry));

    size_t num_subtree_nodes =
        internal::IndexTable::CountSubtreeNodes(other.index_table_, src_entry);

    for (size_t i = 0; i < num_subtree_nodes; ++i) {
      const auto& src_pair = other.nodes_[src_entry->node_id + i];
      auto& dst_pair = this->nodes_[dst_entry->node_id + i];
      dst_pair.second = T(src_pair.second);
    }

    return absl::OkStatus();
  }

  void CopySubtreeFrom(const TupleTree<T>& other, const ShapeIndex& src_index,
                       const ShapeIndex& dst_index) {
    absl::StatusOr<Node> src_node_or = other.ToNode(src_index);
    CHECK_OK(src_node_or.status());
    Node src_node = std::move(src_node_or).value();

    if (dst_index.empty()) {
      Initialize(std::move(src_node));
      return;
    }

    absl::StatusOr<Node> root_node_or = ToNode();
    CHECK_OK(root_node_or.status());
    Node root_node = std::move(root_node_or).value();

    Node* target_node = GetOrCreateNode(root_node, dst_index,
                                        /*preserve_leaf_value=*/true);
    *target_node = std::move(src_node);

    Initialize(std::move(root_node));
  }

  absl::StatusOr<TupleTree<T>> Subtree(const ShapeIndex& index) const {
    TF_ASSIGN_OR_RETURN(const internal::IndexTable::Entry* root_entry,
                        index_table_.GetEntry(index));
    size_t root_node_id = root_entry->node_id;

    TF_ASSIGN_OR_RETURN(
        internal::IndexTable subtree_index_table,
        internal::IndexTable::CreateFromSubtree(index_table_, index));

    if (!subtree_index_table.entries().has_value()) {
      // This case should ideally not be reached if GetEntry succeeded.
      return absl::InternalError("Subtree index table creation failed");
    }

    size_t num_subtree_nodes = subtree_index_table.entries()->size();
    typename TupleTree<T>::NodePairs subtree_nodes;
    subtree_nodes.reserve(num_subtree_nodes);

    for (size_t i = 0; i < num_subtree_nodes; ++i) {
      const auto& original_pair = nodes_[root_node_id + i];
      ShapeIndex new_index(original_pair.first.begin() + index.size(),
                           original_pair.first.end());
      subtree_nodes.emplace_back(std::move(new_index), original_pair.second);
    }

    return TupleTree<T>(std::move(subtree_index_table),
                        std::move(subtree_nodes));
  }

  using iterator = typename NodePairs::iterator;
  using const_iterator = typename NodePairs::const_iterator;
  using reverse_iterator = typename NodePairs::reverse_iterator;
  using const_reverse_iterator = typename NodePairs::const_reverse_iterator;

  template <typename Iterator, typename ValueType>
  class LeafIterator;

  using leaf_iterator = LeafIterator<iterator, NodePair>;
  using const_leaf_iterator = LeafIterator<const_iterator, const NodePair>;
  using reverse_leaf_iterator = std::reverse_iterator<leaf_iterator>;
  using const_reverse_leaf_iterator =
      std::reverse_iterator<const_leaf_iterator>;

  iterator begin() { return nodes_.begin(); }
  iterator end() { return nodes_.end(); }
  const_iterator begin() const { return nodes_.begin(); }
  const_iterator end() const { return nodes_.end(); }

  reverse_iterator rbegin() { return nodes_.rbegin(); }
  reverse_iterator rend() { return nodes_.rend(); }
  const_reverse_iterator rbegin() const { return nodes_.rbegin(); }
  const_reverse_iterator rend() const { return nodes_.rend(); }

  // leaf_begin()/leaf_end() iterates over all nodes for which IsLeaf() is true
  // (i.e., nodes with no children).
  leaf_iterator leaf_begin() { return leaf_iterator(*this, nodes_.begin()); }
  leaf_iterator leaf_end() { return leaf_iterator(*this, nodes_.end()); }
  const_leaf_iterator leaf_begin() const {
    return const_leaf_iterator(*this, nodes_.begin());
  }
  const_leaf_iterator leaf_end() const {
    return const_leaf_iterator(*this, nodes_.end());
  }
  // range-based iterator for leaf_begin()/leaf_end().
  tsl::gtl::iterator_range<leaf_iterator> leaves() {
    return tsl::gtl::make_range(leaf_begin(), leaf_end());
  }
  tsl::gtl::iterator_range<const_leaf_iterator> leaves() const {
    return tsl::gtl::make_range(leaf_begin(), leaf_end());
  }

  reverse_leaf_iterator leaf_rbegin() {
    return reverse_leaf_iterator(leaf_end());
  }
  reverse_leaf_iterator leaf_rend() {
    return reverse_leaf_iterator(leaf_begin());
  }
  const_reverse_leaf_iterator leaf_rbegin() const {
    return const_reverse_leaf_iterator(leaf_end());
  }
  const_reverse_leaf_iterator leaf_rend() const {
    return const_reverse_leaf_iterator(leaf_begin());
  }

  size_t num_leaves() const { return std::distance(leaf_begin(), leaf_end()); }

  // Returns an iterator pointing to the node at the given ShapeIndex.
  // Returns end() if the index is not found.
  iterator find(ShapeIndexView index) {
    absl::StatusOr<const internal::IndexTable::Entry*> entry_or =
        index_table_.GetEntry(index);
    if (!entry_or.ok()) {
      return nodes_.end();
    }
    return nodes_.begin() + (*entry_or)->node_id;
  }
  const_iterator find(ShapeIndexView index) const {
    absl::StatusOr<const internal::IndexTable::Entry*> entry_or =
        index_table_.GetEntry(index);
    if (!entry_or.ok()) {
      return nodes_.end();
    }
    return nodes_.begin() + (*entry_or)->node_id;
  }

  // Traversal functions for all elements. These iterate over ALL nodes.
  void ForEachElement(
      absl::FunctionRef<void(const ShapeIndex&, const T&)> func) const {
    for (const NodePair& node : nodes_) {
      func(node.first, node.second);
    }
  }

  void ForEachMutableElement(
      absl::FunctionRef<void(const ShapeIndex&, T*)> func) {
    for (NodePair& node : nodes_) {
      func(node.first, &node.second);
    }
  }

  absl::Status ForEachElementWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, const T&)> func) const {
    for (const NodePair& node : nodes_) {
      TF_RETURN_IF_ERROR(func(node.first, node.second));
    }
    return absl::OkStatus();
  }

  absl::Status ForEachMutableElementWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, T*)> func) {
    for (NodePair& node : nodes_) {
      TF_RETURN_IF_ERROR(func(node.first, &node.second));
    }
    return absl::OkStatus();
  }

  bool operator==(const TupleTree<T>& other) const {
    if (nodes_.size() != other.nodes_.size()) {
      return false;
    }
    // The order in nodes_ is deterministic (pre-order).
    return nodes_ == other.nodes_;
  }
  bool operator!=(const TupleTree<T>& other) const { return !(*this == other); }

  // Returns a range to iterate over all nodes in pre-order.
  tsl::gtl::iterator_range<iterator> nodes() {
    return tsl::gtl::make_range(begin(), end());
  }
  // Returns a const range to iterate over all nodes in pre-order.
  tsl::gtl::iterator_range<const_iterator> nodes() const {
    return tsl::gtl::make_range(begin(), end());
  }

  // Maps each node's value to generate a new tree with the same structure.
  // The function `func` is applied to the value of *every* node.
  template <typename U>
  TupleTree<U> Map(absl::FunctionRef<U(const T&)> func) const {
    typename TupleTree<U>::NodePairs result_nodes;
    result_nodes.reserve(nodes_.size());
    for (const NodePair& node : nodes_) {
      result_nodes.emplace_back(node.first, func(node.second));
    }

    return TupleTree<U>(index_table_, std::move(result_nodes));
  }

  // Maps each node's value to generate a new tree with the same structure,
  // allowing the mapping function to return a StatusOr.
  // The function `func` is applied to the value of *every* node.
  template <typename U>
  absl::StatusOr<TupleTree<U>> MapWithStatus(
      absl::FunctionRef<absl::StatusOr<U>(const T&)> func) const {
    typename TupleTree<U>::NodePairs result_nodes;
    result_nodes.reserve(nodes_.size());
    for (const NodePair& node : nodes_) {
      TF_ASSIGN_OR_RETURN(U result, func(node.second));
      result_nodes.emplace_back(node.first, std::move(result));
    }

    return TupleTree<U>(index_table_, std::move(result_nodes));
  }

  absl::StatusOr<Node> ToNode(ShapeIndexView index_view = {}) const {
    if (!index_table_.entries().has_value()) {
      return Node::Tuple(T());
    }
    ShapeIndex index(index_view.begin(), index_view.end());
    return ToNodeImpl(index);
  }

 private:
  template <typename U>
  friend class TupleTree;

  // Private constructor for internal use (e.g., Map).
  TupleTree(const IndexTable& index_table, NodePairs&& nodes)
      : nodes_(std::move(nodes)), index_table_(index_table) {}

  // Private constructor for SubTree.
  TupleTree(internal::IndexTable&& index_table, NodePairs&& nodes)
      : nodes_(std::move(nodes)), index_table_(std::move(index_table)) {}

  void Initialize(Node root) {
    // First, build the IndexTable from the structure.
    index_table_ = internal::IndexTable::template Create<T>(root);

    // Then, build the nodes_ vector, moving values from root.
    nodes_.clear();
    if (index_table_.entries().has_value()) {
      nodes_.reserve(index_table_.entries()->size());
    }
    ShapeIndex current_index;
    BuildNodesVector(std::move(root), current_index);
  }

  void BuildNodesVector(Node node, ShapeIndex& current_index) {
    nodes_.emplace_back(current_index, std::move(*node.mutable_value()));
    if (!node.IsLeaf()) {
      std::vector<Node>* children = node.mutable_children();
      for (size_t i = 0; i < children->size(); ++i) {
        current_index.push_back(i);
        BuildNodesVector(std::move((*children)[i]), current_index);
        current_index.pop_back();
      }
    }
  }

  template <typename... Args>
  Node* GetOrCreateNode(Node& root, const ShapeIndex& index,
                        bool preserve_leaf_value, Args&&... args) {
    Node* node = &root;
    if (index.empty()) {
      return node;
    }
    for (int i = 0; i < index.size(); ++i) {
      int64_t idx = index[i];
      CHECK_GE(idx, 0);

      if (node->IsLeaf()) {
        // Transition from leaf to tuple.
        if (preserve_leaf_value) {
          T original_value = std::move(*node->mutable_value());
          *node = Node::Tuple(T(args...));
          // The original leaf value is placed at index 0.
          node->mutable_children()->push_back(
              Node::Leaf(std::move(original_value)));
        } else {
          *node = Node::Tuple(T(args...));
        }
      }

      std::vector<Node>* children = node->mutable_children();
      while (idx >= children->size()) {
        children->push_back(Node::Tuple(T(args...)));
      }
      node = &children->at(idx);
    }
    return node;
  }

  absl::StatusOr<Node> ToNodeImpl(const ShapeIndex& index) const {
    TF_ASSIGN_OR_RETURN(const internal::IndexTable::Entry* entry,
                        index_table_.GetEntry(index));

    const T& value = nodes_[entry->node_id].second;

    if (entry->children_start_id == -1) {  // Is a leaf node
      return Node::Leaf(value);
    }

    // Is an internal tuple node
    std::vector<Node> children;
    children.reserve(entry->num_children);
    ShapeIndex child_index = index;
    child_index.push_back(0);
    for (size_t i = 0; i < entry->num_children; ++i) {
      child_index.back() = i;
      TF_ASSIGN_OR_RETURN(Node child_node, ToNodeImpl(child_index));
      children.push_back(std::move(child_node));
    }
    return Node::Tuple(value, std::move(children));
  }

  // Leaves sorted in pre-order.
  NodePairs nodes_;
  IndexTable index_table_;
};

// Internal iterator that performs a pre-order walk of the leaves. This is cheap
// to copy. The iterator value_type is equivalent to a std::pair<ShapeIndex,T>&,
// similar to std::map.
template <typename T>
template <typename Iterator, typename ValueType>
class TupleTree<T>::LeafIterator {
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = ValueType;
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  LeafIterator(const TupleTree& tree, Iterator it) : tree_(tree), it_(it) {
    while ((it_ != tree_.nodes_.end()) && !tree_.IsLeaf(it_->first)) {
      ++it_;
    }
  }

  LeafIterator& operator++() {
    do {
      ++it_;
    } while ((it_ != tree_.nodes_.end()) && !tree_.IsLeaf(it_->first));
    return *this;
  }

  LeafIterator operator++(int) {
    auto prev = *this;
    ++(*this);
    return prev;
  }

  LeafIterator& operator--() {
    do {
      --it_;
    } while ((it_ != tree_.nodes_.begin()) && !tree_.IsLeaf(it_->first));
    return *this;
  }

  LeafIterator operator--(int) {
    auto prev = *this;
    --(*this);
    return prev;
  }

  bool operator==(const LeafIterator& other) const { return it_ == other.it_; }
  bool operator!=(const LeafIterator& other) const { return !(*this == other); }
  ValueType& operator*() const { return *it_; }
  ValueType* operator->() const { return &*it_; }

 private:
  const TupleTree<T>& tree_;
  Iterator it_;
};

}  // namespace xla

#endif  // XLA_TUPLE_TREE_H_
