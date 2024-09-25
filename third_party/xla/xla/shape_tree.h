/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SHAPE_TREE_H_
#define XLA_SHAPE_TREE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {

namespace internal {

class IndexTable {
 public:
  // Use indices, rather than pointers, so index table can be copied between
  // ShapeTrees.
  struct Entry {
    // Index of the node in the nodes vector.
    size_t node_id;
    // Index of the first child of this node in the index table (-1 for leaves).
    std::make_signed_t<size_t> children_start_id = -1;
  };

  IndexTable() = default;
  explicit IndexTable(const Shape& shape);

  bool empty() const { return entries_.empty(); }

  const Entry& operator[](ShapeIndexView index) const;

 private:
  void CreateEntry(Entry& entry, const Shape& shape, size_t& next_node_id);

  absl::InlinedVector<Entry, 1> entries_;
};

}  // namespace internal

// A ShapeTree<T> is a recursive data structure which mirrors the structure of a
// XLA shape and holds a value of type T for each subshape (i.e. tuple or array)
// in the shape. For array shapes, a ShapeTree trivially holds a single value of
// type T.
//
// For tuple shapes which can be an arbitrary tree with arrays at the leaves, a
// ShapeTree is an identically structured tree with data elements of type T at
// every node. I.e. the root is a tuple by definition, all interior nodes are
// also tuples, and all leaves are arrays.
//
// Like the Shape data structure, this is a tree and tuple elements cannot be
// duplicated. That is, every distinct ShapeIndex in the Shape has a unique T
// object.
//
// Normally a ShapeTree owns its Shape, but for efficiency reasons, sometimes
// it's helpful not to copy a Shape just to make a ShapeTree.  In these cases,
// you can pass a Shape* instead of a Shape to the ShapeTree constructor.  It's
// then up to you to ensure that the pointed-to Shape isn't freed, moved or
// modified before its ShapeTree goes away.
template <typename T>
class ShapeTree {
  template <typename U>
  friend class ShapeTree;

 public:
  // TODO(cjfj): Don't store ShapeIndex with data. Generate it or cache it?
  using Node = std::pair<ShapeIndex, T>;
  using Nodes = absl::InlinedVector<Node, 1>;
  using IndexTable = internal::IndexTable;

  template <typename Iterator, typename ValueType>
  class LeafIterator;

  // Default constructor creates a tree with a nil shape (i.e. an empty tuple).
  ShapeTree() : ShapeTree(ShapeUtil::MakeNil()) {}

  // Create ShapeTree with the given shape, and default-constructed T values for
  // all nodes.
  //
  // The version that takes a pointer may be cheaper because it doesn't require
  // any Shape copies, but then it's up to you to ensure that the pointer stays
  // alive longer than this ShapeTree.
  explicit ShapeTree(Shape shape)
      : ShapeTree(std::make_shared<Shape>(std::move(shape))) {}

  explicit ShapeTree(const Shape* shape)
      : ShapeTree(shape, CreateNodes(*shape)) {}

  // Create ShapeTree with the given shape, and init_value for all nodes.
  ShapeTree(Shape shape, const T& init_value)
      : ShapeTree(std::make_shared<Shape>(std::move(shape)), init_value) {}

  ShapeTree(const Shape* shape, const T& init_value)
      : ShapeTree(shape, CreateNodes(*shape, init_value)) {}

  // Returns the data element associated with the array in the shape at the
  // given index (see ShapeUtil::GetSubshape for how indexes are defined).
  const T& element(ShapeIndexView index) const { return find(index)->second; }
  T* mutable_element(ShapeIndexView index) { return &find(index)->second; }

  // Return the shape represented with this ShapeTree.
  const Shape& shape() const { return *shape_; }

  // A ShapeTree object can own the underlying Shape pointer (via the
  // shape_storage_ member), or can point to a Shape object owned by the caller.
  // This API replaces the underlying Shape object to the one supplied by the
  // caller, whom must ensure the object remain valid for the whole lifetime of
  // this ShapeTree object, and also that the Shape is consistent with it.
  void replace_shape_ptr(const Shape& shape) {
    if (shape_storage_ != nullptr) {
      DCHECK_EQ(shape, *shape_storage_);
      shape_storage_ = nullptr;
    }
    shape_ = &shape;
  }

  // Returns true if the node at the given index is a leaf node (an array
  // shape).
  bool IsLeaf(ShapeIndexView index) const {
    return index_table_[index].children_start_id == -1;
  }

  using iterator = typename Nodes::iterator;
  using const_iterator = typename Nodes::const_iterator;
  using reverse_iterator = typename Nodes::reverse_iterator;
  using const_reverse_iterator = typename Nodes::const_reverse_iterator;

  using leaf_iterator = LeafIterator<iterator, Node>;
  using const_leaf_iterator = LeafIterator<const_iterator, const Node>;
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

  // leaf_begin()/leaf_end() iterates over all leaf nodes (nodes with no
  // children).
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

  // Returns an iterator pointing to the given ShapeIndex.
  // REQUIRES: index must exist in the ShapeTree.
  iterator find(ShapeIndexView index) {
    return nodes_.begin() + index_table_[index].node_id;
  }
  const_iterator find(ShapeIndexView index) const {
    return nodes_.begin() + index_table_[index].node_id;
  }

  // Returns the number of leaf nodes in the tree.
  int64_t leaf_count() const { return std::distance(leaf_begin(), leaf_end()); }

  // TODO(cjfj): Remove the `ForEach...` methods. They are redundant.
  // Recursively traverses the shape and calls the given function at each
  // element.
  void ForEachElement(
      absl::FunctionRef<void(const ShapeIndex&, const T&)> func) const {
    for (const Node& node : nodes_) {
      func(node.first, node.second);
    }
  }

  void ForEachMutableElement(
      absl::FunctionRef<void(const ShapeIndex&, T*)> func) {
    for (Node& node : nodes_) {
      func(node.first, &node.second);
    }
  }

  // Like ForEach(Mutable)Element, but the callable returns a absl::Status
  // instead of void.  The first non-OK return value is returned by the ForEach*
  // function.
  absl::Status ForEachElementWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, const T&)> func) const {
    for (const Node& node : nodes_) {
      TF_RETURN_IF_ERROR(func(node.first, node.second));
    }
    return absl::OkStatus();
  }

  absl::Status ForEachMutableElementWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, T*)> func) {
    for (Node& node : nodes_) {
      TF_RETURN_IF_ERROR(func(node.first, &node.second));
    }
    return absl::OkStatus();
  }

  // Like the above, but traverses in post-order.  Note children are visited in
  // right-to-left order.
  void ForEachElementPostOrder(
      absl::FunctionRef<void(const ShapeIndex&, const T&)> func) const {
    for (auto node = nodes_.rbegin(); node != nodes_.rend(); ++node) {
      func(node->first, node->second);
    }
  }

  void ForEachMutableElementPostOrder(
      absl::FunctionRef<void(const ShapeIndex&, T*)> func) {
    for (auto node = nodes_.rbegin(); node != nodes_.rend(); ++node) {
      func(node->first, &node->second);
    }
  }

  absl::Status ForEachElementPostOrderWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, const T&)> func) const {
    for (auto node = nodes_.rbegin(); node != nodes_.rend(); ++node) {
      TF_RETURN_IF_ERROR(func(node->first, node->second));
    }
    return absl::OkStatus();
  }

  absl::Status ForEachMutableElementPostOrderWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, T*)> func) {
    for (auto node = nodes_.rbegin(); node != nodes_.rend(); ++node) {
      TF_RETURN_IF_ERROR(func(node->first, &node->second));
    }
    return absl::OkStatus();
  }

  // Maps each element to generate a new tree with the same shape.
  template <typename U>
  ShapeTree<U> Map(absl::FunctionRef<U(const T&)> func) {
    typename ShapeTree<U>::Nodes result_nodes;
    result_nodes.reserve(nodes_.size());
    for (const Node& node : nodes_) {
      result_nodes.push_back({node.first, func(node.second)});
    }

    ShapeTree<U> result(shape_, std::move(result_nodes));
    result.index_table_ = index_table_;
    result.shape_storage_ = shape_storage_;
    return result;
  }

  template <typename U>
  absl::StatusOr<ShapeTree<U>> MapWithStatus(
      absl::FunctionRef<absl::StatusOr<U>(const T&)> func) {
    typename ShapeTree<U>::Nodes result_nodes;
    result_nodes.reserve(nodes_.size());
    for (const Node& node : nodes_) {
      TF_ASSIGN_OR_RETURN(U result, func(node.second));
      result_nodes.push_back({node.first, std::move(result)});
    }

    ShapeTree<U> result(shape_, std::move(result_nodes));
    result.index_table_ = index_table_;
    result.shape_storage_ = shape_storage_;
    return result;
  }

  // Copy the subtree of values from 'other' rooted at ShapeIndex 'src_index'
  // into the subtree of value in this ShapeTree rooted at 'dst_index'.
  //
  // Precondition: The subshape of other.shape() at index src_index must be
  // compatible with the subshape of shape() at index dst_index.
  void CopySubtreeFrom(const ShapeTree<T>& other, const ShapeIndex& src_index,
                       const ShapeIndex& dst_index) {
    const Shape& src_shape = ShapeUtil::GetSubshape(other.shape(), src_index);
    const Shape& dst_shape = ShapeUtil::GetSubshape(shape(), dst_index);
    CHECK(ShapeUtil::Compatible(src_shape, dst_shape))
        << src_shape << ", " << dst_shape;

    // Replace the prefix `src_index` with `dst_index`.
    auto replace_shape_index_prefix = [&](const ShapeIndex& index) {
      auto without_prefix = ShapeIndexView(index).subspan(src_index.size());
      ShapeIndex result;
      result.reserve(dst_index.size() + without_prefix.size());
      result.insert(result.end(), dst_index.begin(), dst_index.end());
      result.insert(result.end(), without_prefix.begin(), without_prefix.end());
      return result;
    };

    auto first = other.find(src_index);
    auto last = first + ShapeUtil::SubshapeCount(src_shape);

    std::transform(first, last, find(dst_index), [&](const Node& node) -> Node {
      return {replace_shape_index_prefix(node.first), node.second};
    });
  }

  absl::StatusOr<ShapeTree<T>> SubShapeTree(const ShapeIndex& index) const {
    TF_ASSIGN_OR_RETURN(const Shape* sub_shape,
                        ShapeUtil::TryGetSubshape(shape(), index));
    size_t count = ShapeUtil::SubshapeCount(*sub_shape);
    Nodes sub_tree_nodes;
    sub_tree_nodes.reserve(count);
    for (auto it = find(index), end = it + count; it != end; ++it) {
      // For each shape index, remove the prefix `index`.
      auto without_prefix = ShapeIndexView(it->first).subspan(index.size());
      sub_tree_nodes.push_back(Node{without_prefix, it->second});
    }
    return ShapeTree(sub_shape, std::move(sub_tree_nodes));
  }

  bool operator==(const ShapeTree<T>& other) const {
    return nodes_ == other.nodes_;
  }
  bool operator!=(const ShapeTree<T>& other) const { return !(*this == other); }

 private:
  explicit ShapeTree(std::shared_ptr<Shape> shape) : ShapeTree(shape.get()) {
    shape_storage_.swap(shape);
  }

  ShapeTree(std::shared_ptr<Shape> shape, const T& init_value)
      : ShapeTree(shape.get(), init_value) {
    shape_storage_.swap(shape);
  }

  ShapeTree(const Shape* shape, Nodes nodes)
      : nodes_(std::move(nodes)), index_table_(*shape), shape_(shape) {
    DCHECK_EQ(nodes_.size(), ShapeUtil::SubshapeCount(*shape));
  }

  template <typename... Ts>
  static Nodes CreateNodes(const Shape& shape, Ts&&... args) {
    Nodes nodes;
    ShapeUtil::ForEachSubshape(
        shape, [&](const Shape&, const ShapeIndex& index) {
          nodes.push_back({index, T(std::forward<Ts>(args)...)});
        });
    return nodes;
  }

  // The nodes in this shape tree.
  Nodes nodes_;

  // Index table for node lookups. Each entry contains the index of the first
  // child of the node at that index, or -1 for leaf nodes. Evaluated lazily.
  IndexTable index_table_;

  // If we own our Shape, this field contains it, and shape_ is a pointer into
  // here.  Otherwise if we don't own our shape, this is nullptr.
  std::shared_ptr<Shape> shape_storage_;

  // The XLA shape mirrored in this ShapeTree.  This is either
  // shape_storage_.get() or the Shape pointer passed to our constructor.
  const Shape* shape_;
};

// Internal iterator that performs a pre-order walk of the leaves. This is cheap
// to copy. The iterator value_type is equivalent to a std::pair<ShapeIndex,T>&,
// similar to std::map.
template <typename T>
template <typename Iterator, typename ValueType>
class ShapeTree<T>::LeafIterator {
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = ValueType;
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  LeafIterator(const ShapeTree& tree, Iterator it) : tree_(tree), it_(it) {
    while ((it_ != tree_.nodes_.end()) && !IsLeaf()) ++it_;
  }

  LeafIterator& operator++() {
    do {
      ++it_;
    } while ((it_ != tree_.nodes_.end()) && !IsLeaf());
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
    } while ((it_ != tree_.nodes_.begin()) && !IsLeaf());
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
  bool IsLeaf() const { return tree_.IsLeaf(it_->first); }

  const ShapeTree<T>& tree_;
  Iterator it_;
};

}  // namespace xla

#endif  // XLA_SHAPE_TREE_H_
