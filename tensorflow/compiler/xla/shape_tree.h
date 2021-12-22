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

#ifndef TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
#define TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_

#include <functional>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace internal {

// Internal representation of each node in a ShapeTree.
template <typename T>
struct ShapeTreeNode {
  // Data corresponding to this node.
  std::pair<ShapeIndex, T> data;

  bool is_leaf = true;

  explicit ShapeTreeNode(ShapeIndex index)
      : ShapeTreeNode(std::move(index), T()) {}
  ShapeTreeNode(ShapeIndex index, T data)
      : data(std::move(index), std::move(data)) {}
};

// Internal representation of an index table entry.
struct IndexTableEntry {
  // Index of the node in the ShapeTreeNode vector.
  uint32_t index;
  // Index of the first child in a IndexTableEntry vector. In the index
  // table all children entries for a given node will be placed next to each
  // other. This allows us to use a single field to index them.
  uint32_t children_start;
#ifndef NDEBUG
  // Number of children, used for bounds checking.
  uint32_t children_count;
#endif
};

}  // namespace internal

template <typename ContainerType, typename IteratorType, typename ValueType>
class ShapeTreeIterator;
template <typename ContainerType, typename IteratorType, typename ValueType>
class ShapeTreeLeafIterator;

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
// you can pass a Shape* instead of a Shape& to the ShapeTree constructor.  It's
// then up to you to ensure that the pointed-to Shape doesn't die or mutate
// before its ShapeTree goes away.
template <typename T>
class ShapeTree {
 public:
  using Node = internal::ShapeTreeNode<T>;
  using Index = internal::IndexTableEntry;

  // Default constructor creates a tree with a nil shape (i.e. an empty tuple).
  ShapeTree() : ShapeTree(ShapeUtil::MakeNil()) {}

  // Create ShapeTree with the given shape, and default-constructed T values for
  // all nodes.
  //
  // The version that takes a pointer may be cheaper because it doesn't require
  // any Shape copies, but then it's up to you to ensure that the pointer stays
  // alive longer than this ShapeTree.
  explicit ShapeTree(Shape shape);
  explicit ShapeTree(const Shape* shape);
  explicit ShapeTree(const std::shared_ptr<Shape>& shape);

  // Create ShapeTree with the given shape, and init_value for all nodes.
  ShapeTree(Shape shape, const T& init_value);
  ShapeTree(const Shape* shape, const T& init_value);
  ShapeTree(const std::shared_ptr<Shape>& shape, const T& init_value);

  // Returns the data element associated with the array in the shape at the
  // given index (see ShapeUtil::GetSubshape for how indexes are defined).
  const T& element(ShapeIndexView index) const;
  T* mutable_element(ShapeIndexView index);

  // Return the shape represented with this ShapeTree.
  const Shape& shape() const { return *shape_; }

  // A ShapeTree object can own the underlying Shape pointer (via the
  // shape_storage_ member), or can point to a Shape object owned by the caller.
  // This API replaces the underlying Shape object to the one supplied by the
  // caller, whom must ensure the object remain valid for the whole lifetime of
  // this ShapeTree object, and also that the Shape is consistent with it.
  void replace_shape_ptr(const Shape* shape) {
    if (shape_storage_ != nullptr) {
      DCHECK_EQ(*shape, *shape_storage_);
      shape_storage_ = nullptr;
    }
    shape_ = shape;
  }

  // Returns true if the node at the given index is a leaf node (an array
  // shape).
  bool IsLeaf(ShapeIndexView index) const { return Lookup(index)->is_leaf; }

  ShapeTree(const ShapeTree&) = default;
  ShapeTree& operator=(const ShapeTree&) = default;
  ShapeTree(ShapeTree&&) = default;
  ShapeTree& operator=(ShapeTree&& other) = default;

  // iterator implements a bidirectional_iterator with
  //  value_type = std::pair<ShapeIndex, T>.
  //
  // The iteration order is guaranteed to be a pre-order walk of the ShapeTree.
  using iterator =
      ShapeTreeIterator<std::vector<Node>, typename std::vector<Node>::iterator,
                        std::pair<ShapeIndex, T>>;
  using const_iterator =
      ShapeTreeIterator<const std::vector<Node>,
                        typename std::vector<Node>::const_iterator,
                        const std::pair<ShapeIndex, T>>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using leaf_iterator =
      ShapeTreeLeafIterator<std::vector<Node>,
                            typename std::vector<Node>::iterator,
                            std::pair<ShapeIndex, T>>;
  using const_leaf_iterator =
      ShapeTreeLeafIterator<const std::vector<Node>,
                            typename std::vector<Node>::const_iterator,
                            const std::pair<ShapeIndex, T>>;
  using reverse_leaf_iterator = std::reverse_iterator<leaf_iterator>;
  using const_reverse_leaf_iterator =
      std::reverse_iterator<const_leaf_iterator>;

  // begin/end for iterating over all nodes.
  iterator begin() { return iterator(&nodes_, nodes_.begin()); }
  iterator end() { return iterator(&nodes_, nodes_.end()); }
  const_iterator begin() const {
    return const_iterator(&nodes_, nodes_.begin());
  }
  const_iterator end() const { return const_iterator(&nodes_, nodes_.end()); }

  // rbegin/rend for iterating over all nodes in reverse.
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  // leaf_begin()/leaf_end() iterates over all leaf nodes (nodes with no
  // children).
  leaf_iterator leaf_begin() { return leaf_iterator(&nodes_, nodes_.begin()); }
  leaf_iterator leaf_end() { return leaf_iterator(&nodes_, nodes_.end()); }
  const_leaf_iterator leaf_begin() const {
    return const_leaf_iterator(&nodes_, nodes_.begin());
  }
  const_leaf_iterator leaf_end() const {
    return const_leaf_iterator(&nodes_, nodes_.end());
  }
  // range-based iterator for leaf_begin()/leaf_end().
  tensorflow::gtl::iterator_range<leaf_iterator> leaves() {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }
  tensorflow::gtl::iterator_range<const_leaf_iterator> leaves() const {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
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
    Node* element = Lookup(index);
    auto element_iter = nodes_.begin() + (element - &nodes_[0]);
    return iterator(&nodes_, element_iter);
  }
  const_iterator find(ShapeIndexView index) const {
    const Node* element = Lookup(index);
    auto element_iter = nodes_.cbegin() + (element - &nodes_[0]);
    return const_iterator(&nodes_, element_iter);
  }

  // Returns the number of leaf nodes in the tree.
  int64_t leaf_count() const { return std::distance(leaf_begin(), leaf_end()); }

  // Recursively traverses the shape and calls the given function at each
  // element. The function has the following arguments:
  //
  //   Fn :    A callable of type void(const ShapeIndex& index, const T& data)
  //           (or compatible).
  //   index : the index of the element in the shape. See ShapeUtil::GetSubshape
  //           for definition of index.
  //   data : The data value at this element.
  template <typename Fn>
  void ForEachElement(const Fn& func) const;

  // Like ForEachElement, but the callable has type
  //
  //   void (const ShapeIndex& index, T* data).
  //
  template <typename Fn>
  void ForEachMutableElement(const Fn& func);

  // Like ForEach(Mutable)Element, but the callable returns a Status instead of
  // void.  The first non-OK return value is returned by the ForEach* function.
  template <typename Fn>
  Status ForEachElementWithStatus(const Fn& func) const;
  template <typename Fn>
  Status ForEachMutableElementWithStatus(const Fn& func);

  // Maps each element to generate a new tree with the same shape.
  template <typename U>
  ShapeTree<U> Map(const std::function<U(const T&)>& func) {
    ShapeTree<U> result(shape_storage_);
    ForEachElement([&](const ShapeIndex& index, const T& t) {
      *result.mutable_element(index) = func(t);
    });
    return result;
  }

  template <typename U>
  ShapeTree<U> Map(const std::function<U(T*)>& func) {
    ShapeTree<U> result(shape_storage_);
    ForEachMutableElement([&](const ShapeIndex& index, T* t) {
      *result.mutable_element(index) = func(t);
    });
    return result;
  }

  // Copy the subtree of values from 'other' rooted at ShapeIndex
  // 'source_base_index' into the subtree of value in this ShapeTree rooted at
  // 'target_base_index'.
  //
  // Precondition: The subshape of other.shape() at index source_base_index must
  // be compatible with the subshape of shape() at index target_base_index.
  void CopySubtreeFrom(const ShapeTree<T>& other,
                       const ShapeIndex& source_base_index,
                       const ShapeIndex& target_base_index);

  StatusOr<ShapeTree<T>> SubShapeTree(const ShapeIndex& index) const;

  bool operator==(const ShapeTree<T>& other) const;
  bool operator!=(const ShapeTree<T>& other) const { return !(*this == other); }

 private:
  // Initialize node->children based on 'shape'. All children are assigned the
  // the given 'init_value'.
  void InitChildren(const Shape& shape, const T& init_value, Node* node,
                    Index* index);

  // Initialize node->children based on 'shape'. All children have
  // default-constructed data values.
  void InitChildren(const Shape& shape, Node* node, Index* index);

  // Returns the number of subshapes, including interior nodes, in shape.
  int64_t CountSubshapes(const Shape& shape);

  // Helpers for traversing the shape via ForEachElement. The helpers
  // recursively traverse the subtree rooted at "index" (defined as in
  // ShapeUtil::GetSubshape).
  template <typename Fn>
  static Status ForEachHelper(const Fn& func, const std::vector<Node>& nodes);
  template <typename Fn>
  static Status ForEachMutableHelper(const Fn& func, std::vector<Node>* nodes);

  // Return the tree node at the given index.
  Node* Lookup(ShapeIndexView index);
  const Node* Lookup(ShapeIndexView index) const;

  // The nodes in this shape tree.
  std::vector<Node> nodes_;

  // Index table for node lookups.
  std::vector<Index> index_table_;

  // If we own our Shape, this field contains it, and shape_ is a pointer into
  // here.  Otherwise if we don't own our shape, this is nullptr.
  std::shared_ptr<Shape> shape_storage_;

  // The XLA shape mirrored in this ShapeTree.  This is either
  // shape_storage_.get() or the Shape pointer passed to our constructor.
  const Shape* shape_;
};

// Internal iterator that performs a pre-order walk. This is cheap to copy.
// The iterator value_type is equivalent to a
// std::pair<ShapeIndex,T>&, similar to std::map.
template <typename ContainerType, typename IteratorType, typename ValueType>
class ShapeTreeIterator
    : public std::iterator<std::bidirectional_iterator_tag, ValueType> {
 public:
  ShapeTreeIterator(ContainerType* nodes, IteratorType node)
      : nodes_(nodes), node_(std::move(node)) {}

  ShapeTreeIterator& operator++() {
    ++node_;
    return *this;
  }
  ShapeTreeIterator operator++(int) {
    auto i = *this;
    ++(*this);
    return i;
  }

  ShapeTreeIterator& operator--() {
    --node_;
    return *this;
  }
  ShapeTreeIterator operator--(int) {
    auto i = *this;
    --(*this);
    return i;
  }

  bool operator==(const ShapeTreeIterator& other) const {
    return node_ == other.node_;
  }
  bool operator!=(const ShapeTreeIterator& other) const {
    return node_ != other.node_;
  }
  ValueType& operator*() const { return node_->data; }
  ValueType* operator->() const { return &node_->data; }

 private:
  ContainerType* nodes_;
  IteratorType node_;
};

// Internal iterator that performs a pre-order walk of the leaves. This is cheap
// to copy. The iterator value_type is equivalent to a std::pair<ShapeIndex,T>&,
// similar to std::map.
template <typename ContainerType, typename IteratorType, typename ValueType>
class ShapeTreeLeafIterator
    : public std::iterator<std::bidirectional_iterator_tag, ValueType> {
 public:
  ShapeTreeLeafIterator(ContainerType* nodes, IteratorType node)
      : nodes_(nodes), node_(std::move(node)) {
    while (node_ != nodes_->end() && !node_->is_leaf) {
      ++node_;
    }
  }

  ShapeTreeLeafIterator& operator++() {
    ++node_;
    while (node_ != nodes_->end() && !node_->is_leaf) {
      ++node_;
    }
    return *this;
  }
  ShapeTreeLeafIterator operator++(int) {
    auto i = *this;
    ++(*this);
    return i;
  }

  ShapeTreeLeafIterator& operator--() {
    --node_;
    while (node_ > nodes_->begin() && !node_->is_leaf) {
      --node_;
    }
    return *this;
  }
  ShapeTreeLeafIterator operator--(int) {
    auto i = *this;
    --(*this);
    return i;
  }

  bool operator==(const ShapeTreeLeafIterator& other) const {
    return node_ == other.node_;
  }
  bool operator!=(const ShapeTreeLeafIterator& other) const {
    return node_ != other.node_;
  }
  ValueType& operator*() const { return node_->data; }
  ValueType* operator->() const { return &node_->data; }

 private:
  ContainerType* nodes_;
  IteratorType node_;
};

template <typename T>
int64_t ShapeTree<T>::CountSubshapes(const Shape& shape) {
  int64_t current_count = 1;
  if (shape.IsTuple()) {
    int64_t count = ShapeUtil::TupleElementCount(shape);
    for (int i = 0; i < count; ++i) {
      current_count += CountSubshapes(shape.tuple_shapes(i));
    }
  }
  return current_count;
}

template <typename T>
void ShapeTree<T>::InitChildren(const Shape& shape, const T& init_value,
                                Node* node, Index* index) {
  if (shape.IsTuple()) {
    const int64_t size = ShapeUtil::TupleElementCount(shape);
#ifndef NDEBUG
    index->children_count = size;
#endif
    node->is_leaf = false;
    ShapeIndex shape_index = node->data.first;
    shape_index.push_back(0);

    // At the end of the index_table, reserve a continuous space to hold the
    // children of current node. In order to enforce the invariant that all
    // children of a given node are placed together, we need to do the
    // reservation before we recurse into any of its children.
    int64_t children_start_position = index_table_.size();
    index_table_.resize(index_table_.size() + size);

    for (int i = 0; i < size; ++i) {
      shape_index[shape_index.size() - 1] = i;
      index_table_[children_start_position + i].index = nodes_.size();
      // The first child of the node in the index table is placed at the end of
      // the table.
      index_table_[children_start_position + i].children_start =
          index_table_.size();
      nodes_.emplace_back(shape_index, init_value);
      InitChildren(shape.tuple_shapes(i), init_value, &nodes_.back(),
                   &index_table_[children_start_position + i]);
    }
  } else {
#ifndef NDEBUG
    index->children_count = 0;
#endif
  }
}

template <typename T>
void ShapeTree<T>::InitChildren(const Shape& shape, Node* node, Index* index) {
  if (shape.IsTuple()) {
    const int64_t size = ShapeUtil::TupleElementCount(shape);
#ifndef NDEBUG
    index->children_count = size;
#endif
    node->is_leaf = false;
    ShapeIndex shape_index = node->data.first;
    shape_index.push_back(0);

    // At the end of the index_table, reserve a continuous space to hold the
    // children of current node. In order to enforce the invariant that all
    // children of a given node are placed together, we need to do the
    // reservation before we recurse into any of its children.
    int64_t children_start_position = index_table_.size();
    index_table_.resize(index_table_.size() + size);

    for (int i = 0; i < size; ++i) {
      shape_index[shape_index.size() - 1] = i;
      index_table_[children_start_position + i].index = nodes_.size();
      // The first child of the node in the index table is placed at the end of
      // the table.
      index_table_[children_start_position + i].children_start =
          index_table_.size();
      nodes_.emplace_back(shape_index);
      InitChildren(shape.tuple_shapes(i), &nodes_.back(),
                   &index_table_[children_start_position + i]);
    }
  } else {
#ifndef NDEBUG
    index->children_count = 0;
#endif
  }
}

template <typename T>
ShapeTree<T>::ShapeTree(Shape shape)
    : shape_storage_(std::make_shared<Shape>(std::move(shape))),
      shape_(shape_storage_.get()) {
  const int64_t count = CountSubshapes(*shape_);
  nodes_.reserve(count);
  nodes_.emplace_back(ShapeIndex{});

  index_table_.reserve(count);
  index_table_.emplace_back(Index{0, 1});
  InitChildren(*shape_, &nodes_[0], &index_table_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape* shape) : shape_(shape) {
  const int64_t count = CountSubshapes(*shape_);
  nodes_.reserve(count);
  nodes_.emplace_back(ShapeIndex{});

  index_table_.reserve(count);
  index_table_.emplace_back(Index{0, 1});
  InitChildren(*shape_, &nodes_[0], &index_table_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const std::shared_ptr<Shape>& shape)
    : shape_storage_(shape), shape_(shape_storage_.get()) {
  const int64_t count = CountSubshapes(*shape_);
  nodes_.reserve(count);
  nodes_.emplace_back(ShapeIndex{});

  index_table_.reserve(count);
  index_table_.emplace_back(Index{0, 1});
  InitChildren(*shape_, &nodes_[0], &index_table_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(Shape shape, const T& init_value)
    : shape_storage_(std::make_shared<Shape>(std::move(shape))),
      shape_(shape_storage_.get()) {
  const int64_t count = CountSubshapes(*shape_);
  nodes_.reserve(count);
  nodes_.emplace_back(ShapeIndex{}, init_value);

  index_table_.reserve(count);
  index_table_.emplace_back(Index{0, 1});
  InitChildren(*shape_, init_value, &nodes_[0], &index_table_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape* shape, const T& init_value)
    : shape_(shape) {
  const int64_t count = CountSubshapes(*shape_);
  nodes_.reserve(count);
  nodes_.emplace_back(ShapeIndex{}, init_value);

  index_table_.reserve(count);
  index_table_.emplace_back(Index{0, 1});
  InitChildren(*shape_, init_value, &nodes_[0], &index_table_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const std::shared_ptr<Shape>& shape,
                        const T& init_value)
    : shape_storage_(shape), shape_(shape_storage_.get()) {
  const int64_t count = CountSubshapes(*shape_);
  nodes_.reserve(count);
  nodes_.emplace_back(ShapeIndex{}, init_value);

  index_table_.reserve(count);
  index_table_.emplace_back(Index{0, 1});
  InitChildren(*shape_, init_value, &nodes_[0], &index_table_[0]);
}

template <typename T>
const T& ShapeTree<T>::element(ShapeIndexView index) const {
  return Lookup(index)->data.second;
}

template <typename T>
T* ShapeTree<T>::mutable_element(ShapeIndexView index) {
  return &Lookup(index)->data.second;
}

template <typename T>
internal::ShapeTreeNode<T>* ShapeTree<T>::Lookup(ShapeIndexView index) {
  Index* iter = &index_table_[0];
  for (const int64_t i : index) {
    CHECK_GE(i, 0);
#ifndef NDEBUG
    CHECK_LT(i, iter->children_count);
#endif
    iter = &index_table_[iter->children_start + i];
  }

  return &nodes_[iter->index];
}

template <typename T>
const internal::ShapeTreeNode<T>* ShapeTree<T>::Lookup(
    ShapeIndexView index) const {
  return const_cast<ShapeTree*>(this)->Lookup(index);
}

/* static */
template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachHelper(const Fn& func,
                                   const std::vector<Node>& nodes) {
  for (const auto& node : nodes) {
    TF_RETURN_IF_ERROR(func(node.data.first, node.data.second));
  }
  return Status::OK();
}

/* static */
template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachMutableHelper(const Fn& func,
                                          std::vector<Node>* nodes) {
  for (auto& node : *nodes) {
    TF_RETURN_IF_ERROR(func(node.data.first, &node.data.second));
  }
  return Status::OK();
}

template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachElementWithStatus(const Fn& func) const {
  return ForEachHelper(func, nodes_);
}

template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachMutableElementWithStatus(const Fn& func) {
  return ForEachMutableHelper(func, &nodes_);
}

template <typename T>
template <typename Fn>
void ShapeTree<T>::ForEachElement(const Fn& func) const {
  return ForEachHelper(
             [&func](const ShapeIndex& index, const T& data) {
               func(index, data);
               return Status::OK();
             },
             nodes_)
      .IgnoreError();
}

template <typename T>
template <typename Fn>
void ShapeTree<T>::ForEachMutableElement(const Fn& func) {
  return ForEachMutableHelper(
             [&func](const ShapeIndex& index, T* data) {
               func(index, data);
               return Status::OK();
             },
             &nodes_)
      .IgnoreError();
}

template <typename T>
void ShapeTree<T>::CopySubtreeFrom(const ShapeTree<T>& other,
                                   const ShapeIndex& source_base_index,
                                   const ShapeIndex& target_base_index) {
  CHECK(ShapeUtil::Compatible(
      ShapeUtil::GetSubshape(shape(), target_base_index),
      ShapeUtil::GetSubshape(other.shape(), source_base_index)))
      << ShapeUtil::GetSubshape(shape(), target_base_index) << " vs "
      << ShapeUtil::GetSubshape(other.shape(), source_base_index);
  ForEachMutableElement([this, &other, &source_base_index, &target_base_index](
                            const ShapeIndex& index, T* data) {
    // Copy the data element only if index is in the
    // subtree rooted at target_base_index.
    for (int i = 0; i < target_base_index.size(); ++i) {
      if (i >= index.size() || index[i] != target_base_index[i]) {
        return;
      }
    }
    // Construct source element index to copy from.
    ShapeIndex source_index = source_base_index;
    for (int i = target_base_index.size(); i < index.size(); ++i) {
      source_index.push_back(index[i]);
    }
    *data = other.element(source_index);
  });
}

template <typename T>
StatusOr<ShapeTree<T>> ShapeTree<T>::SubShapeTree(
    const ShapeIndex& index) const {
  TF_ASSIGN_OR_RETURN(const Shape* sub_shape,
                      ShapeUtil::TryGetSubshape(shape(), index));
  ShapeTree<T> sub_shape_tree(*sub_shape);
  sub_shape_tree.CopySubtreeFrom(*this, index, {});
  return std::move(sub_shape_tree);
}

template <typename T>
bool ShapeTree<T>::operator==(const ShapeTree<T>& other) const {
  bool equal = true;
  ForEachElement([&other, &equal](const ShapeIndex& index, const T& data) {
    if (data != other.element(index)) {
      equal = false;
    }
  });
  return equal;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
