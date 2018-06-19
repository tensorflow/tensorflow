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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace internal {

// Internal representation of each node in a ShapeTree.
template <typename T>
struct ShapeTreeNode {
  // Data corresponding to this node.
  std::pair<ShapeIndex, T> data;

  // Children of this node, as indices into the container's nodes_ array.
  std::vector<size_t> children;

  // Tells whether this is a leaf node.
  bool is_leaf = true;

  explicit ShapeTreeNode(ShapeIndex index)
      : ShapeTreeNode(std::move(index), T()) {}
  ShapeTreeNode(ShapeIndex index, T data)
      : data(std::move(index), std::move(data)) {}
};

}  // namespace internal

template <typename ContainerType, typename IteratorType, typename ValueType>
class ShapeTreeIterator;

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
  const T& element(const ShapeIndex& index) const;
  T* mutable_element(const ShapeIndex& index);

  // Return the shape represented with this ShapeTree.
  const Shape& shape() const { return *shape_; }

  // Replaces *only* the underlying shape of this ShapeTree. The caller must own
  // the Shape object and hence shape_storage_ is not updated.
  //
  // Only safe to use this if the ShapeTree was constructed with 'explicit
  // ShapeTree(const Shape* shape)' or is moved from one such ShapeTree. The
  // caller must ensure that the input shape is consistent with the underlying
  // tree.
  void replace_shape_ptr(const Shape* shape) {
    CHECK(shape_storage_.get() == nullptr);
    shape_ = shape;
  }

  // Returns true if the node at the given index is a leaf node (an array
  // shape).
  bool IsLeaf(const ShapeIndex& index) const { return Lookup(index)->is_leaf; }

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

  // begin/end for iterating over all nodes.
  iterator begin() {
    return iterator(&nodes_, nodes_.begin(),
                    /*iterate_leaves_only=*/false);
  }
  iterator end() {
    return iterator(&nodes_, nodes_.end(),
                    /*iterate_leaves_only=*/false);
  }
  const_iterator begin() const {
    return const_iterator(&nodes_, nodes_.begin(),
                          /*iterate_leaves_only=*/false);
  }
  const_iterator end() const {
    return const_iterator(&nodes_, nodes_.end(),
                          /*iterate_leaves_only=*/false);
  }

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
  iterator leaf_begin() {
    return iterator(&nodes_, nodes_.begin(),
                    /*iterate_leaves_only=*/true);
  }
  iterator leaf_end() {
    return iterator(&nodes_, nodes_.end(),
                    /*iterate_leaves_only=*/true);
  }
  const_iterator leaf_begin() const {
    return const_iterator(&nodes_, nodes_.begin(),
                          /*iterate_leaves_only=*/true);
  }
  const_iterator leaf_end() const {
    return const_iterator(&nodes_, nodes_.end(),
                          /*iterate_leaves_only=*/true);
  }
  // range-based iterator for leaf_begin()/leaf_end().
  tensorflow::gtl::iterator_range<iterator> leaves() {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }
  tensorflow::gtl::iterator_range<const_iterator> leaves() const {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }

  reverse_iterator leaf_rbegin() { return reverse_iterator(leaf_end()); }
  reverse_iterator leaf_rend() { return reverse_iterator(leaf_begin()); }
  const_reverse_iterator leaf_rbegin() const {
    return const_reverse_iterator(leaf_end());
  }
  const_reverse_iterator leaf_rend() const {
    return const_reverse_iterator(leaf_begin());
  }

  // Returns an iterator pointing to the given ShapeIndex.
  // REQUIRES: index must exist in the ShapeTree.
  iterator find(const ShapeIndex& index) {
    Node* element = Lookup(index);
    return iterator(&nodes_, typename std::vector<Node>::iterator(element),
                    /*iterate_leaves_only=*/false);
  }
  const_iterator find(const ShapeIndex& index) const {
    Node* element = Lookup(index);
    return iterator(&nodes_,
                    typename std::vector<Node>::const_iterator(element),
                    /*iterate_leaves_only=*/false);
  }

  // Returns the number of leaf nodes in the tree.
  int64 leaf_count() const { return std::distance(leaf_begin(), leaf_end()); }

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

  // Copy the subtree of values from 'other' rooted at ShapeIndex
  // 'source_base_index' into the subtree of value in this ShapeTree rooted at
  // 'target_base_index'.
  //
  // Precondition: The subshape of other.shape() at index source_base_index must
  // be compatible with the subshape of shape() at index target_base_index.
  void CopySubtreeFrom(const ShapeTree<T>& other,
                       const ShapeIndex& source_base_index,
                       const ShapeIndex& target_base_index);

  bool operator==(const ShapeTree<T>& other) const;
  bool operator!=(const ShapeTree<T>& other) const { return !(*this == other); }

 private:
  // Initialize node->children based on 'shape'. All children are assigned the
  // the given 'init_value'.
  void InitChildren(const Shape& shape, const T& init_value, Node* node);

  // Initialize node->children based on 'shape'. All children have
  // default-constructed data values.
  void InitChildren(const Shape& shape, Node* node);

  // Returns the number of subshapes, including interior nodes, in shape.
  int64 CountSubshapes(const Shape& shape);

  // Helpers for traversing the shape via ForEachElement. The helpers
  // recursively traverse the subtree rooted at "index" (defined as in
  // ShapeUtil::GetSubshape).
  template <typename Fn>
  static Status ForEachHelper(const Fn& func, const std::vector<Node>& nodes);
  template <typename Fn>
  static Status ForEachMutableHelper(const Fn& func, std::vector<Node>* nodes);

  // Return the tree node at the given index.
  Node* Lookup(const ShapeIndex& index);
  const Node* Lookup(const ShapeIndex& index) const;

  // The nodes in this shape tree.
  std::vector<Node> nodes_;

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
  ShapeTreeIterator(ContainerType* nodes, IteratorType node,
                    bool iterate_leaves_only)
      : nodes_(nodes),
        node_(std::move(node)),
        iterate_leaves_only_(iterate_leaves_only) {
    while (iterate_leaves_only && node_ != nodes_->end() && !node_->is_leaf) {
      ++node_;
    }
  }

  ShapeTreeIterator& operator++() {
    ++node_;
    while (iterate_leaves_only_ && node_ != nodes_->end() && !node_->is_leaf) {
      ++node_;
    }
    return *this;
  }
  ShapeTreeIterator operator++(int) {
    auto i = *this;
    ++(*this);
    return i;
  }

  ShapeTreeIterator& operator--() {
    --node_;
    while (iterate_leaves_only_ && node_ > nodes_->begin() && !node_->is_leaf) {
      --node_;
    }
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
  ValueType& operator*() { return node_->data; }
  ValueType* operator->() { return &node_->data; }

 private:
  ContainerType* nodes_;
  IteratorType node_;
  // True if we should not include interior nodes in our walk.
  const bool iterate_leaves_only_;
};

template <typename T>
int64 ShapeTree<T>::CountSubshapes(const Shape& shape) {
  int64 current_count = 1;
  if (ShapeUtil::IsTuple(shape)) {
    int64 count = ShapeUtil::TupleElementCount(shape);
    for (int i = 0; i < count; ++i) {
      current_count += CountSubshapes(shape.tuple_shapes(i));
    }
  }
  return current_count;
}

template <typename T>
void ShapeTree<T>::InitChildren(const Shape& shape, const T& init_value,
                                Node* node) {
  if (ShapeUtil::IsTuple(shape)) {
    const int64 size = ShapeUtil::TupleElementCount(shape);
    node->children.reserve(size);
    node->is_leaf = false;
    ShapeIndex shape_index = node->data.first;
    shape_index.push_back(0);
    for (int i = 0; i < size; ++i) {
      shape_index[shape_index.size() - 1] = i;
      node->children.push_back(nodes_.size());
      nodes_.emplace_back(shape_index, init_value);
      InitChildren(shape.tuple_shapes(i), init_value, &nodes_.back());
    }
  }
}

template <typename T>
void ShapeTree<T>::InitChildren(const Shape& shape, Node* node) {
  if (ShapeUtil::IsTuple(shape)) {
    const int64 size = ShapeUtil::TupleElementCount(shape);
    node->children.reserve(size);
    node->is_leaf = false;
    ShapeIndex shape_index = node->data.first;
    shape_index.push_back(0);
    for (int i = 0; i < size; ++i) {
      shape_index[shape_index.size() - 1] = i;
      node->children.push_back(nodes_.size());
      nodes_.emplace_back(shape_index);
      InitChildren(shape.tuple_shapes(i), &nodes_.back());
    }
  }
}

template <typename T>
ShapeTree<T>::ShapeTree(Shape shape)
    : shape_storage_(std::make_shared<Shape>(std::move(shape))),
      shape_(shape_storage_.get()) {
  // The shape_ field is just used to hold the structure of the shape.
  // It should not be relied upon to store layout information.
  LayoutUtil::ClearLayout(shape_storage_.get());
  nodes_.reserve(CountSubshapes(*shape_));
  nodes_.emplace_back(ShapeIndex{});
  InitChildren(*shape_, &nodes_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape* shape) : shape_(shape) {
  nodes_.reserve(CountSubshapes(*shape_));
  nodes_.emplace_back(ShapeIndex{});
  InitChildren(*shape_, &nodes_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const std::shared_ptr<Shape>& shape)
    : shape_storage_(shape), shape_(shape_storage_.get()) {
  nodes_.reserve(CountSubshapes(*shape_));
  nodes_.emplace_back(ShapeIndex{});
  InitChildren(*shape_, &nodes_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(Shape shape, const T& init_value)
    : shape_storage_(std::make_shared<Shape>(std::move(shape))),
      shape_(shape_storage_.get()) {
  // The shape_ field is just used to hold the structure of the shape.
  // It should not be relied upon to store layout information.
  LayoutUtil::ClearLayout(shape_storage_.get());
  nodes_.reserve(CountSubshapes(*shape_));
  nodes_.emplace_back(ShapeIndex{}, init_value);
  InitChildren(*shape_, init_value, &nodes_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape* shape, const T& init_value)
    : shape_(shape) {
  nodes_.reserve(CountSubshapes(*shape_));
  nodes_.emplace_back(ShapeIndex{}, init_value);
  InitChildren(*shape_, init_value, &nodes_[0]);
}

template <typename T>
ShapeTree<T>::ShapeTree(const std::shared_ptr<Shape>& shape,
                        const T& init_value)
    : shape_storage_(shape), shape_(shape_storage_.get()) {
  nodes_.reserve(CountSubshapes(*shape_));
  nodes_.emplace_back(ShapeIndex{}, init_value);
  InitChildren(*shape_, init_value, &nodes_[0]);
}

template <typename T>
const T& ShapeTree<T>::element(const ShapeIndex& index) const {
  return Lookup(index)->data.second;
}

template <typename T>
T* ShapeTree<T>::mutable_element(const ShapeIndex& index) {
  return &Lookup(index)->data.second;
}

template <typename T>
internal::ShapeTreeNode<T>* ShapeTree<T>::Lookup(const ShapeIndex& index) {
  Node* node = &nodes_[0];
  for (const int64 i : index) {
    CHECK_GE(i, 0);
    CHECK_LT(i, node->children.size());
    node = &nodes_[node->children[i]];
  }
  return node;
}

template <typename T>
const internal::ShapeTreeNode<T>* ShapeTree<T>::Lookup(
    const ShapeIndex& index) const {
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
      ShapeUtil::GetSubshape(other.shape(), source_base_index)));
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
bool ShapeTree<T>::operator==(const ShapeTree<T>& other) const {
  bool equal = true;
  ForEachElement(
      [this, &other, &equal](const ShapeIndex& index, const T& data) {
        if (data != other.element(index)) {
          equal = false;
        }
      });
  return equal;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
