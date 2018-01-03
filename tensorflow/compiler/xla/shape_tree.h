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
  T data;

  // Children of this node.
  std::vector<std::unique_ptr<ShapeTreeNode>> children;

  ShapeTreeNode() = default;
  explicit ShapeTreeNode(const T& data) : data(data) {}

  ShapeTreeNode(const ShapeTreeNode& other)
      : data(other.data), children(other.children.size()) {
    for (size_t i = 0; i < children.size(); ++i) {
      children[i] = MakeUnique<ShapeTreeNode>(*other.children[i]);
    }
  }

  ShapeTreeNode& operator=(const ShapeTreeNode& other) {
    if (this != &other) {
      data = other.data;
      children.resize(other.children.size());
      for (size_t i = 0; i < children.size(); ++i) {
        children[i] = MakeUnique<ShapeTreeNode>(*other.children[i]);
      }
    }
    return *this;
  }
};

}  // namespace internal

template <typename T, bool is_const>
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
  friend class ShapeTreeIterator<T, /*is_const=*/true>;
  friend class ShapeTreeIterator<T, /*is_const=*/false>;

 public:
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

  // Create ShapeTree with the given shape, and init_value for all nodes.
  ShapeTree(Shape shape, const T& init_value);
  ShapeTree(const Shape* shape, const T& init_value);

  ShapeTree(const ShapeTree& other) { *this = other; }
  ShapeTree(ShapeTree&&) = default;

  ShapeTree& operator=(const ShapeTree& other) {
    root_ = other.root_;

    // Fix up internal pointer if necessary.
    if (other.shape_storage_) {
      CHECK_EQ(other.shape_, other.shape_storage_.get());
      shape_storage_.reset(new Shape(*other.shape_));
      shape_ = shape_storage_.get();
    } else {
      shape_ = other.shape_;
    }

    return *this;
  }

  ShapeTree& operator=(ShapeTree&& other) = default;

  // Returns the data element associated with the array in the shape at the
  // given index (see ShapeUtil::GetSubshape for how indexes are defined).
  const T& element(const ShapeIndex& index) const;
  T* mutable_element(const ShapeIndex& index);

  // Return the shape represented with this ShapeTree.
  const Shape& shape() const { return *shape_; }

  // Returns true if the node at the given index is a leaf node (an array
  // shape).
  bool IsLeaf(const ShapeIndex& index) const {
    return Lookup(index)->children.empty();
  }

  // iterator implements a forward_iterator with value_type =
  // std::pair<ShapeIndex, T&>
  using iterator = ShapeTreeIterator<T, /*is_const=*/false>;
  using const_iterator = ShapeTreeIterator<T, /*is_const=*/true>;

  // begin/end for iterating over all nodes.
  iterator begin() {
    return iterator(&root_, /*iterate_leaves_only=*/false,
                    /*reverse=*/false);
  }
  iterator end() {
    return iterator(nullptr, /*iterate_leaves_only=*/false,
                    /*reverse=*/false);
  }
  const_iterator begin() const {
    return const_iterator(&root_, /*iterate_leaves_only=*/false,
                          /*reverse=*/false);
  }
  const_iterator end() const {
    return const_iterator(nullptr, /*iterate_leaves_only=*/false,
                          /*reverse=*/false);
  }

  // rbegin/rend for iterating over all nodes in reverse.
  iterator rbegin() {
    return iterator(&root_, /*iterate_leaves_only=*/false,
                    /*reverse=*/true);
  }
  iterator rend() {
    return iterator(nullptr, /*iterate_leaves_only=*/false,
                    /*reverse=*/true);
  }
  const_iterator rbegin() const {
    return const_iterator(&root_, /*iterate_leaves_only=*/false,
                          /*reverse=*/true);
  }
  const_iterator rend() const {
    return const_iterator(nullptr, /*iterate_leaves_only=*/false,
                          /*reverse=*/true);
  }

  // leaf_begin()/leaf_end() iterates over all leaf nodes (nodes with no
  // children).
  iterator leaf_begin() {
    return iterator(&root_, /*iterate_leaves_only=*/true, /*reverse=*/false);
  }
  iterator leaf_end() {
    return iterator(nullptr, /*iterate_leaves_only=*/true,
                    /*reverse=*/false);
  }
  const_iterator leaf_begin() const {
    return const_iterator(&root_, /*iterate_leaves_only=*/true,
                          /*reverse=*/false);
  }
  const_iterator leaf_end() const {
    return const_iterator(nullptr, /*iterate_leaves_only=*/true,
                          /*reverse=*/false);
  }
  // range-based iterator for leaf_begin()/leaf_end().
  tensorflow::gtl::iterator_range<iterator> leaves() {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }
  tensorflow::gtl::iterator_range<const_iterator> leaves() const {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }

  iterator leaf_rbegin() {
    return iterator(&root_, /*iterate_leaves_only=*/true, /*reverse=*/true);
  }
  iterator leaf_rend() {
    return iterator(nullptr, /*iterate_leaves_only=*/true,
                    /*reverse=*/true);
  }
  const_iterator leaf_rbegin() const {
    return const_iterator(&root_, /*iterate_leaves_only=*/true,
                          /*reverse=*/true);
  }
  const_iterator leaf_rend() const {
    return const_iterator(nullptr, /*iterate_leaves_only=*/true,
                          /*reverse=*/true);
  }

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
  using Node = internal::ShapeTreeNode<T>;

  // Initialize node->children based on 'shape'. All children are assigned the
  // the given 'init_value'.
  void InitChildren(const Shape& shape, const T& init_value, Node* node);

  // Initialize node->children based on 'shape'. All children have
  // default-constructed data values.
  void InitChildren(const Shape& shape, Node* node);

  // Helpers for traversing the shape via ForEachElement. The helpers
  // recursively traverse the subtree rooted at "index" (defined as in
  // ShapeUtil::GetSubshape).
  template <typename Fn>
  static Status ForEachHelper(const Fn& func, const Node& node,
                              ShapeIndex* index);
  template <typename Fn>
  static Status ForEachMutableHelper(const Fn& func, Node* node,
                                     ShapeIndex* index);

  // Return the tree node at the given index.
  Node* Lookup(const ShapeIndex& index);
  const Node* Lookup(const ShapeIndex& index) const;

  // The root node, which contains all other nodes.
  Node root_;

  // If we own our Shape, this field contains it, and shape_ is a pointer into
  // here.  Otherwise if we don't own our shape, this is nullptr.
  std::unique_ptr<Shape> shape_storage_;

  // The XLA shape mirrored in this ShapeTree.  This is either
  // shape_storage_.get() or the Shape pointer passed to our constructor.
  const Shape* shape_;
};

// Internal iterator that performs a pre-order walk. This is copyable, but
// contains a vector so isn't cheap to copy. This also means post-increment is
// expensive. The iterator value_type is equivalent to a std::pair<ShapeIndex,
// T&>, similar to std::map. The non-const iterator's T& type can be mutated
// in-place.
template <typename T, bool is_const>
class ShapeTreeIterator : public std::iterator<std::forward_iterator_tag,
                                               std::pair<ShapeIndex, T&>> {
 public:
  using value_type =
      typename std::conditional<is_const, std::pair<ShapeIndex, const T&>,
                                std::pair<ShapeIndex, T&>>::type;
  using NodeType =
      typename std::conditional<is_const, const typename ShapeTree<T>::Node,
                                typename ShapeTree<T>::Node>::type;

  // Construct an iterator pointing at node. Node must either be the tree root
  // or nullptr (which is equivalent to end() and should not be dereferenced or
  // incremented). If iterate_leaves_only is true, the iterator will not include
  // interior tree nodes, only leaves. If reverse is true, the iterator will
  // visit nodes in the reverse of pre-order traversal.
  ShapeTreeIterator(NodeType* node, bool iterate_leaves_only, bool reverse)
      : node_(node),
        iterate_leaves_only_(iterate_leaves_only),
        reverse_(reverse) {
    if (node_) {
      if (reverse_) {
        while (!node_->children.empty()) {
          const int child_index = node_->children.size() - 1;
          stack_.push_back({node_, child_index});
          node_ = node_->children[child_index].get();
        }
      } else {
        if (!node_->children.empty() && iterate_leaves_only) {
          ++*this;
        }
      }
    }
  }
  ShapeTreeIterator(const ShapeTreeIterator& other)
      : node_(other.node_),
        stack_(other.stack_),
        iterate_leaves_only_(other.iterate_leaves_only_),
        reverse_(other.reverse_) {}

  ShapeTreeIterator& operator++() {
    CHECK_NE(nullptr, node_) << "walking off the end() of an iterator!";
    if (reverse_) {
      while (!stack_.empty()) {
        node_ = stack_.back().first;
        int64 next_child_index = stack_.back().second - 1;
        stack_.pop_back();
        if (next_child_index < 0) {
          if (!iterate_leaves_only_) {
            // All children are visited, yield <node_>.
            return *this;
          }
        } else {
          stack_.push_back({node_, next_child_index});
          node_ = node_->children[next_child_index].get();
          while (!node_->children.empty()) {
            const int child_index = node_->children.size() - 1;
            stack_.push_back({node_, child_index});
            node_ = node_->children[child_index].get();
          }
          return *this;
        }
      }
    } else {
      // We're doing a pre-order walk, so if our current node has children take
      // the first child.
      if (!node_->children.empty()) {
        stack_.push_back({node_, /*child-index=*/0});
        node_ = node_->children[0].get();
        if (node_->children.empty() || !iterate_leaves_only_) {
          return *this;
        } else {
          // This is a non-leaf; tail-recurse.
          return ++(*this);
        }
      }
      // Otherwise we are currently at a leaf. Walk back up until a node
      // contains a child we haven't visited yet.
      while (!stack_.empty()) {
        node_ = stack_.back().first;
        int64 next_child_index = stack_.back().second + 1;
        stack_.pop_back();
        if (node_->children.size() > next_child_index) {
          stack_.push_back({node_, next_child_index});
          node_ = node_->children[next_child_index].get();

          if (node_->children.empty() || !iterate_leaves_only_) {
            return *this;
          } else {
            // This is a non-leaf; tail-recurse.
            return ++(*this);
          }
        }
      }
    }
    // We've walked off the end of the tree. Set node_ to nullptr to signify
    // end().
    node_ = nullptr;
    current_.reset();
    return *this;
  }
  ShapeTreeIterator operator++(int) {
    auto i = *this;
    ++(*this);
    return i;
  }
  bool operator==(const ShapeTreeIterator& other) const {
    return node_ == other.node_;
  }
  bool operator!=(const ShapeTreeIterator& other) const {
    return node_ != other.node_;
  }
  value_type& operator*() { return UpdateCurrent(); }
  value_type* operator->() { return &UpdateCurrent(); }

 private:
  // Updates the current_ member to reflect the current state.
  value_type& UpdateCurrent() {
    ShapeIndex index;
    for (auto& node_and_index : stack_) {
      index.push_back(node_and_index.second);
    }
    current_ = MakeUnique<value_type>(index, node_->data);
    return *current_;
  }

  // The node to which this iterator is pointing. This is the source of truth in
  // the iterator - the stack only exists to facilitate walking back from
  // children to parents.
  NodeType* node_;
  // Stack of {node, child-index} pairs of the path taken from the root to get
  // to node_. This allows us to backtrack and know where to go next.
  std::vector<std::pair<NodeType*, int64>> stack_;
  // True if we should not include interior nodes in our walk.
  bool iterate_leaves_only_;
  // True if we should yield the reverse of the pre-order traversal.
  bool reverse_;
  // Placeholder for the current value. Ideally this wouldn't exist and would
  // just be an rvalue, but operator -> needs to return a pointer to something.
  // We cannot just use a plain old value_type as it contains a reference so
  // cannot be default-constructed.
  std::unique_ptr<value_type> current_;
};

template <typename T>
void ShapeTree<T>::InitChildren(const Shape& shape, const T& init_value,
                                Node* node) {
  if (ShapeUtil::IsTuple(shape)) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      node->children.emplace_back(new Node(init_value));
      InitChildren(shape.tuple_shapes(i), init_value,
                   node->children.back().get());
    }
  }
}

template <typename T>
void ShapeTree<T>::InitChildren(const Shape& shape, Node* node) {
  if (ShapeUtil::IsTuple(shape)) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      node->children.emplace_back(new Node());
      InitChildren(shape.tuple_shapes(i), node->children.back().get());
    }
  }
}

template <typename T>
ShapeTree<T>::ShapeTree(Shape shape)
    : root_(),
      shape_storage_(MakeUnique<Shape>(std::move(shape))),
      shape_(shape_storage_.get()) {
  // The shape_ field is just used to hold the structure of the shape.
  // It should not be relied upon to store layout information.
  LayoutUtil::ClearLayout(shape_storage_.get());
  InitChildren(*shape_, &root_);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape* shape) : root_(), shape_(shape) {
  InitChildren(*shape_, &root_);
}

template <typename T>
ShapeTree<T>::ShapeTree(Shape shape, const T& init_value)
    : root_(init_value),
      shape_storage_(MakeUnique<Shape>(std::move(shape))),
      shape_(shape_storage_.get()) {
  // The shape_ field is just used to hold the structure of the shape.
  // It should not be relied upon to store layout information.
  LayoutUtil::ClearLayout(shape_storage_.get());
  InitChildren(*shape_, init_value, &root_);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape* shape, const T& init_value)
    : root_(init_value), shape_(shape) {
  InitChildren(*shape_, init_value, &root_);
}

template <typename T>
const T& ShapeTree<T>::element(const ShapeIndex& index) const {
  return Lookup(index)->data;
}

template <typename T>
T* ShapeTree<T>::mutable_element(const ShapeIndex& index) {
  return &Lookup(index)->data;
}

template <typename T>
internal::ShapeTreeNode<T>* ShapeTree<T>::Lookup(const ShapeIndex& index) {
  Node* node = &root_;
  for (const int64 i : index) {
    CHECK_GE(i, 0);
    CHECK_LT(i, node->children.size());
    node = node->children[i].get();
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
Status ShapeTree<T>::ForEachHelper(const Fn& func, const Node& node,
                                   ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(*index, node.data));
  for (int64 i = 0; i < node.children.size(); ++i) {
    index->push_back(i);
    TF_RETURN_IF_ERROR(ForEachHelper(func, *node.children[i], index));
    index->pop_back();
  }
  return Status::OK();
}

/* static */
template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachMutableHelper(const Fn& func, Node* node,
                                          ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(*index, &node->data));
  for (int64 i = 0; i < node->children.size(); ++i) {
    index->push_back(i);
    TF_RETURN_IF_ERROR(
        ForEachMutableHelper(func, node->children[i].get(), index));
    index->pop_back();
  }
  return Status::OK();
}

template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachElementWithStatus(const Fn& func) const {
  ShapeIndex index;
  return ForEachHelper(func, root_, &index);
}

template <typename T>
template <typename Fn>
Status ShapeTree<T>::ForEachMutableElementWithStatus(const Fn& func) {
  ShapeIndex index;
  return ForEachMutableHelper(func, &root_, &index);
}

template <typename T>
template <typename Fn>
void ShapeTree<T>::ForEachElement(const Fn& func) const {
  ShapeIndex index;
  return ForEachHelper(
             [&func](const ShapeIndex& index, const T& data) {
               func(index, data);
               return Status::OK();
             },
             root_, &index)
      .IgnoreError();
}

template <typename T>
template <typename Fn>
void ShapeTree<T>::ForEachMutableElement(const Fn& func) {
  ShapeIndex index;
  return ForEachMutableHelper(
             [&func](const ShapeIndex& index, T* data) {
               func(index, data);
               return Status::OK();
             },
             &root_, &index)
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
