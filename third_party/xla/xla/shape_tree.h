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
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/utility/utility.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "xla/tuple_tree.h"

namespace xla {

// A ShapeTree<T> is a tree data structure that mirrors the structure of an
// XLA Shape and holds a value of type T for each subshape.
//
// Key Characteristics:
// - Mirrors an XLA Shape: The tree structure is identical to the Shape's
//   tuple nesting.
// - Value at Each Node: Every node in the tree, whether it corresponds to a
//   tuple or an array, has an associated value of type T.
// - Leaf Nodes: Nodes corresponding to array Shapes are leaf nodes in the
//   ShapeTree.
// - Internal Nodes: Nodes corresponding to tuple Shapes are internal nodes.
// - Unique Elements: Each ShapeIndex in the Shape corresponds to a unique
//   element of type T in the ShapeTree.
//
// Underlying Implementation:
// This class is primarily a wrapper around TupleTree<T>, binding it to an
// xla::Shape. The actual tree data is stored and managed by the internal
// TupleTree instance.
//
// Shape Ownership:
// Normally a ShapeTree owns its Shape (stored in a std::shared_ptr), but for
// efficiency, you can construct it with a const Shape* to avoid copies. In this
// case, the caller must ensure the Shape outlives the ShapeTree.
//
// Example:
//   Shape shape = ShapeUtil::MakeTupleShape({
//       ShapeUtil::MakeShape(F32, {}),  // Index {0}
//       ShapeUtil::MakeShape(S32, {})   // Index {1}
//   });
//   ShapeTree<int> tree(shape, 0); // Initialize all nodes with 0
//   *tree.mutable_element({0}) = 10;
//   *tree.mutable_element({1}) = 20;
//   // The root element at {} also has a value, initialized to 0.
//   LOG(INFO) << tree.element({}); // Prints 0
//   LOG(INFO) << tree.element({0}); // Prints 10
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
  using Node = std::pair<ShapeIndex, T>;
  using Nodes = typename TupleTree<T>::NodePairs;

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
      : ShapeTree(absl::in_place_t{}, shape) {}

  // Create ShapeTree with the given shape, and init_value for all nodes.
  ShapeTree(Shape shape, const T& init_value)
      : ShapeTree(std::make_shared<Shape>(std::move(shape)), init_value) {}

  ShapeTree(const Shape* shape, const T& init_value)
      : ShapeTree(absl::in_place_t{}, shape, init_value) {}

  // Returns the data element associated with the subshape at the
  // given index. This works for any valid index, including internal tuple
  // nodes.
  const T& element(ShapeIndexView index) const {
    return tuple_tree_.element(index);
  }
  T* mutable_element(ShapeIndexView index) {
    return tuple_tree_.mutable_element(index);
  }

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

  bool IsLeaf(ShapeIndexView index) const { return tuple_tree_.IsLeaf(index); }

  using iterator = typename TupleTree<T>::iterator;
  using const_iterator = typename TupleTree<T>::const_iterator;
  using reverse_iterator = typename TupleTree<T>::reverse_iterator;
  using const_reverse_iterator = typename TupleTree<T>::const_reverse_iterator;

  using leaf_iterator = typename TupleTree<T>::leaf_iterator;
  using const_leaf_iterator = typename TupleTree<T>::const_leaf_iterator;
  using reverse_leaf_iterator = typename TupleTree<T>::reverse_leaf_iterator;
  using const_reverse_leaf_iterator =
      typename TupleTree<T>::const_reverse_leaf_iterator;

  iterator begin() { return tuple_tree_.begin(); }
  iterator end() { return tuple_tree_.end(); }
  const_iterator begin() const { return tuple_tree_.begin(); }
  const_iterator end() const { return tuple_tree_.end(); }

  reverse_iterator rbegin() { return tuple_tree_.rbegin(); }
  reverse_iterator rend() { return tuple_tree_.rend(); }
  const_reverse_iterator rbegin() const { return tuple_tree_.rbegin(); }
  const_reverse_iterator rend() const { return tuple_tree_.rend(); }

  // leaf_begin()/leaf_end() iterates over all nodes for which IsLeaf() is true
  // (i.e., array shapes).
  leaf_iterator leaf_begin() { return tuple_tree_.leaf_begin(); }
  leaf_iterator leaf_end() { return tuple_tree_.leaf_end(); }
  const_leaf_iterator leaf_begin() const { return tuple_tree_.leaf_begin(); }
  const_leaf_iterator leaf_end() const { return tuple_tree_.leaf_end(); }

  // range-based iterator for leaf_begin()/leaf_end().
  tsl::gtl::iterator_range<leaf_iterator> leaves() {
    return tuple_tree_.leaves();
  }
  tsl::gtl::iterator_range<const_leaf_iterator> leaves() const {
    return tuple_tree_.leaves();
  }

  reverse_leaf_iterator leaf_rbegin() { return tuple_tree_.leaf_rbegin(); }
  reverse_leaf_iterator leaf_rend() { return tuple_tree_.leaf_rend(); }
  const_reverse_leaf_iterator leaf_rbegin() const {
    return tuple_tree_.leaf_rbegin();
  }
  const_reverse_leaf_iterator leaf_rend() const {
    return tuple_tree_.leaf_rend();
  }

  // Returns an iterator pointing to the given ShapeIndex.
  // REQUIRES: index must exist in the ShapeTree.
  iterator find(ShapeIndexView index) { return tuple_tree_.find(index); }
  const_iterator find(ShapeIndexView index) const {
    return tuple_tree_.find(index);
  }

  // Returns the number of leaf nodes in the tree.
  int64_t leaf_count() const { return ShapeUtil::GetLeafCount(*shape_); }

  // TODO(cjfj): Remove the `ForEach...` methods. They are redundant.
  // Traverses all nodes in the tree in pre-order and calls the given function
  // at each element.
  void ForEachElement(
      absl::FunctionRef<void(const ShapeIndex&, const T&)> func) const {
    tuple_tree_.ForEachElement(func);
  }

  void ForEachMutableElement(
      absl::FunctionRef<void(const ShapeIndex&, T*)> func) {
    tuple_tree_.ForEachMutableElement(func);
  }

  // Like ForEach(Mutable)Element, but the callable returns a absl::Status
  // instead of void.  The first non-OK return value is returned by the ForEach*
  // function.
  absl::Status ForEachElementWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, const T&)> func) const {
    return tuple_tree_.ForEachElementWithStatus(func);
  }

  absl::Status ForEachMutableElementWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, T*)> func) {
    return tuple_tree_.ForEachMutableElementWithStatus(func);
  }

  // Like the above, but traverses all nodes in post-order. Note children are
  // visited in right-to-left order.
  void ForEachElementPostOrder(
      absl::FunctionRef<void(const ShapeIndex&, const T&)> func) const {
    for (auto node = tuple_tree_.rbegin(); node != tuple_tree_.rend(); ++node) {
      func(node->first, node->second);
    }
  }

  void ForEachMutableElementPostOrder(
      absl::FunctionRef<void(const ShapeIndex&, T*)> func) {
    for (auto node = tuple_tree_.rbegin(); node != tuple_tree_.rend(); ++node) {
      func(node->first, &node->second);
    }
  }

  absl::Status ForEachElementPostOrderWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, const T&)> func) const {
    for (auto node = tuple_tree_.rbegin(); node != tuple_tree_.rend(); ++node) {
      TF_RETURN_IF_ERROR(func(node->first, node->second));
    }
    return absl::OkStatus();
  }

  absl::Status ForEachMutableElementPostOrderWithStatus(
      absl::FunctionRef<absl::Status(const ShapeIndex&, T*)> func) {
    for (auto node = tuple_tree_.rbegin(); node != tuple_tree_.rend(); ++node) {
      TF_RETURN_IF_ERROR(func(node->first, &node->second));
    }
    return absl::OkStatus();
  }

  // Maps each element's value in this tree to generate a new ShapeTree<U>
  // with the same shape structure. The function `func` is applied to the value
  // of *every* node (both array and tuple nodes).
  template <typename U>
  ShapeTree<U> Map(absl::FunctionRef<U(const T&)> func) const {
    return ShapeTree<U>(shape_, tuple_tree_.Map(func), shape_storage_);
  }

  template <typename U>
  absl::StatusOr<ShapeTree<U>> MapWithStatus(
      absl::FunctionRef<absl::StatusOr<U>(const T&)> func) const {
    TF_ASSIGN_OR_RETURN(TupleTree<U> new_tuple_tree,
                        tuple_tree_.MapWithStatus(func));
    return ShapeTree<U>(shape_, std::move(new_tuple_tree), shape_storage_);
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

    // Although the shapes are compatible, the underlying tuple tree structures
    // might differ, e.g. if one side was constructed from shape and the other
    // from node pairs.
    CHECK_OK(tuple_tree_.CopyCompatibleSubtreeFrom(other.tuple_tree_, src_index,
                                                   dst_index));
  }

  absl::StatusOr<ShapeTree<T>> SubShapeTree(const ShapeIndex& index) const {
    TF_ASSIGN_OR_RETURN(const Shape* sub_shape,
                        ShapeUtil::TryGetSubshape(shape(), index));
    TF_ASSIGN_OR_RETURN(TupleTree<T> sub_tuple_tree,
                        tuple_tree_.Subtree(index));
    return ShapeTree<T>(sub_shape, std::move(sub_tuple_tree), shape_storage_);
  }

  bool operator==(const ShapeTree<T>& other) const {
    return tuple_tree_ == other.tuple_tree_;
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

  ShapeTree(const Shape* shape, TupleTree<T>&& tuple_tree,
            std::shared_ptr<Shape> shape_storage)
      : tuple_tree_(std::move(tuple_tree)),
        shape_storage_(shape_storage),
        shape_(shape) {}

  // This constructor now always takes an init_value.
  ShapeTree(absl::in_place_t, const Shape* shape, const T& init_value)
      : tuple_tree_(*shape, init_value), shape_(shape) {}

  template <typename... Ts>
  ShapeTree(absl::in_place_t, const Shape* shape)
      : tuple_tree_(TupleTree<T>::Node::FromShape(*shape)), shape_(shape) {}

  TupleTree<T> tuple_tree_;

  // If we own our Shape, this field contains it, and shape_ is a pointer into
  // here.  Otherwise if we don't own our shape, this is nullptr.
  std::shared_ptr<Shape> shape_storage_;

  // The XLA shape mirrored in this ShapeTree.  This is either
  // shape_storage_.get() or the Shape pointer passed to our constructor.
  const Shape* shape_;
};

}  // namespace xla

#endif  // XLA_SHAPE_TREE_H_
