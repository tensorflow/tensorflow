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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A ShapeTree<T> is a recursive data structure which mirrors the structure of a
// XLA shape and holds a value of type T for each array in the shape. For
// array shapes, a ShapeTree trivially holds a single value of type T. For tuple
// shapes which can be an arbitrary tree with arrays at the leaves, a ShapeTree
// is an identically structured tree with data elements of type T at the leaves.
//
// Like the Shape data structure, this is a tree and tuple elements cannot be
// duplicated. That is, every distinct element position in the Shape has a
// unique T object.
template <typename T>
class ShapeTree {
 public:
  explicit ShapeTree(const Shape& shape);
  ShapeTree(const Shape& shape, const T& init_value);
  ShapeTree(const ShapeTree<T>& other);
  ShapeTree<T>& operator=(const ShapeTree<T>& other);

  // Returns the data element associated with the array in the shape at the
  // given index (see ShapeUtil::GetSubshape for how indexes are defined).
  const T& element(const ShapeIndex& index) const;
  T* mutable_element(const ShapeIndex& index);

  // Return the shape represented with this ShapeTree.
  const Shape& shape() const { return *shape_; }

  // Returns true if the node at the given index is a leaf node (an array
  // shape).
  bool IsLeaf(const ShapeIndex& index) const {
    return Lookup(index).elements_.empty();
  }

  // Recursively traverses the shape and calls the given function at each
  // element. The function has the following arguments:
  //
  //   index : the index of the element in the shape. See ShapeUtil::GetSubshape
  //           for definition of index.
  //   is_leaf : Whether this element is a leaf element in the shape. That is,
  //             whether this index corresponds to an array and not a (nested)
  //             tuple element.
  //   data : The data value at this elemnt.
  //
  // If any call to the given function returns a non-OK status, then traversal
  // is aborted and the status value is returned.
  using VisitorFunction = std::function<tensorflow::Status(
      const ShapeIndex& /*index*/, bool /*is_leaf*/, const T& /*data*/)>;
  tensorflow::Status ForEachElement(VisitorFunction func) const;

  using MutableVisitorFunction = std::function<tensorflow::Status(
      const ShapeIndex& /*index*/, bool /*is_leaf*/, T* /*data*/)>;
  tensorflow::Status ForEachMutableElement(MutableVisitorFunction func);

 private:
  // Private default constructor for non-root nodes of the tree.
  ShapeTree() = default;

  // Helpers for traversing the shape via ForEachElement. The helpers
  // recursively traverse the subtree rooted at "index" (defined as in
  // ShapeUtil::GetSubshape).
  static tensorflow::Status ForEachHelperMutable(ShapeIndex* index,
                                                 ShapeTree<T>* shape_tree,
                                                 MutableVisitorFunction func);
  static tensorflow::Status ForEachHelper(ShapeIndex* index,
                                          const ShapeTree<T>& shape_tree,
                                          VisitorFunction func);

  // Copy all the data elements (of type T) from "other" into "this". "this"
  // must have the same tree structure as "other" prior to calling this method.
  void CopyDataElements(const ShapeTree<T>& other);

  // Recursive helper for constructing a subtree beneath "this" node.
  void BuildTree(const Shape& shape);

  // Return the tree node at the given index.
  ShapeTree<T>& Lookup(const ShapeIndex& index);
  const ShapeTree<T>& Lookup(const ShapeIndex& index) const;

  // The data corresponding to the array at this node.
  T data_;

  // The XLA shape mirrored in this ShapeTree. Only the root of the
  // ShapeTree has this member set.
  std::unique_ptr<Shape> shape_;

  // The children of this node in the tree.
  std::vector<std::unique_ptr<ShapeTree>> elements_;
};

template <typename T>
void ShapeTree<T>::BuildTree(const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      elements_.emplace_back(new ShapeTree());
      elements_.back()->BuildTree(shape.tuple_shapes(i));
    }
  }
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape& shape) : shape_(MakeUnique<Shape>(shape)) {
  // The shape_ field is just used to hold the structure of the shape. It should
  // not be relied upon to store layout information.
  LayoutUtil::ClearLayout(shape_.get());
  BuildTree(*shape_);
}

template <typename T>
ShapeTree<T>::ShapeTree(const Shape& shape, const T& init_value)
    : shape_(MakeUnique<Shape>(shape)) {
  LayoutUtil::ClearLayout(shape_.get());
  BuildTree(*shape_);
  TF_CHECK_OK(ForEachMutableElement(
      [&init_value](const ShapeIndex& /*index*/, bool /*is_leaf*/, bool* data) {
        *data = init_value;
        return tensorflow::Status::OK();
      }));
}

template <typename T>
ShapeTree<T>::ShapeTree(const ShapeTree& other)
    : shape_(MakeUnique<Shape>(other.shape())) {
  LayoutUtil::ClearLayout(shape_.get());
  BuildTree(*shape_);
  CopyDataElements(other);
}

template <typename T>
ShapeTree<T>& ShapeTree<T>::operator=(const ShapeTree<T>& other) {
  if (this == &other) {
    return *this;
  }
  elements_.clear();
  shape_ = MakeUnique<Shape>(other.shape());
  LayoutUtil::ClearLayout(shape_.get());

  BuildTree(*shape_);
  CopyDataElements(other);
  return *this;
}

template <typename T>
void ShapeTree<T>::CopyDataElements(const ShapeTree<T>& other) {
  CHECK(ShapeUtil::Compatible(shape(), other.shape()));
  TF_CHECK_OK(ForEachMutableElement(
      [&other](const ShapeIndex& index, bool /*is_leaf*/, T* data) {
        *data = other.element(index);
        return tensorflow::Status::OK();
      }));
}

template <typename T>
const T& ShapeTree<T>::element(const ShapeIndex& index) const {
  return Lookup(index).data_;
}

template <typename T>
T* ShapeTree<T>::mutable_element(const ShapeIndex& index) {
  return &Lookup(index).data_;
}

template <typename T>
ShapeTree<T>& ShapeTree<T>::Lookup(const ShapeIndex& index) {
  ShapeTree<T>* node = this;
  for (auto& i : index) {
    CHECK_GE(i, 0);
    CHECK_LT(i, node->elements_.size());
    node = node->elements_[i].get();
  }
  return *node;
}

template <typename T>
const ShapeTree<T>& ShapeTree<T>::Lookup(const ShapeIndex& index) const {
  return const_cast<ShapeTree<T>*>(this)->Lookup(index);
}

/* static */
template <typename T>
tensorflow::Status ShapeTree<T>::ForEachHelperMutable(
    ShapeIndex* index, ShapeTree<T>* shape_tree,
    ShapeTree<T>::MutableVisitorFunction func) {
  TF_RETURN_IF_ERROR(
      func(*index, shape_tree->elements_.empty(), &shape_tree->data_));
  for (int i = 0; i < shape_tree->elements_.size(); ++i) {
    index->push_back(i);
    TF_RETURN_IF_ERROR(
        ForEachHelperMutable(index, shape_tree->elements_[i].get(), func));
    index->pop_back();
  }

  return tensorflow::Status::OK();
}

/* static */
template <typename T>
tensorflow::Status ShapeTree<T>::ForEachHelper(
    ShapeIndex* index, const ShapeTree<T>& shape_tree,
    ShapeTree<T>::VisitorFunction func) {
  TF_RETURN_IF_ERROR(
      func(*index, shape_tree.elements_.empty(), shape_tree.data_));
  for (int i = 0; i < shape_tree.elements_.size(); ++i) {
    index->push_back(i);
    TF_RETURN_IF_ERROR(ForEachHelper(index, *shape_tree.elements_[i], func));
    index->pop_back();
  }

  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status ShapeTree<T>::ForEachElement(
    ShapeTree<T>::VisitorFunction func) const {
  ShapeIndex index;
  return ForEachHelper(&index, *this, func);
}

template <typename T>
tensorflow::Status ShapeTree<T>::ForEachMutableElement(
    ShapeTree<T>::MutableVisitorFunction func) {
  ShapeIndex index;
  return ForEachHelperMutable(&index, this, func);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
