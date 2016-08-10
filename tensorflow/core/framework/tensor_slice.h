/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_FRAMEWORK_TENSOR_SLICE_H_
#define TENSORFLOW_FRAMEWORK_TENSOR_SLICE_H_

#include <string>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// A tensor slice represents a slice of a given tensor. It is represented by a
// list of (start, length) pairs, where the size of the list is the rank of the
// tensor.

class TensorSlice {
 public:
  // Construct a tensor slice: you have a number of ways:
  // -- creating an empty slice
  // -- from just a dimension (in this case it will create a full slice)
  // -- from an array of pairs of integers.
  // -- from a TensorSliceProto protocol buffer
  // -- from a string format of "start,length:start,length..." where each
  //    "start,length" pair represents the slice on one dimension. We allow a
  //    special "-" that means "everything for this dimension". One such example
  //    is:  0,10:-:14,1:-:-
  TensorSlice() {}
  explicit TensorSlice(int dim);
  explicit TensorSlice(const TensorSliceProto& proto);
  explicit TensorSlice(std::initializer_list<std::pair<int, int>> extents);

  static Status Parse(const string& str, TensorSlice* output);
  static TensorSlice ParseOrDie(const string& str) {
    TensorSlice ret;
    Status s = Parse(str, &ret);
    if (!s.ok()) {
      LOG(FATAL) << "Could not parse TensorSlice";
    }
    return ret;
  }

  void Clear();

  // Accessors
  int dims() const { return starts_.size(); }

  int start(int d) const {
    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return starts_[d];
  }

  int length(int d) const {
    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return lengths_[d];
  }

  int end(int d) const {
    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return start(d) + length(d);
  }

  void set_start(int d, int x) {
    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    DCHECK_GE(x, 0);
    starts_[d] = x;
  }

  void set_length(int d, int x) {
    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    lengths_[d] = x;
  }

  // If we have a full slice along dimension "d".
  bool IsFullAt(int d) const { return lengths_[d] < 0; }

  // Set the slice to be a full slice of "dim" dimensions
  void SetFullSlice(int dim);

  // Extend a slice to "dim" dimensions: all the added dimensions are full.
  // Requires: dim >= dims().
  void Extend(int dim);

  // Conversion of a TensorSlice to other formats
  void AsProto(TensorSliceProto* proto) const;
  string DebugString() const;

  // Fill *indices and *sizes from *this (so that we can use the slice()
  // function in eigen tensor). We need a tensor shape in case some of the
  // slices are full slices.
  // We allow NDIMS to be greater than dims(), in which case we will pad the
  // higher dimensions with trivial dimensions.
  template <int NDIMS>
  void FillIndicesAndSizes(
      const TensorShape& shape,
      Eigen::DSizes<Eigen::DenseIndex, NDIMS>* indices,
      Eigen::DSizes<Eigen::DenseIndex, NDIMS>* sizes) const;

  // Interaction with other TensorSlices.

  // Compute the intersection with another slice and if "result" is not
  // nullptr, store the results in *result; returns true is there is any real
  // intersection.
  bool Intersect(const TensorSlice& other, TensorSlice* result) const;
  // A short hand.
  bool Overlaps(const TensorSlice& other) const {
    return Intersect(other, nullptr);
  }

  // Interaction with TensorShape.

  // Slices a shape and stores the result into *result_shape.
  // Requires that the shape and *this have the same rank.
  // For example, given a tensor shape of {3, 4, 5}, and a slice of
  // 1,2:-:0,2, the result shape is {2, 4, 2}.
  Status SliceTensorShape(const TensorShape& shape,
                          TensorShape* result_shape) const;

  // Given slice "sub" where "sub" is fully contained in *this,
  // (meaning that the intersection of "sub" and *this equals "sub"), computes
  // the "relative" slice of "sub" with respect to *this.
  //
  // In other words, if we use A>S to denote slicing a shape S with a slice A,
  // then the function is computing a slice X such that:
  //   X > (this > S) = sub > S
  // for any shape S.
  //
  // In general, along every dimension, the start of the relative slice is the
  // start of the "sub" slice minus the start of *this; the length of the
  // relative slice is the length of the "sub" slice.
  //
  // For example, say we have a shape of {3, 4, 5}, "this" is 0,2:-:1,2, and
  // "sub" is 1,1:2:2,1,2, then the related slice is 1,1:2,2:0,2.
  //
  // The caller needs to make sure that "sub" is indeed a sub-slice of *this;
  // otherwise the result is undefined.
  void ComputeRelative(const TensorSlice& sub, TensorSlice* relative) const;

  // Updates the slice in such a way that it fully covers "other" slice.
  // Note, "other" slice should refer to the same tensor shape.
  // Example:
  //   given a slice [2:4, :, 3:] and "other" slice [:, 1:4, 2:4] the
  //   updated slice would be [:, :, 2:]. Here is why:
  //   dim 0: "2:4"  U  ":"    ->  ":"
  //   dim 1: ":"    U  "1-4"  ->  ":"
  //   dim 2: "3:"   U  "2:4"  ->  "2:"
  void UpdateToCover(const TensorSlice& other);

  // Returns true if the length field was specified in an Extent.
  static bool HasExtentLength(const TensorSliceProto::Extent& extent);

  // Returns the value of the length field in an Extent, or -1 if it
  // is not present.
  static int64 GetExtentLength(const TensorSliceProto::Extent& extent);

 private:
  // a length value of kFullExtent (-1) means we have a full slice at this
  // dimension. It's defined in tensor_slice.cc.
  static const int kFullExtent;

  // TODO(yangke): switch to Eigen once it supports variable size arrays.
  // A value of
  gtl::InlinedVector<int, 4> starts_;
  gtl::InlinedVector<int, 4> lengths_;
};

template <int NDIMS>
void TensorSlice::FillIndicesAndSizes(
    const TensorShape& shape, Eigen::DSizes<Eigen::DenseIndex, NDIMS>* indices,
    Eigen::DSizes<Eigen::DenseIndex, NDIMS>* sizes) const {
  CHECK_EQ(shape.dims(), dims()) << "Incompatible dimensions between shape "
                                 << "slices: shape = " << shape.DebugString()
                                 << ", slice = " << DebugString();
  CHECK_GE(NDIMS, dims()) << "Asking for a " << NDIMS << "-dim slice from "
                          << "a slice of dimension " << dims();
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      (*indices)[d] = 0;
      (*sizes)[d] = shape.dim_size(d);
    } else {
      (*indices)[d] = starts_[d];
      (*sizes)[d] = lengths_[d];
    }
  }
  for (int d = dims(); d < NDIMS; ++d) {
    (*indices)[d] = 0;
    (*sizes)[d] = 1;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TENSOR_SLICE_H_
