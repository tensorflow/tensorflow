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

#ifndef TENSORFLOW_CORE_FRAMEWORK_PARTIAL_TENSOR_SHAPE_H_
#define TENSORFLOW_CORE_FRAMEWORK_PARTIAL_TENSOR_SHAPE_H_

#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class PartialTensorShapeIter;  // Declared below

/// Manages the partially known dimensions of a Tensor and their sizes.
class PartialTensorShape {
 public:
  /// \brief Construct an unknown `PartialTensorShape`.
  PartialTensorShape() : is_unknown_(true) {}

  /// \brief Construct a `PartialTensorShape` from the provided sizes.
  /// REQUIRES: `dim_sizes[i] >= 0`
  explicit PartialTensorShape(gtl::ArraySlice<int64> dim_sizes);
  PartialTensorShape(std::initializer_list<int64> dim_sizes)
      : PartialTensorShape(gtl::ArraySlice<int64>(dim_sizes)) {}

  /// REQUIRES: `IsValid(proto)`
  explicit PartialTensorShape(const TensorShapeProto& proto);

  /// Returns `true` iff `proto` is a valid partial tensor shape.
  static bool IsValid(const TensorShapeProto& proto);

  /// Returns `OK` iff `proto` is a valid tensor shape, and a descriptive error
  /// status otherwise.
  static Status IsValidShape(const TensorShapeProto& proto);

  /// Add a dimension to the end ("inner-most"), returns a new
  /// PartialTensorShape.
  /// REQUIRES: `size >= -1`, where -1 means unknown
  PartialTensorShape Concatenate(int64 size) const;

  /// Appends all the dimensions from `shape`.  Returns a new
  /// PartialTensorShape.
  PartialTensorShape Concatenate(const PartialTensorShape& shape) const;

  /// Merges all the dimensions from `shape`.  Returns
  /// `InvalidArgument` error if either `shape` has a different rank
  /// or if any of the dimensions are incompatible.
  Status MergeWith(const PartialTensorShape& shape,
                   PartialTensorShape* result) const;

  /// Return the number of dimensions in the tensor. If the number of
  /// dimensions is unknown, return -1.
  int dims() const { return is_unknown_ ? -1 : dim_sizes_.size(); }

  /// Return true iff the rank and all of the dimensions are well defined
  bool IsFullyDefined() const;

  /// Return true iff the ranks match, and if the
  /// dimensions all either match or one is unknown.
  bool IsCompatibleWith(const PartialTensorShape& shape) const;

  /// Return true iff the dimensions of `shape` are compatible with
  /// `*this`.
  bool IsCompatibleWith(const TensorShape& shape) const;

  /// \brief Returns the number of elements in dimension `d`.
  /// REQUIRES: `0 <= d < dims()`
  int64 dim_size(int d) const {
    DCHECK_GE(d, 0);
    if (is_unknown_) {
      return -1;
    } else {
      DCHECK_LT(d, dims());
      return dim_sizes_[d];
    }
  }

  /// Returns sizes of all dimensions.
  gtl::ArraySlice<int64> dim_sizes() const { return dim_sizes_; }

  /// Fill `*proto` from `*this`.
  void AsProto(TensorShapeProto* proto) const;

  // Fill `*tensor_shape` from `*this`.
  // If `*this` is not fully defined, returns false and
  // `*tensor_shape` is left in an intermediate state.  Otherwise
  // returns true.
  bool AsTensorShape(TensorShape* tensor_shape) const;

  /// For error messages.
  string DebugString() const;
  static string DebugString(const TensorShapeProto& proto);

  /// \brief Returns a `PartialTensorShape` whose dimensions are
  /// `dims[0]`, `dims[1]`, ..., `dims[n-1]`.  Values of -1 are
  /// considered "unknown".
  static Status MakePartialShape(const int32* dims, int n,
                                 PartialTensorShape* out);
  static Status MakePartialShape(const int64* dims, int n,
                                 PartialTensorShape* out);

 private:
  bool is_unknown_;
  gtl::InlinedVector<int64, 4> dim_sizes_;
};

/// \brief Static helper routines for `PartialTensorShape`. Includes a few
/// common predicates on a partially known tensor shape.
class PartialTensorShapeUtils {
 public:
  static string PartialShapeListString(
      const gtl::ArraySlice<PartialTensorShape>& shapes);

  static bool AreCompatible(const gtl::ArraySlice<PartialTensorShape>& shapes0,
                            const gtl::ArraySlice<PartialTensorShape>& shapes1);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_PARTIAL_TENSOR_SHAPE_H_
