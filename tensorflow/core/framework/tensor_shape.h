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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_H_

#include <string>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

// START_SKIP_DOXYGEN
template <class Shape>
class TensorShapeIter;
class TensorShape;
class TensorShapeProto;
class PartialTensorShape;
// END_SKIP_DOXYGEN

/// Internal representation for both TensorShape and PartialTensorShape.
class TensorShapeRep {
 public:
  ~TensorShapeRep();

  /// Copy the specified shape
  TensorShapeRep(const TensorShapeRep& b);
  void operator=(const TensorShapeRep& b);

  /// Move the specified shape.  After moving, `b` is safe for destruction and
  // can be reassigned into, but its dimensions and number of elements can be
  // nonsensical (e.g., negative dimension sizes, or number of elements not
  // properly recomputed).
  TensorShapeRep(TensorShapeRep&& b);
  void operator=(TensorShapeRep&& b);

  /// Clear a tensor shape, producing the scalar shape.
  void Clear();

  // Maximum number of dimensions in a tensor.
  // It's 254 because 255 = kUnknownRank is used to represent unknown rank.
  static constexpr int MaxDimensions() { return 254; }

  /// \brief Returns the number of elements in the tensor.
  ///
  /// We use `int64` and not `size_t` to be compatible with `Eigen::Tensor`
  /// which uses `ptrdiff_t`.  For PartialTensorShape, -1 means not fully
  /// defined.
  int64_t num_elements() const { return num_elements_; }

  /// For error messages.
  std::string DebugString() const;
  static std::string DebugString(const TensorShapeProto& proto);

 protected:
  // Constructable only via TensorShapeBase
  TensorShapeRep() = default;

  void ClearAllButDataType();

  // We use 16 bytes to represent a TensorShape.  Because we need to
  // be able to support full 64-bit dimension sizes and an arbitrary
  // number of dimensions for a Tensor, but most tensor dimensions are
  // significantly smaller than 64 bits and most tensors are 1, 2, or 3
  // dimensions, we have several representations.
  // Rep16: Supports up to 6 dimensions where each dimension is < 2^16 - 1
  // Rep32: Supports up to 3 dimensions where each dimension is < 2^32 - 1
  // Rep64: Supports arbitrary dimensionality, 64-bit dimensions using
  //        an out of line vector.
  // For PartialTensorShape, a dimension of static_cast<uint??>(-1) is unknown.
  // This value is not allowed in TensorShape either for format compatibility.
  struct Rep16 {
    uint16 dims_[6];
  };
  struct Rep32 {
    uint32 dims_[3];
  };
  struct Rep64 {
    absl::InlinedVector<int64_t, 4UL>* dims_;
  };

  // We use the max value of uint16 or uint32 to represent unknown shapes, so
  // the maximum representable valid shape in these representations is one less.
  static constexpr int64_t kMaxRep16 = std::numeric_limits<uint16>::max() - 1;
  static constexpr int64_t kMaxRep32 = std::numeric_limits<uint32>::max() - 1;
  static constexpr uint16 kUnknownRep16 = std::numeric_limits<uint16>::max();
  static constexpr uint32 kUnknownRep32 = std::numeric_limits<uint32>::max();

  Rep16* as16() { return reinterpret_cast<Rep16*>(buf()); }
  Rep32* as32() { return reinterpret_cast<Rep32*>(buf()); }
  Rep64* as64() { return reinterpret_cast<Rep64*>(buf()); }

  const Rep16* as16() const { return reinterpret_cast<const Rep16*>(buf()); }
  const Rep32* as32() const { return reinterpret_cast<const Rep32*>(buf()); }
  const Rep64* as64() const { return reinterpret_cast<const Rep64*>(buf()); }

  enum RepTag { REP16 = 0, REP32 = 1, REP_OUT_OF_LINE = 2 };

  // Since we have a convenient extra byte available, we allow the
  // Tensor class to store an 8-bit value in this extra storage.  This
  // allows it to store the Tensor's datatype enum value here and avoid
  // an extra word of storage.
  friend class Tensor;
  friend class TensorShapeTestHelper;
  DataType data_type() const { return static_cast<DataType>(buf()[13]); }
  void set_data_type(DataType dt) {
    // We only have 8 bits available to store DataType, so make sure it fits
    DCHECK_LT(static_cast<uint32>(dt), 256u);
    buf()[13] = static_cast<uint8>(dt);
  }

  // We store the number of dimensions in byte 14, and the RepTag in byte 15.
  // Bytes [0..13] vary depending on the representation.
  // A value of 255 indicates unknown rank in the PartialTensorShape case.
  static constexpr uint8 kUnknownRank = 255;
  uint8 ndims_byte() const { return buf()[14]; }
  void set_ndims_byte(uint8 nd) { buf()[14] = nd; }

  RepTag tag() const { return static_cast<RepTag>(buf()[15]); }
  void set_tag(RepTag tag) { buf()[15] = static_cast<uint8>(tag); }

  void set_num_elements(int64_t n) { num_elements_ = n; }

 private:
  void DestructorOutOfLine();
  void SlowCopyFrom(const TensorShapeRep& b);

  uint8* buf() { return &u_.buf[0]; }
  const uint8* buf() const { return &u_.buf[0]; }

  union {
    uint8 buf[16];
    // Force data to be aligned enough for a pointer.
    Rep64* unused_aligner;
  } u_;
  int64_t num_elements_;
};

/// Base class for TensorShape and PartialTensorShape.
/// The class is templatized by either TensorShape or PartialTensorShape to
/// allow skipping known/unknown checks in the TensorShape case, but the
/// representation is shared exactly for fast conversion.
template <class Shape>
class TensorShapeBase : public TensorShapeRep {
 public:
  /// \brief Construct a `TensorShapeBase` from the provided sizes.
  /// REQUIRES: `dim_sizes[i] >= 0` (or >= -1 for PartialTensorShape)
  explicit TensorShapeBase(absl::Span<const int64_t> dim_sizes);
  TensorShapeBase(std::initializer_list<int64_t> dim_sizes)
      : TensorShapeBase(absl::Span<const int64_t>(dim_sizes)) {}

  /// Construct an empty TensorShape, or an unknown rank PartialTensorShape
  TensorShapeBase();

  // Cannot be made explicit because we rely on conversion between proto and
  // `TensorShapeBase` throughtout the codebase (needs bigger cleanup)
  TensorShapeBase(const TensorShapeProto& proto);

  // These factory methods should be used instead of the constructors that take
  // an array of sizes if calling code cannot validate that the sizes specify a
  // valid `TensorShape`.
  // The value in `*out` is valid iff the returned value is `Status::OK`.
  static absl::Status BuildTensorShapeBase(absl::Span<const int64_t> dim_sizes,
                                           TensorShapeBase* out);
  static absl::Status BuildTensorShapeBase(
      std::initializer_list<int64_t> dim_sizes, TensorShapeBase* out) {
    return BuildTensorShapeBase(absl::Span<const int64_t>(dim_sizes), out);
  }
  static absl::Status BuildTensorShapeBase(const TensorShapeProto& proto,
                                           TensorShapeBase* out);

  /// Returns `true` iff `proto` is a valid tensor shape.
  // For TensorShape, the proto shape must be fully defined.
  static bool IsValid(const TensorShapeProto& proto);

  /// Returns `OK` iff `proto` is a valid tensor shape, and a descriptive error
  /// status otherwise.
  static absl::Status IsValidShape(const TensorShapeProto& proto);

  /// Returns `true` iff this is a valid tensor shape.
  bool IsValid();

  /// \brief Add a dimension to the end ("inner-most").
  /// REQUIRES: `size >= 0`
  void AddDim(int64_t size);

  /// Same as `AddDim` but returns a `Status`.
  /// Use if unsure is `size >= 0`, to prevent `CHECK`-crashes.
  absl::Status AddDimWithStatus(int64_t size);

  /// Appends all the dimensions from `shape`.
  void AppendShape(const TensorShapeBase& shape);

  /// Same as `RemoveDim` but returns a `Status`.
  /// Use if you cannot validate all invariants, to prevent `CHECK`-fail.
  absl::Status AppendShapeWithStatus(const TensorShapeBase& shape);

  /// \brief Insert a dimension somewhere in the `TensorShape`.
  /// REQUIRES: `0 <= d <= dims()`
  /// REQUIRES: `size >= 0`
  void InsertDim(int d, int64_t size);

  /// Same as `InsertDim` but returns a `Status`.
  /// Use if unsure if requirements in `InsertDim` are satistified, to prevent
  /// `CHECK`-fail crashes.
  absl::Status InsertDimWithStatus(int d, int64_t size);

  /// \brief Modifies the size of the dimension `d` to be `size`
  /// REQUIRES: `0 <= d < dims()`
  /// REQUIRES: `size >= 0`
  void set_dim(int d, int64_t size);

  /// Same as `set_dim` but returns a `Status`.
  /// Use if unsure if requirements in `set_dim` are satistified, to prevent
  /// `CHECK`-fail crashes.
  absl::Status SetDimWithStatus(int d, int64_t size);

  /// \brief Removes dimension `d` from the `TensorShape`.
  /// REQUIRES: `0 <= d < dims()`
  void RemoveDim(int d) {
    CHECK_GE(d, 0);
    RemoveDimRange(d, d + 1);
  }

  /// Same as `RemoveDim` but returns a `Status`.
  /// Use if unsure is `0 <= d < dims()`, to prevent `CHECK`-crashes.
  absl::Status RemoveDimWithStatus(int64_t d) {
    if (TF_PREDICT_FALSE(d < 0)) {
      return errors::Internal(
          "Expected dimension index to be non-negative, got ", d);
    }
    return RemoveDimRangeWithStatus(d, d + 1);
  }

  /// \brief Removes last `n` dimensions from the `TensorShape`.
  /// REQUIRES: `0 <= n <= dims()`
  void RemoveLastDims(int n) {
    CHECK_LE(n, dims());
    RemoveDimRange(dims() - n, dims());
  }

  /// Same as `RemoveLastDims` but returns a `Status`.
  /// Use if unsure is `0 <= n <= dims()`, to prevent `CHECK`-crashes.
  absl::Status RemoveLastDimsWithStatus(int64_t n) {
    if (TF_PREDICT_FALSE(n > dims())) {
      return errors::Internal("Expected dimension index to be at most ", dims(),
                              " got ", n);
    }
    return RemoveDimRangeWithStatus(dims() - n, dims());
  }

  /// \brief Removes the dimensions in range `[begin:end)` from `TensorShape`.
  /// Negative values of `end` are interpreted as `dims() + end + 1` (as in
  /// Python). The same is true for negative values of `begin`.
  /// REQUIRES: `-(dims()+1) <= begin <= dims()`
  /// REQUIRES: `-(dims()+1) <= end <= dims()`
  void RemoveDimRange(int begin, int end);

  /// Same as `RemoveDimRange` but returns a `Status`.
  /// Use if unsure if requirements in `RemoveDimRange` are satistified, to
  /// prevent `CHECK`-fail crashes.
  absl::Status RemoveDimRangeWithStatus(int begin, int end);

  /// Return whether the rank is unknown
  bool unknown_rank() const {
    return kIsPartial && ndims_byte() == kUnknownRank;
  }

  /// Return the number of dimensions in the tensor.
  /// Can be -1 meaning unknown rank for PartialTensorShape.
  int dims() const {
    uint8 dims = ndims_byte();
    return kIsPartial && dims == kUnknownRank ? -1 : dims;
  }

  /// \brief Returns the number of elements in dimension `d`.
  /// REQUIRES: `0 <= d < dims()`
  // TODO(touts): Rename to `dimension()` to match
  // `Eigen::Tensor::dimension()`?
  int64_t dim_size(int d) const;

  /// Returns sizes of all dimensions.
  // Returns an empty list for unknown rank PartialTensorShape.
  absl::InlinedVector<int64_t, 4UL> dim_sizes() const;

  /// Return true iff the rank and all of the dimensions are well defined
  // TODO(irving): Rename to is_fully_defined now that it's fast.
  bool IsFullyDefined() const { return !kIsPartial || num_elements() != -1; }

  /// Fill `*proto` from `*this`.
  void AsProto(TensorShapeProto* proto) const;
  TensorShapeProto AsProto() const;

  /// For iterating through the dimensions.
  TensorShapeIter<Shape> begin() const;
  TensorShapeIter<Shape> end() const;

 protected:
  // Optimized constructor for a shape representing an empty vector.
  //
  // This constructor is provided to optimize the default constructor for
  // `Tensor`.
  explicit TensorShapeBase(DataType dt);

 private:
  absl::Status RecomputeNumElements();
  absl::Status InitDims(absl::Span<const int64_t> dim_sizes);

  // True for PartialTensorShape, false for TensorShape
  static constexpr bool kIsPartial =
      std::is_same<Shape, PartialTensorShape>::value;
  static_assert(kIsPartial || std::is_same<Shape, TensorShape>::value,
                "Shape is neither TensorShape nor PartialTensorShape");

  // Used by AddDim and MakeShapeHelper.  Does no error checking.
  void UnsafeAddDim(int64_t size, int64_t new_num_elements);

  // For use by TensorShapeUtils::MakeShape
  template <class T, class S>
  friend absl::Status MakeShapeHelper(const T*, int64_t, S*);
};

/// Outputs `TensorShapeBase` to `std::ostream`.
template <typename Shape>
std::ostream& operator<<(std::ostream& os, const TensorShapeBase<Shape>& tsb) {
  return os << tsb.DebugString();
}

/// Represents the shape of a Tensor.
///
/// A tensor's shape is denoted by its number of dimensions and a size for each
/// dimension.  For example, a Tensor represented by a 3 x 4 matrix would have
/// a shape of 2-D, [3,4].
///
/// If you know the exact shape of your Tensor when you create the TensorShape
/// object, you can specify it then, or you can create a TensorShape with
/// zero dimensions and one element, and call AddDim() to add dimensions later.
class TensorShape : public TensorShapeBase<TensorShape> {
 public:
  using TensorShapeBase<TensorShape>::TensorShapeBase;

  // These factory methods should be used instead of the constructors that take
  // an array of sizes if calling code cannot validate that the sizes specify a
  // valid `TensorShape`.
  // The value in `*out` is valid iff the returned value is `Status::OK`.
  static absl::Status BuildTensorShape(absl::Span<const int64_t> dim_sizes,
                                       TensorShape* out) {
    return BuildTensorShapeBase(dim_sizes, out);
  }
  static absl::Status BuildTensorShape(std::initializer_list<int64_t> dim_sizes,
                                       TensorShape* out) {
    return BuildTensorShape(absl::Span<const int64_t>(dim_sizes), out);
  }
  static absl::Status BuildTensorShape(const TensorShapeProto& proto,
                                       TensorShape* out) {
    return BuildTensorShapeBase(proto, out);
  }

  static absl::StatusOr<TensorShape> BuildTensorShape(
      const TensorShapeProto& proto) {
    TensorShape out;
    TF_RETURN_IF_ERROR(BuildTensorShape(proto, &out));
    return out;
  }

  /// Allow a TensorShape to be used as a PartialTensorShape without copying
  operator const PartialTensorShape&() const;  // NOLINT(runtime/explicit)

  /// Returns true if `*this` and `b` have the same sizes. Ignores
  /// dimension names.
  bool IsSameSize(const TensorShape& b) const;

  /// Fill `*dsizes` from `*this`.
  /// Notice: Using IndexType=int32 in combination with To32Bit() can
  /// significantly improve performance on GPU.
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizes() const;

  // Same as `AsEigenDSizes()` but returns a `Status` instead.
  // Use this method to surface error to user instead of crashing if `NDMIS` is
  // not equal to `dims()`.
  // Caller must take ownership of `out`.
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  absl::Status AsEigenDSizesWithStatus(
      Eigen::DSizes<IndexType, NDIMS>* out) const;

  /// Same as `AsEigenDSizes()` but allows for `NDIMS > dims()` -- in
  /// which case we pad the rest of the sizes with 1.
  /// Notice: Using IndexType=int32 in combination with To32Bit() can
  /// significantly improve performance on GPU.
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesWithPadding() const;

  // Same as `AsEigenDSizesWithPadding()` but returns a `Status` instead.
  // Use this method to surface error to user instead of crashing if `NDMIS` is
  // not equal to `dims()`.
  // Caller must take ownership of `out`.
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  absl::Status AsEigenDSizesWithPaddingWithStatus(
      Eigen::DSizes<IndexType, NDIMS>* out) const;

 private:
  // These CHECK fail to ease debugging.
  // REQUIRES: dims() == NDIMS
  void CheckDimsEqual(int NDIMS) const;
  // REQUIRES: dims() <= NDIMS
  void CheckDimsAtMost(int NDIMS) const;

  // Fill output from `*this`.
  // Helper method for common code between `AsEigenDSize()` and
  // `AsEigenDSizeWithStatus()`.
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesCopy() const;

  // Fill output from `*this`.
  // Helper method for common code between `AsEigenDSizesWithPadding()` and
  // `AsEigenDSizeWithPaddingWithStatus()`.
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesCopyAndPad() const;

  // For access to TensorShapeBase(DataType).
  friend class Tensor;
};

inline bool operator==(const TensorShape& a, const TensorShape& b) {
  return a.IsSameSize(b);
}
inline bool operator!=(const TensorShape& a, const TensorShape& b) {
  return !(a == b);
}

/// Outputs `TensorShapeBase` to `std::ostream`.
inline std::ostream& operator<<(std::ostream& os, const TensorShape& ts) {
  return os << ts.DebugString();
}

/// Represents the value of one dimension in a TensorShape.
struct TensorShapeDim {
  explicit TensorShapeDim(int64_t s) : size(s) {}
  int64_t size;
};

// START_SKIP_DOXYGEN
template <class Shape>
class TensorShapeIter {
 public:
  TensorShapeIter(const Shape* shape, int d) : shape_(shape), d_(d) {}
  bool operator==(const TensorShapeIter& rhs) {
    DCHECK(shape_ == rhs.shape_);
    return d_ == rhs.d_;
  }
  bool operator!=(const TensorShapeIter& rhs) {
    DCHECK(shape_ == rhs.shape_);
    return d_ != rhs.d_;
  }
  void operator++() { ++d_; }
  TensorShapeDim operator*() { return TensorShapeDim(shape_->dim_size(d_)); }

 private:
  const Shape* shape_;
  int d_;
};
// END_SKIP_DOXYGEN

/// \brief Static helper routines for `TensorShape`. Includes a few common
/// predicates on a tensor shape.
class TensorShapeUtils {
 public:
  static bool IsScalar(const TensorShape& shape) { return shape.dims() == 0; }

  static bool IsVector(const TensorShape& shape) { return shape.dims() == 1; }

  static bool IsVectorOrHigher(const TensorShape& shape) {
    return shape.dims() >= 1;
  }

  static bool IsMatrix(const TensorShape& shape) { return shape.dims() == 2; }

  static bool IsSquareMatrix(const TensorShape& shape) {
    return shape.dims() == 2 && shape.dim_size(0) == shape.dim_size(1);
  }

  static bool IsMatrixOrHigher(const TensorShape& shape) {
    return shape.dims() >= 2;
  }

  /// \brief Returns a `TensorShape` whose dimensions are
  /// `dims[0]`, `dims[1]`, ..., `dims[n-1]`.
  static absl::Status MakeShape(const int32* dims, int64_t n, TensorShape* out);
  static absl::Status MakeShape(const int64_t* dims, int64_t n,
                                TensorShape* out);
  static absl::Status MakeShape(absl::Span<const int32> shape,
                                TensorShape* out);
  static absl::Status MakeShape(absl::Span<const int64_t> shape,
                                TensorShape* out);
  static absl::Status MakeShape(const int32* dims, int64_t n,
                                PartialTensorShape* out);
  static absl::Status MakeShape(const int64_t* dims, int64_t n,
                                PartialTensorShape* out);
  static absl::Status MakeShape(absl::Span<const int32> shape,
                                PartialTensorShape* out);
  static absl::Status MakeShape(absl::Span<const int64_t> shape,
                                PartialTensorShape* out);

  static std::string ShapeListString(
      const absl::Span<const TensorShape>& shapes);

  /// \brief Returns true iff `shape` starts with `prefix`.
  static bool StartsWith(const TensorShape& shape, const TensorShape& prefix);

  /// \brief Returns true iff `shape` ends with `suffix`.
  static bool EndsWith(const TensorShape& shape, const TensorShape& suffix);

  /// \brief Returns the product of values in an int64 array,
  /// or a failing Status if the array represents a value larger than
  /// a `TensorShape` can hold.
  static absl::Status NumElements(absl::Span<const int64_t> shape,
                                  int64_t* num_elements);
};

/// Manages the partially known dimensions of a Tensor and their sizes.
class PartialTensorShape : public TensorShapeBase<PartialTensorShape> {
 public:
  PartialTensorShape() {}
  using TensorShapeBase<PartialTensorShape>::TensorShapeBase;

  // These factory methods should be used instead of the constructors that take
  // an array of sizes if calling code cannot validate that the sizes specify a
  // valid `PartialTensorShape`.
  // The value in `*out` is valid iff the returned value is `Status::OK`.
  static absl::Status BuildPartialTensorShape(
      absl::Span<const int64_t> dim_sizes, PartialTensorShape* out) {
    return BuildTensorShapeBase(dim_sizes, out);
  }
  static absl::Status BuildPartialTensorShape(
      std::initializer_list<int64_t> dim_sizes, PartialTensorShape* out) {
    return BuildPartialTensorShape(absl::Span<const int64_t>(dim_sizes), out);
  }
  static absl::Status BuildPartialTensorShape(const TensorShapeProto& proto,
                                              PartialTensorShape* out) {
    return BuildTensorShapeBase(proto, out);
  }

  static absl::StatusOr<PartialTensorShape> BuildPartialTensorShape(
      const TensorShapeProto& proto) {
    PartialTensorShape out;
    TF_RETURN_IF_ERROR(BuildTensorShapeBase(proto, &out));
    return out;
  }

  /// Add a dimension to the end ("inner-most"), returns a new
  /// PartialTensorShape.
  /// REQUIRES: `size >= -1`, where -1 means unknown
  PartialTensorShape Concatenate(int64_t size) const;

  /// Similar to `Concatenate` but returning `Status`.
  /// Use if calling code cannot validate all requirements and if `CHECK`-fails
  /// are to be avoided.
  absl::Status ConcatenateWithStatus(int64_t size,
                                     PartialTensorShape* out) const;

  /// Appends all the dimensions from `shape`.  Returns a new
  /// PartialTensorShape.
  PartialTensorShape Concatenate(const PartialTensorShape& shape) const;

  /// Similar to `Concatenate` but returning `Status`.
  /// Use if calling code cannot validate all requirements and if `CHECK`-fails
  /// are to be avoided.
  absl::Status ConcatenateWithStatus(const PartialTensorShape& shape,
                                     PartialTensorShape* out) const;

  /// Merges all the dimensions from `shape`.  Returns
  /// `InvalidArgument` error if either `shape` has a different rank
  /// or if any of the dimensions are incompatible.
  absl::Status MergeWith(const PartialTensorShape& shape,
                         PartialTensorShape* result) const;

  /// Exact equality test. Returns true iff the ranks match (i.e., both are
  /// unknown, or both are known and equal), and all dimensions are equal (i.e.,
  /// both dimensions are known, or both are known and equal). This is a
  /// stronger condition that IsCompatibleWith.
  bool IsIdenticalTo(const PartialTensorShape& shape) const;

  /// Return true iff the ranks match, and if the
  /// dimensions all either match or one is unknown.
  bool IsCompatibleWith(const PartialTensorShape& shape) const;

  // Fill `*shape` from `*this`.
  // If `*this` is not fully defined, returns false and
  // `*shape` is left in an intermediate state.  Otherwise
  // returns true.
  bool AsTensorShape(TensorShape* shape) const;

  /// \brief Returns a `PartialTensorShape` whose dimensions are
  /// `dims[0]`, `dims[1]`, ..., `dims[n-1]`.  Values of -1 are
  /// considered "unknown".
  template <class T>
  static absl::Status MakePartialShape(const T* dims, int n,
                                       PartialTensorShape* out) {
    return TensorShapeUtils::MakeShape(dims, n, out);
  }
};

inline bool operator==(const PartialTensorShape& a,
                       const PartialTensorShape& b) {
  return a.IsIdenticalTo(b);
}

/// \brief Static helper routines for `PartialTensorShape`. Includes a few
/// common predicates on a partially known tensor shape.
class PartialTensorShapeUtils {
 public:
  static std::string PartialShapeListString(
      const absl::Span<const PartialTensorShape>& shapes);

  static bool AreIdentical(const absl::Span<const PartialTensorShape>& shapes0,
                           const absl::Span<const PartialTensorShape>& shapes1);

  static bool AreCompatible(
      const absl::Span<const PartialTensorShape>& shapes0,
      const absl::Span<const PartialTensorShape>& shapes1);
};

// ----------------------------------------------------------------------------
// Template method implementation details below
// ----------------------------------------------------------------------------

template <int NDIMS, typename IndexType>
Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizesCopy() const {
  Eigen::DSizes<IndexType, NDIMS> dsizes;
  for (int d = 0; d < NDIMS; d++) {
    dsizes[d] = static_cast<IndexType>(dim_size(d));
  }
  return dsizes;
}

template <int NDIMS, typename IndexType>
Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizesCopyAndPad() const {
  static_assert(NDIMS <= TensorShape::MaxDimensions(), "Too many dimensions");
  Eigen::DSizes<IndexType, NDIMS> dsizes;
  for (int d = 0; d < dims(); d++) {
    dsizes[d] = static_cast<IndexType>(dim_size(d));
  }
  for (int d = dims(); d < NDIMS; d++) {
    dsizes[d] = 1;
  }
  return dsizes;
}

template <int NDIMS, typename IndexType>
Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizes() const {
  CheckDimsEqual(NDIMS);
  return AsEigenDSizesCopy<NDIMS, IndexType>();
}

template <int NDIMS, typename IndexType>
absl::Status TensorShape::AsEigenDSizesWithStatus(
    Eigen::DSizes<IndexType, NDIMS>* out) const {
  if (TF_PREDICT_FALSE(NDIMS != dims())) {
    return errors::Internal("Asking for tensor of ", NDIMS,
                            " dimensions from a tensor of ", dims(),
                            " dimensions");
  }
  *out = AsEigenDSizesCopy<NDIMS, IndexType>();
  return absl::OkStatus();
}

template <int NDIMS, typename IndexType>
Eigen::DSizes<IndexType, NDIMS> TensorShape::AsEigenDSizesWithPadding() const {
  CheckDimsAtMost(NDIMS);
  return AsEigenDSizesCopyAndPad<NDIMS, IndexType>();
}

template <int NDIMS, typename IndexType>
absl::Status TensorShape::AsEigenDSizesWithPaddingWithStatus(
    Eigen::DSizes<IndexType, NDIMS>* out) const {
  if (TF_PREDICT_FALSE(NDIMS < dims())) {
    return errors::Internal("Asking for tensor of at most ", NDIMS,
                            " dimensions from a tensor of ", dims(),
                            " dimensions");
  }
  *out = AsEigenDSizesCopyAndPad<NDIMS, IndexType>();
  return absl::OkStatus();
}

// ----------------------------------------------------------------------------
// Inlining of some performance critical routines
// ----------------------------------------------------------------------------

inline TensorShapeRep::TensorShapeRep(const TensorShapeRep& b) {
  num_elements_ = b.num_elements_;
  if (b.tag() != REP_OUT_OF_LINE) {
    memcpy(buf(), b.buf(), sizeof(u_.buf));
    // memcpy above Implicitly does:
    //   set_ndims_byte(b.ndims_byte());
    //   set_tag(b.tag());
  } else {
    set_tag(REP16);  // So that SlowCopyFrom does not try to deallocate
    SlowCopyFrom(b);
  }
}

inline TensorShapeRep::TensorShapeRep(TensorShapeRep&& b) {
  num_elements_ = b.num_elements_;
  memcpy(buf(), b.buf(), sizeof(u_.buf));
  // memcpy above Implicitly does:
  //   set_ndims_byte(b.ndims_byte());
  //   set_tag(b.tag());
  b.set_tag(REP16);  // other shape no longer owns out-of-line data, if any.
}

inline TensorShapeRep::~TensorShapeRep() {
  if (tag() == REP_OUT_OF_LINE) {
    DestructorOutOfLine();
  }
}

inline void TensorShapeRep::operator=(const TensorShapeRep& b) {
  num_elements_ = b.num_elements_;
  if (tag() != REP_OUT_OF_LINE && b.tag() != REP_OUT_OF_LINE) {
    memcpy(buf(), b.buf(), sizeof(u_.buf));
    // memcpy above implicitly also does:
    //   set_tag(b.tag());
    //   set_ndims_byte(b.ndims_byte());
  } else {
    SlowCopyFrom(b);
  }
}

inline void TensorShapeRep::operator=(TensorShapeRep&& b) {
  if (tag() == REP_OUT_OF_LINE) {
    DestructorOutOfLine();
  }
  num_elements_ = b.num_elements_;
  memcpy(buf(), b.buf(), sizeof(u_.buf));
  // memcpy above Implicitly does:
  //   set_ndims_byte(b.ndims_byte());
  //   set_tag(b.tag());
  b.set_tag(REP16);  // other shape no longer owns out-of-line data, if any.
}

inline TensorShape::operator const PartialTensorShape&() const {
  // Downcast to the shared representation and upcast to PartialTensorShape
  const TensorShapeRep* rep = this;
  return *static_cast<const PartialTensorShape*>(rep);
}

template <class Shape>
inline TensorShapeBase<Shape>::TensorShapeBase(DataType dt) {
  set_tag(REP16);
  set_data_type(dt);

  // Optimized implementation of InitDims() where the shape is statically known
  // to be {0}.
  set_ndims_byte(1);
  uint16* dst = as16()->dims_;
  *dst = 0;
  set_num_elements(0);
}

// Declare explicit instantiations in .cc file
extern template class TensorShapeBase<TensorShape>;
extern template class TensorShapeBase<PartialTensorShape>;

// A convenient struct to represent a (DataType, PartialTensorShape) pair. It's
// often used in shape inference.
struct DtypeAndPartialTensorShape {
  DataType dtype;
  PartialTensorShape shape;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_H_
