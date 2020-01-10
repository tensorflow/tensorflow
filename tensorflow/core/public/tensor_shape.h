#ifndef TENSORFLOW_PUBLIC_TENSOR_SHAPE_H_
#define TENSORFLOW_PUBLIC_TENSOR_SHAPE_H_

#include <string>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class TensorShapeIter;  // Declared below

/// Manages the dimensions of a Tensor and their sizes.
class TensorShape {
 public:
  /// \brief Construct a TensorShape from the provided sizes..
  /// REQUIRES: dim_sizes[i] >= 0
  explicit TensorShape(gtl::ArraySlice<int64> dim_sizes);
  TensorShape(std::initializer_list<int64> dim_sizes)
      : TensorShape(gtl::ArraySlice<int64>(dim_sizes)) {}

  /// REQUIRES: IsValid(proto)
  explicit TensorShape(const TensorShapeProto& proto);

  /// Create a tensor shape with no dimensions and one element, which you can
  /// then call AddDim() on.
  TensorShape();

  /// Returns true iff "proto" is a valid tensor shape.
  static bool IsValid(const TensorShapeProto& proto);

  /// Clear a tensor shape
  void Clear();

  /// \brief Add a dimension to the end ("inner-most").
  /// REQUIRES: size >= 0
  void AddDim(int64 size);

  /// Appends all the dimensions from shape.
  void AppendShape(const TensorShape& shape);

  /// \brief Insert a dimension somewhere in the TensorShape.
  /// REQUIRES: "0 <= d <= dims()"
  /// REQUIRES: size >= 0
  void InsertDim(int d, int64 size);

  /// \brief Modifies the size of the dimension 'd' to be 'size'
  /// REQUIRES: "0 <= d < dims()"
  /// REQUIRES: size >= 0
  void set_dim(int d, int64 size);

  /// \brief Removes dimension 'd' from the TensorShape.
  /// REQUIRES: "0 <= d < dims()"
  void RemoveDim(int d);

  /// Return the number of dimensions in the tensor.
  int dims() const { return dim_sizes_.size(); }

  /// \brief Returns the number of elements in dimension "d".
  /// REQUIRES: "0 <= d < dims()"
  // TODO(touts): Rename to dimension() to match Eigen::Tensor::dimension()?
  int64 dim_size(int d) const {
    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return dim_sizes_[d];
  }

  /// Returns sizes of all dimensions.
  gtl::ArraySlice<int64> dim_sizes() const { return dim_sizes_; }

  /// \brief Returns the number of elements in the tensor.
  ///
  /// We use int64 and
  /// not size_t to be compatible with Eigen::Tensor which uses ptr_fi
  int64 num_elements() const { return num_elements_; }

  /// Returns true if *this and b have the same sizes. Ignores dimension names.
  bool IsSameSize(const TensorShape& b) const;
  bool operator==(const TensorShape& b) const { return IsSameSize(b); }

  /// Fill *proto from *this.
  void AsProto(TensorShapeProto* proto) const;

  /// Fill *dsizes from *this.
  template <int NDIMS>
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> AsEigenDSizes() const;

  /// Same as AsEigenDSizes() but allows for NDIMS > dims() -- in which case we
  /// pad the rest of the sizes with 1.
  template <int NDIMS>
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> AsEigenDSizesWithPadding() const;

  /// For iterating through the dimensions.
  TensorShapeIter begin() const;
  TensorShapeIter end() const;

  /// For error messages.
  string DebugString() const;
  // TODO(vrv): Remove this, this is the same as DebugString().
  string ShortDebugString() const;

 private:
  /// Recalculates the dimensions of this tensor after they are modified.
  void recompute_dims();

  // TODO(josh11b): Maybe use something from the Eigen Tensor library
  /// for the sizes.
  gtl::InlinedVector<int64, 4> dim_sizes_;

  /// total number of elements (avoids recomputing it each time).
  int64 num_elements_;
};

struct TensorShapeDim {
  explicit TensorShapeDim(int64 s) : size(s) {}
  int size;
};

class TensorShapeIter {
 public:
  TensorShapeIter(const TensorShape* shape, int d) : shape_(shape), d_(d) {}
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
  const TensorShape* shape_;
  int d_;
};

// In some places, allow shape (1,) to be treated as a scalar and shape () to be
// treated as a vector.  This flag is for temporary backwards compatibility
// only, and will be changed to strict within Google around November 15, 2015.
#if defined(PLATFORM_GOOGLE)
// TODO(irving): Become strict on November 15, 2015.
static const bool kAllowLegacyScalars = true;
#else
// For open source (outside Google), we are strict.
static const bool kAllowLegacyScalars = false;
#endif

/// \brief Static helper routines for TensorShape. Includes a few common
/// predicates on a tensor shape.
class TensorShapeUtils {
 public:
  static bool IsScalar(const TensorShape& shape) { return shape.dims() == 0; }

  static bool IsVector(const TensorShape& shape) { return shape.dims() == 1; }

  // Allow either scalars or (if allowing legacy scalars) shape (1,).
  static bool IsLegacyScalar(const TensorShape& shape) {
    return shape.dims() == 0 ||
           (kAllowLegacyScalars && shape.dims() == 1 && shape.dim_size(0) == 1);
  }

  // Allow rank 1 or (if allowing legacy scalars) rank 0.
  static bool IsLegacyVector(const TensorShape& shape) {
    return shape.dims() == 1 || (kAllowLegacyScalars && shape.dims() == 0);
  }

  static bool IsVectorOrHigher(const TensorShape& shape) {
    return shape.dims() >= 1;
  }

  static bool IsMatrix(const TensorShape& shape) { return shape.dims() == 2; }

  static bool IsMatrixOrHigher(const TensorShape& shape) {
    return shape.dims() >= 2;
  }

  /// \brief Returns a TensorShape whose dimensions are dims[0], dims[1], ...,
  /// dims[n-1].
  template <typename T>
  static TensorShape MakeShape(const T* dims, int n) {
    TensorShape shape;
    for (int i = 0; i < n; ++i) shape.AddDim(dims[i]);
    return shape;
  }

  static string ShapeListString(const gtl::ArraySlice<TensorShape>& shapes) {
    string result = "[";
    bool first = true;
    for (const TensorShape& shape : shapes) {
      strings::StrAppend(&result, (first ? "" : ", "), shape.DebugString());
      first = false;
    }
    strings::StrAppend(&result, "]");
    return result;
  }

  static bool StartsWith(const TensorShape& shape0, const TensorShape& shape1);
};

// TODO(josh11b): Add TensorStrides once we support strides
// struct TensorStrides {
//   gtl::InlinedVector<int, 4> strides_;
// };

// ----------------------------------------------------------------------------
// Template method implementation details below
// ----------------------------------------------------------------------------

template <int NDIMS>
Eigen::DSizes<Eigen::DenseIndex, NDIMS> TensorShape::AsEigenDSizes() const {
  CHECK_EQ(NDIMS, dims()) << "Asking for tensor of " << NDIMS
                          << " for a tensor of " << dims() << " dimensions";
  return AsEigenDSizesWithPadding<NDIMS>();
}

template <int NDIMS>
Eigen::DSizes<Eigen::DenseIndex, NDIMS> TensorShape::AsEigenDSizesWithPadding()
    const {
  CHECK_GE(NDIMS, dims()) << "Asking for tensor of " << NDIMS
                          << " for a tensor of " << dims() << " dimensions";
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes;
  for (int d = 0; d < dims(); d++) {
    dsizes[d] = dim_size(d);
  }
  for (int d = dims(); d < NDIMS; d++) {
    dsizes[d] = 1;
  }
  return dsizes;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PUBLIC_TENSOR_SHAPE_H_
