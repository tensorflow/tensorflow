#ifndef TENSORFLOW_PUBLIC_TENSOR_H_
#define TENSORFLOW_PUBLIC_TENSOR_H_

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class TensorBuffer;  // Forward declaration.
class TensorCApi;

/// Represents an n-dimensional array of values.
class Tensor {
 public:
  /// Default Tensor constructor. Creates a 1-dimension, 0-element float tensor.
  Tensor();

  /// \brief Creates a Tensor of the given datatype and shape.
  ///
  /// The underlying buffer is allocated using a CPUAllocator.
  Tensor(DataType type, const TensorShape& shape);

  /// \brief Creates a tensor with the input datatype and shape, using the
  /// allocator 'a' to allocate the underlying buffer.
  ///
  /// 'a' must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape);

  /// Creates an uninitialized Tensor of the given data type.
  explicit Tensor(DataType type);

  Tensor(const Tensor& other);  /// Copy constructor.

  ~Tensor();

  /// Returns the data type.
  DataType dtype() const { return type_; }

  /// Returns the shape of the tensor.
  const TensorShape& shape() const { return shape_; }

  /// \brief Convenience accessor for the tensor shape.
  ///
  /// For all shape accessors, see comments for relevant methods of
  /// TensorShape in tensor_shape.h.
  int dims() const { return shape().dims(); }

  /// Convenience accessor for the tensor shape.
  int64 dim_size(int d) const { return shape().dim_size(d); }

  /// Convenience accessor for the tensor shape.
  int64 NumElements() const { return shape().num_elements(); }

  bool IsSameSize(const Tensor& b) const {
    return shape().IsSameSize(b.shape());
  }

  /// Has this Tensor been initialized?
  bool IsInitialized() const;

  /// Returns the estimated memory usage of this tensor.
  size_t TotalBytes() const;

  /// Assign operator. This tensor shares other's underlying storage.
  Tensor& operator=(const Tensor& other) {
    CopyFromInternal(other, other.shape());
    return *this;
  }

  /// \brief Copy the other tensor into this tensor and reshape it.
  ///
  /// This tensor shares other's underlying storage. Returns
  /// true iff other.shape() has the same number of elements of the
  /// given "shape".
  bool CopyFrom(const Tensor& other,
                const TensorShape& shape) TF_MUST_USE_RESULT {
    if (other.NumElements() != shape.num_elements()) return false;
    CopyFromInternal(other, shape);
    return true;
  }

  /// \brief Slice this tensor along the 1st dimension.

  /// I.e., the returned
  /// tensor satisifies returned[i, ...] == this[dim0_start + i, ...].
  /// The returned tensor shares the underlying tensor buffer with this
  /// tensor.
  ///
  /// NOTE: The returned tensor may not satisfies the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., flat(), tensor()).
  ///
  /// REQUIRES: dims() >= 1
  /// REQUIRES: 0 <= dim0_start <= dim0_limit <= dim_size(0)
  Tensor Slice(int64 dim0_start, int64 dim0_limit) const;

  /// \brief Parse "other' and construct the tensor. 

  /// Returns true iff the
  /// parsing succeeds. If the parsing fails, the state of "*this" is
  /// unchanged.
  bool FromProto(const TensorProto& other) TF_MUST_USE_RESULT;
  bool FromProto(Allocator* a, const TensorProto& other) TF_MUST_USE_RESULT;

  /// \brief Fills in "proto" with "*this" tensor's content.
  ///
  /// AsProtoField() fills in the repeated field for proto.dtype(), while
  /// AsProtoTensorContent() encodes the content in proto.tensor_content() in a
  /// compact form.
  void AsProtoField(TensorProto* proto) const;
  void AsProtoTensorContent(TensorProto* proto) const;

  /// \brief Return the Tensor data as an Eigen::Tensor with the type and
  /// sizes of this Tensor.
  ///
  /// Use these methods when you know the data type and the number of
  /// dimensions of the Tensor and you want an Eigen::Tensor
  /// automatically sized to the Tensor sizes. The implementation check
  /// fails if either type or sizes mismatch.
  ///
  /// Example:
  ///
  /// ```c++
  ///
  ///     typedef float T;
  ///     Tensor my_mat(...built with Shape{rows: 3, cols: 5}...);
  ///     auto mat = my_mat.matrix<T>();    // 2D Eigen::Tensor, 3 x 5.
  ///     auto mat = my_mat.tensor<T, 2>(); // 2D Eigen::Tensor, 3 x 5.
  ///     auto vec = my_mat.vec<T>();       // CHECK fails as my_mat is 2D.
  ///     auto vec = my_mat.tensor<T, 3>(); // CHECK fails as my_mat is 2D.
  ///     auto mat = my_mat.matrix<int32>();// CHECK fails as type mismatch.
  ///
  /// ```
  template <typename T>
  typename TTypes<T>::Vec vec() {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix() {
    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor();

  /// \brief Return the Tensor data as an Eigen::Tensor of the data type and a
  /// specified shape.
  ///
  /// These methods allow you to access the data with the dimensions
  /// and sizes of your choice.  You do not need to know the number of
  /// dimensions of the Tensor to call them.  However, they CHECK that
  /// the type matches and the dimensions requested creates an
  /// Eigen::Tensor with the same number of elements as the Tensor.
  ///
  /// Example:
  ///
  /// ```c++
  ///
  ///     typedef float T;
  ///     Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
  ///     // 1D Eigen::Tensor, size 60:
  ///     auto flat = my_ten.flat<T>();
  ///     // 2D Eigen::Tensor 12 x 5:
  ///     auto inner = my_ten.flat_inner_dims<T>();
  ///     // 2D Eigen::Tensor 4 x 15:
  ///     auto outer = my_ten.shaped<T, 2>({4, 15});
  ///     // CHECK fails, bad num elements:
  ///     auto outer = my_ten.shaped<T, 2>({4, 8});
  ///     // 3D Eigen::Tensor 6 x 5 x 2:
  ///     auto weird = my_ten.shaped<T, 3>({6, 5, 2});
  ///     // CHECK fails, type mismatch:
  ///     auto bad   = my_ten.flat<int32>();
  ///
  /// ```
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedFlat unaligned_flat() {
    return unaligned_shaped<T, 1>({NumElements()});
  }

  /// Returns the data as an Eigen::Tensor with 2 dimensions, collapsing all
  /// Tensor dimensions but the last one into the first dimension of the result.
  template <typename T>
  typename TTypes<T>::Matrix flat_inner_dims() {
    int64 last_size = dims() > 0 ? dim_size(dims() - 1) : 1;
    if (last_size == 0) {
      DCHECK_EQ(NumElements(), 0);
      // Return something empty, avoiding divide by 0
      return shaped<T, 2>({0, 0});
    } else {
      return shaped<T, 2>({NumElements() / last_size, last_size});
    }
  }

  /// Returns the data as an Eigen::Tensor with 2 dimensions, collapsing all
  /// Tensor dimensions but the first one into the last dimension of the result.
  template <typename T>
  typename TTypes<T>::Matrix flat_outer_dims() {
    int64 first_size = dims() > 0 ? dim_size(0) : 1;
    if (first_size == 0) {
      DCHECK_EQ(NumElements(), 0);
      // Return something empty, avoiding divide by 0
      return shaped<T, 2>({0, 0});
    } else {
      return shaped<T, 2>({first_size, NumElements() / first_size});
    }
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64> new_sizes);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedTensor unaligned_shaped(
      gtl::ArraySlice<int64> new_sizes);

  /// \brief Return the Tensor data as a Tensor Map of fixed size 1:
  /// TensorMap<TensorFixedSize<T, 1>>.

  /// Using scalar() allows the compiler to
  /// perform optimizations as the size of the tensor is known at compile time.
  template <typename T>
  typename TTypes<T>::Scalar scalar();

  /// Const versions of all the methods above.
  template <typename T>
  typename TTypes<T>::ConstVec vec() const {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor tensor() const;

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedConstFlat unaligned_flat() const {
    return unaligned_shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix flat_inner_dims() const {
    int64 last_size = dims() > 0 ? dim_size(dims() - 1) : 1;
    if (last_size == 0) {
      DCHECK_EQ(NumElements(), 0);
      // Return something empty, avoiding divide by 0
      return shaped<T, 2>({0, 0});
    } else {
      return shaped<T, 2>({NumElements() / last_size, last_size});
    }
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix flat_outer_dims() const {
    int64 first_size = dims() > 0 ? dim_size(0) : 1;
    if (first_size == 0) {
      DCHECK_EQ(NumElements(), 0);
      // Return something empty, avoiding divide by 0
      return shaped<T, 2>({0, 0});
    } else {
      return shaped<T, 2>({first_size, NumElements() / first_size});
    }
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor shaped(
      gtl::ArraySlice<int64> new_sizes) const;
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedConstTensor unaligned_shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  template <typename T>
  typename TTypes<T>::ConstScalar scalar() const;

  /// Render the first max_entries values in *this into a string.
  string SummarizeValue(int64 max_entries) const;

  /// A human-readable summary of the Tensor suitable for debugging.
  string DebugString() const;

  /// Fill in the TensorDescription proto with metadata about the
  /// Tensor that is useful for monitoring and debugging.
  void FillDescription(TensorDescription* description) const;

  /// \brief Returns a StringPiece mapping the current tensor's buffer.
  ///
  /// The returned StringPiece may point to memory location on devices
  /// that the CPU cannot address directly.
  ///
  /// NOTE: The underlying Tensor buffer is refcounted, so the lifetime
  /// of the contents mapped by the StringPiece matches the lifetime of
  /// the buffer; callers should arrange to make sure the buffer does
  /// not get destroyed while the StringPiece is still used.
  ///
  /// REQUIRES: DataTypeCanUseMemcpy(dtype()).
  StringPiece tensor_data() const;

 private:
  DataType type_;
  TensorShape shape_;
  TensorBuffer* buf_;

  friend class DMAHelper;
  friend class TensorCApi;
  friend class VariableOp;            // For access to set_shape
  friend class AutoReloadVariableOp;  // For access to set_shape

  // Creates a tensor with the input datatype, shape and buf.
  //
  // Acquires a ref on buf that belongs to this Tensor.
  Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

  bool CanUseDMA() const;

  // Only needed by variable op to set the shape of an uninitialized
  // Tensor.
  // TODO: Remove this when we have a better story for detecting
  // uninitialized tensors.
  void set_shape(const TensorShape& shape) { shape_ = shape; }

  void CopyFromInternal(const Tensor& other, const TensorShape& shape);

  template <typename T>
  T* base() const;
};

// Implementation details

// Interface to access the raw ref-counted data buffer.
class TensorBuffer : public core::RefCounted {
 public:
  ~TensorBuffer() override {}

  // data() points to a memory region of size() bytes.
  virtual void* data() const = 0;
  virtual size_t size() const = 0;

  // If this TensorBuffer is sub-buffer of another TensorBuffer,
  // returns that TensorBuffer. Otherwise, returns this.
  virtual TensorBuffer* root_buffer() = 0;

  // Fill metadata about the allocation into the proto.
  virtual void FillAllocationDescription(
      AllocationDescription* proto) const = 0;

  template <typename T>
  T* base() const {
    return reinterpret_cast<T*>(data());
  }
};

inline void CheckEigenAlignment(const void* ptr) {
#if EIGEN_ALIGN == 1
  CHECK_EQ(reinterpret_cast<intptr_t>(ptr) % EIGEN_ALIGN_BYTES, 0);
#endif
}

template <typename T>
T* Tensor::base() const {
  return buf_ == nullptr ? nullptr : buf_->base<T>();
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() {
  CHECK_EQ(dtype(), DataTypeToEnum<T>::v());
  CheckEigenAlignment(base<T>());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::tensor() const {
  CheckEigenAlignment(base<T>());
  CHECK_EQ(dtype(), DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CheckEigenAlignment(base<T>());
  CHECK_EQ(dtype(), DataTypeToEnum<T>::v());
  CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    dims[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CHECK_EQ(dtype(), DataTypeToEnum<T>::v());
  CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    dims[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
  return typename TTypes<T, NDIMS>::UnalignedTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CheckEigenAlignment(base<T>());
  CHECK_EQ(dtype(), DataTypeToEnum<T>::v());
  CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    dims[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedConstTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CHECK_EQ(dtype(), DataTypeToEnum<T>::v());
  CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    dims[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
  return typename TTypes<T, NDIMS>::UnalignedConstTensor(base<T>(), dims);
}

template <typename T>
typename TTypes<T>::Scalar Tensor::scalar() {
  CheckEigenAlignment(base<T>());
  CHECK_EQ(1, NumElements()) << "Must have a one element tensor";
  return typename TTypes<T>::Scalar(base<T>());
}

template <typename T>
typename TTypes<T>::ConstScalar Tensor::scalar() const {
  CheckEigenAlignment(base<T>());
  CHECK_EQ(1, NumElements()) << "Must have a one element tensor";
  return typename TTypes<T>::ConstScalar(base<T>());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PUBLIC_TENSOR_H_
