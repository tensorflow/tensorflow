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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class TensorBuffer;  // Forward declaration.
class TensorCApi;

/// Represents an n-dimensional array of values.
class Tensor {
 public:
  /// \brief Creates a 1-dimensional, 0-element float tensor.
  ///
  /// The returned Tensor is not a scalar (shape {}), but is instead
  /// an empty one-dimensional Tensor (shape {0}, NumElements() ==
  /// 0). Since it has no elements, it does not need to be assigned a
  /// value and is initialized by default (IsInitialized() is
  /// true). If this is undesirable, consider creating a one-element
  /// scalar which does require initialization:
  ///
  /// ```c++
  ///
  ///     Tensor(DT_FLOAT, TensorShape({}))
  ///
  /// ```
  Tensor();

  /// \brief Creates a Tensor of the given `type` and `shape`.  If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  Tensor(DataType type, const TensorShape& shape);

  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` to allocate the underlying buffer. If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// `a` must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape);

  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` and the specified "allocation_attr" to
  /// allocate the underlying buffer. If the kernel and step are known
  /// allocation_attr.allocation_will_be_logged should be set to true
  /// and LogMemory::RecordTensorAllocation should be called after the
  /// tensor is constructed. Calling the Tensor constructor directly
  /// from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// `a` must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape,
         const AllocationAttributes& allocation_attr);

  /// \brief Creates an empty Tensor of the given data type.
  ///
  /// Like Tensor(), returns a 1-dimensional, 0-element Tensor with
  /// IsInitialized() returning True. See the Tensor() documentation
  /// for details.
  explicit Tensor(DataType type);

  Tensor(const Tensor& other);  /// Copy constructor.

  // Move constructor.  After this call, <other> is safely destructible and can
  // be assigned to, but other calls on it (e.g. shape manipulation) are not
  // valid.
  Tensor(Tensor&& other);

  ~Tensor();

  /// Returns the data type.
  DataType dtype() const { return shape_.data_type(); }

  /// Returns the shape of the tensor.
  const TensorShape& shape() const { return shape_; }

  /// \brief Convenience accessor for the tensor shape.
  ///
  /// For all shape accessors, see comments for relevant methods of
  /// `TensorShape` in `tensor_shape.h`.
  int dims() const { return shape().dims(); }

  /// Convenience accessor for the tensor shape.
  int64 dim_size(int d) const { return shape().dim_size(d); }

  /// Convenience accessor for the tensor shape.
  int64 NumElements() const { return shape().num_elements(); }

  bool IsSameSize(const Tensor& b) const {
    return shape().IsSameSize(b.shape());
  }

  // True iff the two tensors use the same underlying refcounted storage
  bool SharesBufferWith(const Tensor& b) const;

  /// \brief If necessary, has this Tensor been initialized?
  ///
  /// Zero-element Tensors are always considered initialized, even if they
  /// have never been assigned to and do not have any memory allocated.
  bool IsInitialized() const;

  /// Returns the estimated memory usage of this tensor.
  size_t TotalBytes() const;

  /// Returns true iff this tensor is aligned.
  bool IsAligned() const {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return true;
#else
    void* ptr = base<void>();
    return reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0;
#endif
  }

  /// Assign operator. This tensor shares other's underlying storage.
  Tensor& operator=(const Tensor& other) {
    CopyFromInternal(other, other.shape());
    return *this;
  }

  /// Move operator.  See move constructor for details.
  Tensor& operator=(Tensor&& other);

  /// \brief Copy the other tensor into this tensor and reshape it.
  ///
  /// This tensor shares other's underlying storage. Returns `true`
  /// iff `other.shape()` has the same number of elements of the given
  /// `shape`.
  bool CopyFrom(const Tensor& other,
                const TensorShape& shape) TF_MUST_USE_RESULT {
    if (other.NumElements() != shape.num_elements()) return false;
    CopyFromInternal(other, shape);
    return true;
  }

  /// \brief Slice this tensor along the 1st dimension.

  /// I.e., the returned tensor satisfies
  ///     returned[i, ...] == this[dim0_start + i, ...].
  /// The returned tensor shares the underlying tensor buffer with this
  /// tensor.
  ///
  /// NOTE: The returned tensor may not satisfies the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= dim0_start <= dim0_limit <= dim_size(0)`
  Tensor Slice(int64 dim0_start, int64 dim0_limit) const;

  /// \brief Parse `other` and construct the tensor.

  /// Returns `true` iff the parsing succeeds. If the parsing fails,
  /// the state of `*this` is unchanged.
  bool FromProto(const TensorProto& other) TF_MUST_USE_RESULT;
  bool FromProto(Allocator* a, const TensorProto& other) TF_MUST_USE_RESULT;

  /// \brief Fills in `proto` with `*this` tensor's content.
  ///
  /// `AsProtoField()` fills in the repeated field for `proto.dtype()`, while
  /// `AsProtoTensorContent()` encodes the content in `proto.tensor_content()`
  /// in a compact form.
  void AsProtoField(TensorProto* proto) const;
  void AsProtoTensorContent(TensorProto* proto) const;

  /// \brief Return the tensor data as an `Eigen::Tensor` with the type and
  /// sizes of this `Tensor`.
  ///
  /// Use these methods when you know the data type and the number of
  /// dimensions of the Tensor and you want an `Eigen::Tensor`
  /// automatically sized to the `Tensor` sizes. The implementation check
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

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// same size but a bitwise cast to the specified dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// NOTE: this is the same as `tensor()` except a bitcast is allowed.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_tensor();

  /// \brief Return the tensor data as an `Eigen::Tensor` of the data type and a
  /// specified shape.
  ///
  /// These methods allow you to access the data with the dimensions
  /// and sizes of your choice.  You do not need to know the number of
  /// dimensions of the Tensor to call them.  However, they `CHECK` that
  /// the type matches and the dimensions requested creates an
  /// `Eigen::Tensor` with the same number of elements as the tensor.
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

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all
  /// Tensor dimensions but the last NDIMS-1 into the first dimension of the
  /// result. If NDIMS > dims() then leading dimensions of size 1 will be
  /// added to make the output rank NDIMS.
  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::Tensor flat_inner_dims();

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all
  /// Tensor dimensions but the first NDIMS-1 into the last dimension of the
  /// result. If NDIMS > dims() then trailing dimensions of size 1 will be
  /// added to make the output rank NDIMS.
  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::Tensor flat_outer_dims();

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64> new_sizes);

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_shaped(
      gtl::ArraySlice<int64> new_sizes);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedTensor unaligned_shaped(
      gtl::ArraySlice<int64> new_sizes);

  /// \brief Return the Tensor data as a `TensorMap` of fixed size 1:
  /// `TensorMap<TensorFixedSize<T, 1>>`.

  /// Using `scalar()` allows the compiler to perform optimizations as
  /// the size of the tensor is known at compile time.
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

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// same size but a bitwise cast to the specified dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// NOTE: this is the same as `tensor()` except a bitcast is allowed.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_tensor() const;

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::UnalignedConstFlat unaligned_flat() const {
    return unaligned_shaped<T, 1>({NumElements()});
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedConstTensor unaligned_shaped(
      gtl::ArraySlice<int64> new_sizes) const;

  template <typename T>
  typename TTypes<T>::ConstScalar scalar() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_dims() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_outer_dims() const;

  /// Render the first `max_entries` values in `*this` into a string.
  string SummarizeValue(int64 max_entries) const;

  /// A human-readable summary of the tensor suitable for debugging.
  string DebugString() const;

  /// Fill in the `TensorDescription` proto with metadata about the
  /// tensor that is useful for monitoring and debugging.
  void FillDescription(TensorDescription* description) const;

  /// \brief Returns a `StringPiece` mapping the current tensor's buffer.
  ///
  /// The returned `StringPiece` may point to memory location on devices
  /// that the CPU cannot address directly.
  ///
  /// NOTE: The underlying tensor buffer is refcounted, so the lifetime
  /// of the contents mapped by the `StringPiece` matches the lifetime of
  /// the buffer; callers should arrange to make sure the buffer does
  /// not get destroyed while the `StringPiece` is still used.
  ///
  /// REQUIRES: `DataTypeCanUseMemcpy(dtype())`.
  StringPiece tensor_data() const;

  /// Copy the other tensor into this tensor and reshape it and reinterpret the
  /// buffer's datatype.
  ///
  /// This tensor shares other's underlying storage.
  void UnsafeCopyFromInternal(const Tensor&, DataType dtype,
                              const TensorShape&);

 private:
  void CheckType(DataType expected_dtype) const;
  void CheckTypeAndIsAligned(DataType expected_dtype) const;
  void CheckIsAlignedAndSingleElement() const;
  void set_dtype(DataType t) { shape_.set_data_type(t); }
  template <size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;

  // TODO(rmlarsen): These shouldn't hardcode '4' so that it lines up with
  // TensorShape's InlineVector.
  gtl::InlinedVector<int64, 4> ComputeFlatInnerDims(int64 num_out_dims) const;
  gtl::InlinedVector<int64, 4> ComputeFlatOuterDims(int64 num_out_dims) const;

  TensorShape shape_;
  TensorBuffer* buf_;

  friend class DMAHelper;
  friend class TensorCApi;
  friend class TensorReference;       // For access to buf_
  friend class VariableOp;            // For access to set_shape
  friend class AutoReloadVariableOp;  // For access to set_shape
  friend class TensorTestHelper;      // For access to set_shape

  // Creates a tensor with the input datatype, shape and buf.
  //
  // Acquires a ref on buf that belongs to this Tensor.
  Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

  bool CanUseDMA() const;

  // Only needed by variable op to set the shape of an uninitialized
  // Tensor.
  // TODO: Remove this when we have a better story for detecting
  // uninitialized tensors.
  void set_shape(const TensorShape& shape) {
    DataType dt = dtype();
    shape_ = shape;
    set_dtype(dt);
  }

  void CopyFromInternal(const Tensor& other, const TensorShape& shape);

  template <typename T>
  T* base() const;

  template <size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims,
      gtl::ArraySlice<int64> new_sizes) const;
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

template <typename T>
T* Tensor::base() const {
  return buf_ == nullptr ? nullptr : buf_->base<T>();
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::tensor() const {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_tensor() {
  CHECK(IsAligned());
  ;
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_tensor() const {
  CHECK(IsAligned());
  ;
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
  CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<NDIMS>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CHECK(IsAligned());
  ;
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<NDIMS>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64> new_sizes) {
  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<NDIMS>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedTensor(base<T>(), dims);
}

template <size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims,
    gtl::ArraySlice<int64> new_sizes) const {
  CHECK_EQ(NDIMS, new_sizes.size());
  int64 new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(&dims, new_sizes);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CHECK(IsAligned());
  ;
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(&dims, new_sizes);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedConstTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64> new_sizes) const {
  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(&dims, new_sizes);
  return typename TTypes<T, NDIMS>::UnalignedConstTensor(base<T>(), dims);
}

template <typename T>
typename TTypes<T>::Scalar Tensor::scalar() {
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::Scalar(base<T>());
}

template <typename T>
typename TTypes<T>::ConstScalar Tensor::scalar() const {
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::ConstScalar(base<T>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_dims() {
  return shaped<T, NDIMS>(ComputeFlatInnerDims(NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_outer_dims() {
  return shaped<T, NDIMS>(ComputeFlatOuterDims(NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_dims() const {
  return shaped<T, NDIMS>(ComputeFlatInnerDims(NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_outer_dims() const {
  return shaped<T, NDIMS>(ComputeFlatOuterDims(NDIMS));
}

inline Tensor::Tensor(const Tensor& other)
    : shape_(other.shape()), buf_(other.buf_) {
  if (buf_) buf_->Ref();
}

inline Tensor::Tensor(Tensor&& other)
    : shape_(std::move(other.shape())), buf_(other.buf_) {
  other.buf_ = nullptr;
}

inline Tensor& Tensor::operator=(Tensor&& other) {
  // Avoid self-assignment, since we might destroy our underlying buffer.
  if (&other != this) {
    shape_ = std::move(other.shape_);
    if (buf_) buf_->Unref();
    buf_ = other.buf_;
    other.buf_ = nullptr;
  }
  return *this;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
