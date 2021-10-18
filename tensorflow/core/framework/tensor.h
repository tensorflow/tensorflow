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

#include <cstdint>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/allocator.h"
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
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Forward declarations.  In particular, we forward declare protos so that their
// symbols can be removed from .so exports.
class AllocationDescription;
class Allocator;
class OpKernelContext;
class Tensor;
class TensorBuffer;
class TensorCApi;
class TensorCord;
class TensorDescription;
class TensorProto;
class Var;

namespace batch_util {
Status CopyElementToSlice(Tensor element, Tensor* parent, int64_t index);
Status CopySliceToElement(const Tensor& parent, Tensor* element, int64_t index);
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64_t index);
Status CopyContiguousSlices(const Tensor& src, int64_t src_offset,
                            int64_t dst_offset, int64_t num_slices,
                            Tensor* dst);
}  // namespace batch_util

/// @ingroup core

/// Interface to access the raw ref-counted data buffer.
class TensorBuffer : public core::RefCounted {
 public:
  explicit TensorBuffer(void* data_ptr) : data_(data_ptr) {}
  ~TensorBuffer() override {}

  /// \brief data() points to a memory region of size() bytes.
  ///
  /// NOTE(mrry): The `data()` method is not virtual for performance reasons.
  /// It can be called multiple times when the contents of a `Tensor` are
  /// accessed, and so making it non-virtual allows the body to be inlined.
  void* data() const { return data_; }

  /// \brief Size (in bytes) of the buffer.
  virtual size_t size() const = 0;

  /// \brief If this TensorBuffer is sub-buffer of another TensorBuffer,
  /// returns that TensorBuffer. Otherwise, returns this.
  virtual TensorBuffer* root_buffer() = 0;

  /// \brief Fills metadata about the allocation into the proto.
  virtual void FillAllocationDescription(
      AllocationDescription* proto) const = 0;

  virtual bool GetAllocatedBytes(size_t* out_bytes) const;

  /// \brief Helper method to reinterpret the buffer as an array of `T`.
  template <typename T>
  T* base() const {
    return reinterpret_cast<T*>(data());
  }

  /// \brief Whether this TensorBuffer owns the underlying memory.
  virtual bool OwnsMemory() const { return true; }

 private:
  void* const data_;
};

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

  /// \brief Creates a tensor with the input datatype, shape and buf.
  ///
  /// Acquires a ref on buf that belongs to this Tensor.
  Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

  /// \brief Creates a tensor with the input datatype, shape and buf.
  ///
  /// Takes an ownership of the bufffer from the reference counted pointer.
  Tensor(DataType type, TensorShape shape, core::RefCountPtr<TensorBuffer> buf);

  /// \brief Creates an empty Tensor of the given data type.
  ///
  /// Like Tensor(), returns a 1-dimensional, 0-element Tensor with
  /// IsInitialized() returning True. See the Tensor() documentation
  /// for details.
  explicit Tensor(DataType type);

  /// \brief Initializes a tensor with the input `type` and `shape`, or returns
  /// an error and leaves `out_tensor` unmodified. This factory method should be
  /// used instead of the corresponding constructor if calling code cannot
  /// validate that the `DataType` is valid and supported.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  static Status BuildTensor(DataType type, const TensorShape& shape,
                            Tensor* out_tensor);

 private:
  // A tag type for selecting the `Tensor` constructor overload that creates a
  // scalar tensor in host memory.
  struct host_scalar_tag {};

  class HostScalarTensorBufferBase;
  template <typename T>
  struct ValueAndTensorBuffer;

  // Creates a tensor with the given scalar `value` in CPU memory.
  template <typename T>
  Tensor(T value, host_scalar_tag tag);

 public:
  // A series of specialized constructors for scalar tensors in host memory.
  //
  // NOTE: The `Variant` host-scalar constructor is not defined, because Variant
  // is implicitly constructible from many different types, and this causes
  // ambiguities with some compilers.
  explicit Tensor(float scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(double scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int32_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint32 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int16_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int8_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(tstring scalar_value)
      : Tensor(std::move(scalar_value), host_scalar_tag{}) {}
  explicit Tensor(complex64 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(complex128 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int64_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint64 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(bool scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(qint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(quint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(qint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(quint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(qint32 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(bfloat16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(Eigen::half scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(ResourceHandle scalar_value)
      : Tensor(std::move(scalar_value), host_scalar_tag{}) {}

  // NOTE: The `const char*` host-scalar constructor is provided as a
  // convenience because otherwise passing a string literal would surprisingly
  // construct a DT_BOOL tensor.
  explicit Tensor(const char* scalar_value)
      : Tensor(tstring(scalar_value), host_scalar_tag{}) {}

  /// Copy constructor.
  Tensor(const Tensor& other);

  /// \brief Move constructor. After this call, <other> is safely destructible
  /// can be assigned to, and IsInitialized() can be called and will return
  /// false. Other calls on <other> (e.g. shape manipulation) are not valid.
  Tensor(Tensor&& other);

  // Explicitly delete constructor that take a pointer (except char*)
  // so that the pointer doesn't get implicitly cast to bool.
  template <typename T, typename std::enable_if<!std::is_same<T, char>::value,
                                                T>::type* = nullptr>
  explicit Tensor(T* t) = delete;

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
  int64_t dim_size(int d) const { return shape().dim_size(d); }

  /// Convenience accessor for the tensor shape.
  int64_t NumElements() const { return shape().num_elements(); }

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

  // Returns the size of allocated memory for this tensor.
  size_t AllocatedBytes() const;

  /// Returns true iff this tensor is aligned.
  bool IsAligned() const {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return true;
#else
    void* ptr = base<void>();
    return dtype() == DT_STRING || NumElements() == 0 ||
           (reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0);
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
  /// NOTE: The returned tensor may not satisfy the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// NOTE: When fed with an N-dimensional tensor, this method returns a tensor
  /// also with N dimensions. If you want to select a sub tensor, see SubSlice.
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= dim0_start <= dim0_limit <= dim_size(0)`
  Tensor Slice(int64_t dim0_start, int64_t dim0_limit) const;

  /// \brief Select a subslice from this tensor along the 1st dimension.
  ///
  /// When fed with an N-dimensional tensor, this method returns a tensor with
  /// N-1 dimensions, where the returned tensor is a subslice of the input
  /// tensor along the first dimension. The N-1 dimensions of the returned
  /// tensor are the last N-1 dimensions of the input tensor.
  ///
  /// NOTE: The returned tensor may not satisfy the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= index < dim_size(0)`
  Tensor SubSlice(int64_t index) const;

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

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// last dimension elements converted into single elements of a larger type.
  ///
  /// For example, this is useful for kernels that can treat NCHW_VECT_C int8
  /// tensors as NCHW int32 tensors. The sizeof(T) should equal the size of
  /// the original element type * num elements in the original last dimension.
  /// NDIMS should be 1 less than the original number of dimensions.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor reinterpret_last_dimension();

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

  /// Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing the
  /// first 'begin' Tensor dimensions into the first dimension of the result and
  /// the Tensor dimensions of the last dims() - 'begin' - NDIMS into the last
  /// dimension of the result. If 'begin' < 0 then the |'begin'| leading
  /// dimensions of size 1 will be added. If 'begin' + NDIMS > dims() then
  /// 'begin' + NDIMS - dims() trailing dimensions of size 1 will be added.
  template <typename T, size_t NDIMS = 3>
  typename TTypes<T, NDIMS>::Tensor flat_inner_outer_dims(int64_t begin);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64_t> new_sizes);

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor bit_casted_shaped(
      gtl::ArraySlice<int64_t> new_sizes);

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedTensor unaligned_shaped(
      gtl::ArraySlice<int64_t> new_sizes);

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

  /// \brief Return the tensor data to an `Eigen::Tensor` with the
  /// last dimension elements converted into single elements of a larger type.
  ///
  /// For example, this is useful for kernels that can treat NCHW_VECT_C int8
  /// tensors as NCHW int32 tensors. The sizeof(T) should equal the size of
  /// the original element type * num elements in the original last dimension.
  /// NDIMS should be 1 less than the original number of dimensions.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor reinterpret_last_dimension() const;

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
      gtl::ArraySlice<int64_t> new_sizes) const;

  /// \brief Return the tensor data to an `Eigen::Tensor` with the new
  /// shape specified in `new_sizes` and cast to a new dtype `T`.
  ///
  /// Using a bitcast is useful for move and copy operations.
  /// The allowed bitcast is the only difference from `shaped()`.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor bit_casted_shaped(
      gtl::ArraySlice<int64_t> new_sizes) const;

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::UnalignedConstTensor unaligned_shaped(
      gtl::ArraySlice<int64_t> new_sizes) const;

  template <typename T>
  typename TTypes<T>::ConstScalar scalar() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_dims() const;

  template <typename T, size_t NDIMS = 2>
  typename TTypes<T, NDIMS>::ConstTensor flat_outer_dims() const;

  template <typename T, size_t NDIMS = 3>
  typename TTypes<T, NDIMS>::ConstTensor flat_inner_outer_dims(
      int64_t begin) const;

  /// Render the first `max_entries` values in `*this` into a string.
  std::string SummarizeValue(int64_t max_entries, bool print_v2 = false) const;

  /// A human-readable summary of the tensor suitable for debugging.
  // `num_values` is the number of actual data values in the tensor
  // included in the message. If the tensor might be resident in
  // GPU/TPU memory use DeviceSafeDebugString instead.
  std::string DebugString(int num_values) const;
  std::string DebugString() const { return DebugString(3); }

  // Variant of DebugString() that should be used for possibly non-CPU tensors.
  // If the tensor is not resident on CPU, we can't read its values as
  // DebugString() does.
  std::string DeviceSafeDebugString() const;

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
  void* data() const;

  /// Copy the other tensor into this tensor, reshape it and reinterpret the
  /// buffer's datatype. If Status::OK() is returned, the two tensors now share
  /// the same underlying storage.
  ///
  /// This call requires that the `other` tensor and the given type and shape
  /// are "compatible" (i.e. they occupy the same number of bytes).
  ///
  /// Specifically:
  ///
  /// shape.num_elements() * DataTypeSize(type)
  ///
  /// must equal
  ///
  /// other.num_elements() * DataTypeSize(other.dtype())
  ///
  /// In addition, this function requires:
  ///   * DataTypeSize(other.dtype()) != 0
  ///   * DataTypeSize(type) != 0
  ///
  /// If any of the requirements are not met, errors::InvalidArgument is
  /// returned.
  Status BitcastFrom(const Tensor& other, DataType dtype,
                     const TensorShape& shape);

  /// Like BitcastFrom, but CHECK fails if any preconditions are not met.
  ///
  /// Deprecated. Use BitcastFrom instead and check the returned Status.
  void UnsafeCopyFromInternal(const Tensor& other, DataType dtype,
                              const TensorShape& shape) {
    TF_CHECK_OK(BitcastFrom(other, dtype, shape));
  }

  // Returns true if the refcount on buf_ and any possible underlying root
  // buffer is one.
  bool RefCountIsOne() const;

 private:
  void CheckType(DataType expected_dtype) const;
  void CheckTypeAndIsAligned(DataType expected_dtype) const;
  void CheckIsAlignedAndSingleElement() const;
  void set_dtype(DataType t) { shape_.set_data_type(t); }

  // TensorShape's InlineVector.
  static gtl::InlinedVector<int64_t, 4> ComputeFlatInnerDims(
      gtl::ArraySlice<int64_t> orig, int64_t num_out_dims);
  static gtl::InlinedVector<int64_t, 4> ComputeFlatOuterDims(
      gtl::ArraySlice<int64_t> orig, int64_t num_out_dims);

  TensorShape shape_;
  TensorBuffer* buf_;

  friend class DMAHelper;             // For access to buf_.
  friend class TensorCApi;            // For access to buf_.
  friend class TensorCord;            // For access to buf_.
  friend class TensorReference;       // For access to buf_.
  friend class VariableOp;            // For access to set_shape.
  friend class AutoReloadVariableOp;  // For access to set_shape.
  friend class TensorTestHelper;      // For access to set_shape.
  friend class CastOpBase;            // For access to set_dtype.
  friend class ScopedAllocator;       // For access to buf_.
  friend Status batch_util::CopyElementToSlice(
      Tensor element, Tensor* parent,
      int64_t index);  // For access to base<T>().
  friend Status batch_util::CopySliceToElement(
      const Tensor& parent, Tensor* element,
      int64_t index);  // For access to base<T>().
  friend Status batch_util::MaybeMoveSliceToElement(
      Tensor* parent, Tensor* element,
      int64_t index);  // For access to base<T>().
  friend Status batch_util::CopyContiguousSlices(
      const Tensor& src, int64_t src_offset, int64_t dst_offset,
      int64_t num_slices,
      Tensor* dst);  // For access to base<T>().

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

  inline void CopyFromInternal(const Tensor& other, const TensorShape& shape) {
    DCHECK_EQ(shape.num_elements(), other.NumElements());
    // Data type will be overwritten if this == &other, since dtype is part of
    // shape.
    DataType other_dtype = other.dtype();
    shape_ = shape;
    set_dtype(other_dtype);
    if (buf_ != other.buf_) {
      if (buf_) buf_->Unref();
      buf_ = other.buf_;
      if (buf_) buf_->Ref();
    }
  }

  template <typename T>
  T* base() const;

  template <size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64_t> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;

  template <typename T, size_t NDIMS>
  void FillDimsAndValidateCompatibleShape(
      gtl::ArraySlice<int64_t> new_sizes,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const;
};

// Implementation details

// START_SKIP_DOXYGEN

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
  return typename TTypes<T, NDIMS>::Tensor(base<T>(),
                                           shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_tensor() const {
  CHECK(IsAligned());
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(),
                                                shape().AsEigenDSizes<NDIMS>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::reinterpret_last_dimension() {
  if (NDIMS == dims()) {
    return tensor<T, NDIMS>();
  }
  CHECK(IsAligned());
  CHECK_EQ(NDIMS, dims() - 1);
  CHECK_EQ(sizeof(T), shape_.dim_sizes()[NDIMS] * DataTypeSize(dtype()));
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = shape_.dim_sizes()[d];
  }
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::reinterpret_last_dimension()
    const {
  if (NDIMS == dims()) {
    return tensor<T, NDIMS>();
  }
  CHECK(IsAligned());
  CHECK_EQ(NDIMS, dims() - 1);
  CHECK_EQ(sizeof(T), shape_.dim_sizes()[NDIMS] * DataTypeSize(dtype()));
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = shape_.dim_sizes()[d];
  }
  return typename TTypes<T, NDIMS>::ConstTensor(base<const T>(), dims);
}

template <size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64_t> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
  CHECK_EQ(NDIMS, new_sizes.size());
  int64_t new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  CHECK_EQ(new_num_elements, NumElements());
}

template <typename T, size_t NDIMS>
void Tensor::FillDimsAndValidateCompatibleShape(
    gtl::ArraySlice<int64_t> new_sizes,
    Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
  CHECK_EQ(NDIMS, new_sizes.size());
  int64_t new_num_elements = 1;
  for (size_t d = 0; d < NDIMS; d++) {
    new_num_elements *= new_sizes[d];
    (*dims)[d] = new_sizes[d];
  }
  const int element_size = DataTypeSize(BaseType(dtype()));
  if (element_size > 0) {
    CHECK_EQ(new_num_elements * sizeof(T), NumElements() * element_size);
  } else {
    // DataTypeSize() returns 0 for some data types. In this case, assume that T
    // has the same size as the buffer type.
    // NOTE: If we can be sure that DataTypeSize() does not return 0 for all POD
    // types, then we should check DataTypeToEnum<T>::v() == dtype(). Or simply
    // check if `element_size > 0` to err when bit cast is attempted on Tensor
    // of unknown data type size.
    CHECK_EQ(new_num_elements, NumElements());
  }
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::shaped(
    gtl::ArraySlice<int64_t> new_sizes) {
  CheckTypeAndIsAligned(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64_t> new_sizes) {
  CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<T>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64_t> new_sizes) {
  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::shaped(
    gtl::ArraySlice<int64_t> new_sizes) const {
  CheckType(DataTypeToEnum<T>::v());
  CHECK(IsAligned()) << "ptr = " << base<void>();
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::bit_casted_shaped(
    gtl::ArraySlice<int64_t> new_sizes) const {
  CHECK(IsAligned());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape<T>(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::UnalignedConstTensor Tensor::unaligned_shaped(
    gtl::ArraySlice<int64_t> new_sizes) const {
  CheckType(DataTypeToEnum<T>::v());
  Eigen::array<Eigen::DenseIndex, NDIMS> dims;
  FillDimsAndValidateCompatibleShape(new_sizes, &dims);
  return typename TTypes<T, NDIMS>::UnalignedConstTensor(base<T>(), dims);
}

template <typename T>
typename TTypes<T>::Scalar Tensor::scalar() {
  static_assert(
      !std::is_same<T, std::string>::value,
      "std::string is no longer a scalar type, use tensorflow::tstring");
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::Scalar(base<T>());
}

template <typename T>
typename TTypes<T>::ConstScalar Tensor::scalar() const {
  static_assert(
      !std::is_same<T, std::string>::value,
      "std::string is no longer a scalar type, use tensorflow::tstring");
  CheckIsAlignedAndSingleElement();
  return typename TTypes<T>::ConstScalar(base<T>());
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_dims() {
  return shaped<T, NDIMS>(ComputeFlatInnerDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_outer_dims() {
  return shaped<T, NDIMS>(ComputeFlatOuterDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::flat_inner_outer_dims(int64_t begin) {
  gtl::InlinedVector<int64_t, 4> flat_outer =
      ComputeFlatOuterDims(shape_.dim_sizes(), begin + NDIMS);
  return shaped<T, NDIMS>(ComputeFlatInnerDims(flat_outer, NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_dims() const {
  return shaped<T, NDIMS>(ComputeFlatInnerDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_outer_dims() const {
  return shaped<T, NDIMS>(ComputeFlatOuterDims(shape_.dim_sizes(), NDIMS));
}

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::ConstTensor Tensor::flat_inner_outer_dims(
    int64_t begin) const {
  gtl::InlinedVector<int64_t, 4> flat_outer =
      ComputeFlatOuterDims(shape_.dim_sizes(), begin + NDIMS);
  return shaped<T, NDIMS>(ComputeFlatInnerDims(flat_outer, NDIMS));
}

inline Tensor::Tensor(const Tensor& other)
    : shape_(other.shape()), buf_(other.buf_) {
  if (buf_) buf_->Ref();
}

inline Tensor::Tensor(Tensor&& other)
    : shape_(std::move(other.shape_)), buf_(other.buf_) {
  other.buf_ = nullptr;
}

class Tensor::HostScalarTensorBufferBase : public TensorBuffer {
 public:
  using TensorBuffer::TensorBuffer;
  bool GetAllocatedBytes(size_t* out_bytes) const final;
  void FillAllocationDescription(AllocationDescription* proto) const final;
};

// A packed representation for a single scalar value of type `T`, and a
// `TensorBuffer` implementation that describes (and manages the lifetime of)
// that value.
template <typename T>
struct Tensor::ValueAndTensorBuffer {
  class HostScalarTensorBuffer : public Tensor::HostScalarTensorBufferBase {
   public:
    explicit HostScalarTensorBuffer(void* data)
        : HostScalarTensorBufferBase(data) {}
    size_t size() const final { return sizeof(T); }
    TensorBuffer* root_buffer() final { return this; }

    // Override `operator delete` so that calling `delete this` in
    // `core::Refcounted::Unref()` for an object of this type will free
    // the enclosing `ValueAndTensorBuffer` for the tensor buffer.
    //
    // NOTE(mrry): The definition of this method must be outside the class
    // definition in order to satisfy some compilers.
    static void operator delete(void* ptr);

    static void operator delete(void*, void*) {
      // Some compilers require an overridden class-specific deallocation
      // function, which will be called if placement `new` throws an
      // exception.
    }

   private:
    ~HostScalarTensorBuffer() override { static_cast<T*>(data())->~T(); }
  };

  T value;
  HostScalarTensorBuffer tensor_buffer;
};

/* static */
template <typename T>
void Tensor::ValueAndTensorBuffer<T>::HostScalarTensorBuffer::operator delete(
    void* ptr) {
  // Use a dummy object to compute to offset of
  // `ValueAndTensorBuffer::tensor_buffer`, because `offsetof()` is not
  // necessarily defined on this non-POD type (until C++17).
  //
  // NOTE(mrry): Using `sizeof(Tensor::ValueAndTensorBuffer<T>)` here requires
  // us to define this method outside the class definition, so that it is not
  // considered an incomplete type.
  typename std::aligned_storage<sizeof(Tensor::ValueAndTensorBuffer<T>),
                                alignof(Tensor::ValueAndTensorBuffer<T>)>::type
      dummy_storage_;
  Tensor::ValueAndTensorBuffer<T>* dummy_object =
      reinterpret_cast<Tensor::ValueAndTensorBuffer<T>*>(&dummy_storage_);
  intptr_t offset = reinterpret_cast<intptr_t>(&dummy_object->tensor_buffer) -
                    reinterpret_cast<intptr_t>(dummy_object);

  port::AlignedFree(static_cast<char*>(ptr) - offset);
}

template <typename T>
Tensor::Tensor(T value, host_scalar_tag tag) {
  auto* value_and_buf = static_cast<Tensor::ValueAndTensorBuffer<T>*>(
      port::AlignedMalloc(sizeof(typename Tensor::ValueAndTensorBuffer<T>),
                          EIGEN_MAX_ALIGN_BYTES));
  new (&value_and_buf->value) T(std::move(value));
  new (&value_and_buf->tensor_buffer)
      typename Tensor::ValueAndTensorBuffer<T>::HostScalarTensorBuffer(
          value_and_buf);
  buf_ = &value_and_buf->tensor_buffer;
  set_dtype(DataTypeToEnum<T>::value);
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

// END_SKIP_DOXYGEN

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
