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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_ARRAY_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_ARRAY_H_

#include <limits.h>

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/aggregate_ops.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensor_array {

// Full implementations are in tensor_array.cc
template <typename Device, typename T>
Status AddToTensor(OpKernelContext* ctx, Tensor* sum, const Tensor* current,
                   const Tensor* add) {
  return errors::InvalidArgument(
      "tensor_array::AddToTensor type not supported: ",
      DataTypeString(DataTypeToEnum<T>::value));
}

#define TENSOR_ARRAY_WRITE_OR_ADD(Device, T)                         \
  template <>                                                        \
  Status AddToTensor<Device, T>(OpKernelContext * ctx, Tensor * sum, \
                                const Tensor* current, const Tensor* add);

#define TENSOR_ARRAY_WRITE_OR_ADD_CPU(T) TENSOR_ARRAY_WRITE_OR_ADD(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_CPU)
#undef TENSOR_ARRAY_WRITE_OR_ADD_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_WRITE_OR_ADD_GPU(T) TENSOR_ARRAY_WRITE_OR_ADD(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
TF_CALL_COMPLEX_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
#undef TENSOR_ARRAY_WRITE_OR_ADD_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_WRITE_OR_ADD

template <typename Device, typename T>
Status TensorSetZero(OpKernelContext* ctx, Tensor* value) {
  return errors::InvalidArgument(
      "tensor_array::TensorSetZero type not supported: ",
      DataTypeString(DataTypeToEnum<T>::value));
}

#define TENSOR_ARRAY_SET_ZERO(Device, T) \
  template <>                            \
  Status TensorSetZero<Device, T>(OpKernelContext * ctx, Tensor * value);

#define TENSOR_ARRAY_SET_ZERO_CPU(T) TENSOR_ARRAY_SET_ZERO(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_CPU);
TF_CALL_bool(TENSOR_ARRAY_SET_ZERO_CPU);
#undef TENSOR_ARRAY_SET_ZERO_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define TENSOR_ARRAY_SET_ZERO_GPU(T) TENSOR_ARRAY_SET_ZERO(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
TF_CALL_COMPLEX_TYPES(TENSOR_ARRAY_SET_ZERO_GPU);
#undef TENSOR_ARRAY_SET_ZERO_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef TENSOR_ARRAY_SET_ZERO

}  // namespace tensor_array

// The TensorArray object keeps an array of Tensors.  It allows reading from the
// array and writing to the array.
//
// Important properties:
//   * Usually, writing to a particular index in the TensorArray is allowed at
//     most once per index.  In a special case, writes with the flag
//     multiple_writes_aggregate allow multiple writes to the same
//     index.  In this case, the writes are summed.
//   * Multiple reads are supported.
//   * Deep copies of Tensors are rarely made.  The only time they are made is
//     when WriteOrAggregate is called at least twice on the same index with the
//     flag multiple_writes_aggregate = True.
//   * Reading and Writing to the array is protected by a mutex.
//     All operations on a TensorArray are thread-safe.
//   * A TensorArray may be preemptively closed, which releases all
//     memory associated with it.
//
// These properties together allow the TensorArray to work as a
// functional object and makes gradient computation easy.  For
// example:
//   * Write-Once semantics mean the gradient of a TensorArray Read never has to
//     worry which of multiple writes to that index the gradient value
//     is meant for.
//   * Read-Many semantics (when using clear_after_read=false) allow the
//     TensorArray to be read, packed, or concatenated multiple times;
//     and the gradient operations use the multiple_writes_aggregate
//     flag to aggregate the backprop writes.  Multiple backprop writes to
//     the same index are partial gradients corresponding to the
//     multiple reads of that index in the forward phase.
//
class TensorArray : public ResourceBase {
 public:
  static std::atomic<int64_t> tensor_array_counter;

  // Construct a TensorArray for holding Tensors of type 'dtype' with
  // 'N' elements.  While the underlying storage is a std::vector and
  // can hold more than MAX_INT entries, in practice we do not expect
  // users to construct this many Tensors for storage in a TensorArray.
  TensorArray(const string& key, const DataType& dtype, const Tensor& handle,
              int32_t N, const PartialTensorShape& element_shape,
              bool identical_element_shapes, bool dynamic_size,
              bool multiple_writes_aggregate, bool is_grad, int32_t marked_size,
              bool clear_after_read)
      : key_(key),
        dtype_(dtype),
        handle_(handle),
        closed_(false),
        dynamic_size_(dynamic_size),
        multiple_writes_aggregate_(multiple_writes_aggregate),
        gradients_disallowed_(false),
        clear_after_read_(clear_after_read),
        is_grad_(is_grad),
        marked_size_(marked_size),
        element_shape_(element_shape),
        identical_element_shapes_(identical_element_shapes),
        tensors_(N) {}

  // Write Tensor 'value' to index 'index'.
  //
  // Preconditions:
  //  * The TensorArray is not closed
  //  * If the array has dynamic size:
  //      The index is >= 0
  //    Otherwise:
  //      The index is in [0, N) where N == Size()
  //  * The dtype of the Tensor in 'value' matches the TensorArray's dtype.
  //  * If multiple_writes_aggregate is false:
  //    The Tensor at 'index' has not yet been written to.
  //  * If multiple_writes_aggregate is true:
  //    The Tensor at 'index' has the same shape as value.
  //
  // Side effects:
  //  * On the first write to 'index':
  //    - The underlying Tensor in 'value' has a new reference to it.
  //    - The index 'index' is marked as written.
  //  * If multiple_writes_aggregate is false, subsequent writes to 'index'
  //    raise an InvalidArgument error.
  //  * If multiple_writes_aggregate is true, subsequent writes to 'index':
  //    - The underlying Tensors in 'value' and from the first write
  //      are released and a local Tensor is created.
  //    - Index 'index' is also marked as local_copy.
  //    - The gradients_disallowed flag is set true (GradientsAllowed()
  //      will now return false).
  //
  // Note, value is passed as a pointer because we its underlying
  // Tensor's shape is accessed.  Otherwise it is not modified.
  template <typename Device, typename T>
  Status WriteOrAggregate(OpKernelContext* ctx, const int32_t index,
                          const Tensor* value) {
    mutex_lock l(mu_);
    return LockedWriteOrAggregate<Device, T>(ctx, index, value);
  }

  template <typename Device, typename T>
  Status WriteOrAggregateMany(OpKernelContext* ctx,
                              const std::vector<int32>& indices,
                              std::vector<Tensor>* values) {
    mutex_lock l(mu_);
    int32_t i = 0;
    for (const int32_t ix : indices) {
      Status s = LockedWriteOrAggregate<Device, T>(ctx, ix, &(*values)[i]);
      ++i;
      TF_RETURN_IF_ERROR(s);
    }
    return OkStatus();
  }

  // Read from index 'index' into Tensor 'value'.
  //
  // Preconditions:
  //  * The TensorArray is not closed
  //  * The index is in [0, N)
  //  * The Tensor at 'index' has been written to.
  //  * The Tensor at 'index' has not been read from with flag
  //    clear_after_read = true.
  //
  // Side effects:
  //  * If clear_after_read is true, the reference to the underlying
  //    Tensor is deleted.
  //  * The reference to the underlying Tensor at 'index' is copied to
  //    the returned '*value'.
  //  * The index is marked as read (it cannot be rewritten to).
  template <typename Device, typename T>
  Status Read(OpKernelContext* ctx, const int32_t index, Tensor* value) {
    mutex_lock l(mu_);
    return LockedRead<Device, T>(ctx, index, value);
  }

  template <typename Device, typename T>
  Status ReadMany(OpKernelContext* ctx, const std::vector<int32>& indices,
                  std::vector<Tensor>* values) {
    mutex_lock l(mu_);
    values->clear();
    values->resize(indices.size());
    int32_t i = 0;
    for (const int32_t ix : indices) {
      Status s = LockedRead<Device, T>(ctx, ix, &(*values)[i]);
      ++i;
      if (!s.ok()) return s;
    }
    return OkStatus();
  }

  DataType ElemType() const { return dtype_; }

  PartialTensorShape ElemShape() {
    mutex_lock l(mu_);
    return element_shape_;
  }

  Status SetElemShape(const PartialTensorShape& candidate) {
    mutex_lock l(mu_);
    PartialTensorShape new_element_shape_;
    Status s = element_shape_.MergeWith(candidate, &new_element_shape_);
    if (!s.ok()) {
      return s;
    }
    element_shape_ = new_element_shape_;
    return OkStatus();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    CHECK(!closed_);
    return strings::StrCat("TensorArray[", tensors_.size(), "]");
  }

  bool IsClosed() {
    mutex_lock l(mu_);
    return closed_;
  }

  // Return the size of the TensorArray.
  Status Size(int32* size) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = tensors_.size();
    return OkStatus();
  }

  // Record the size of the TensorArray after an unpack or split.
  Status SetMarkedSize(int32_t size) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    if (!is_grad_) {
      marked_size_ = size;
    }
    return OkStatus();
  }

  // Return the marked size of the TensorArray.
  Status MarkedSize(int32* size) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = marked_size_;
    return OkStatus();
  }

  // Return the size that should be used by pack or concat op.
  Status PackOrConcatSize(int32* size) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = is_grad_ ? marked_size_ : tensors_.size();
    return OkStatus();
  }

  // Once a TensorArray is being used for gradient calculations, it
  // should be marked as no longer resizeable.
  void DisableDynamicSize() {
    mutex_lock l(mu_);
    dynamic_size_ = false;
  }

  bool HasDynamicSize() {
    mutex_lock l(mu_);
    return dynamic_size_;
  }

  bool GradientsAllowed() {
    mutex_lock l(mu_);
    return !gradients_disallowed_;
  }

  bool HasIdenticalElementShapes() const { return identical_element_shapes_; }

  // Copy the TensorShapes from another TensorArray into this one.
  // If `shapes_to_prepend` is set, expands the rank of the copied shape by
  // prepending the passed in shape prefix to the shape values in `rhs`.
  // The sizes of the two TensorArrays must match and this one
  // may not have any entries filled in.  This performs a "soft copy",
  // essentially filling the current TensorArray with virtual
  // zero-tensors, which will be replaced by future aggregate writes,
  // or instantiated by future reads.  Requires a non-const pointer
  // to the rhs to access its mutex.
  Status CopyShapesFrom(TensorArray* rhs, const TensorShape* shape_to_prepend);

  // Clear the TensorArray, including any Tensor references, and mark as closed.
  void ClearAndMarkClosed() {
    mutex_lock l(mu_);
    tensors_.clear();
    closed_ = true;
  }

  mutex* mu() { return &mu_; }
  Tensor* handle() { return &handle_; }

  ResourceHandle resource_handle(OpKernelContext* ctx) {
    return ctx->step_container()->MakeResourceHandle<TensorArray>(
        key_, *ctx->device());
  }

 private:
  Status LockedWrite(OpKernelContext* ctx, const int32_t index, Tensor* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename Device, typename T>
  Status LockedWriteOrAggregate(OpKernelContext* ctx, const int32_t index,
                                const Tensor* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename Device, typename T>
  Status LockedRead(OpKernelContext* ctx, const int32_t index, Tensor* value)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status LockedReturnIfClosed() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (closed_) {
      return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                     " has already been closed.");
    }
    return OkStatus();
  }

  const string key_;

  const DataType dtype_;
  Tensor handle_;

  mutable mutex mu_;

  // Marks that the tensor_array_ has been cleared.
  bool closed_ TF_GUARDED_BY(mu_);

  // Writes are allowed to grow the array.
  bool dynamic_size_;

  // Multiple writes to the same index will result in summation of the
  // values (used by backprop)
  const bool multiple_writes_aggregate_;

  // If multiple Writes were attempted (e.g. via attribute
  // multiple_writes_aggregate), then gradients are disallowed.
  bool gradients_disallowed_ TF_GUARDED_BY(mu_);

  // After a read at an index, clear away its Tensor to release memory.
  const bool clear_after_read_;

  // True iff this is a gradient tensor array.
  const bool is_grad_;

  // The size of the TensorArray after a (legacy) unpack or split is performed.
  // -1 if there has been no unpack or split performed on the TensorArray.
  int32 marked_size_;

  // The shape of each element in the TensorArray, may be partially known or not
  // known at all.
  PartialTensorShape element_shape_ TF_GUARDED_BY(mu_);

  // Whether all elements in the TensorArray have identical shapes.
  // This allows certain behaviors, like dynamically checking for
  // consistent shapes on write, and being able to fill in properly
  // shaped zero tensors on stack -- even if the initial element_shape
  // was not fully defined.
  const bool identical_element_shapes_;

  // TensorAndState is used to keep track of the Tensors stored in the
  // TensorArray, along with their shapes, and a boolean that determines whether
  // they have already been read or not.
  struct TensorAndState {
    TensorAndState()
        : written(false), read(false), cleared(false), local_copy(false) {}
    Tensor tensor;
    TensorShape shape;
    bool written;  // True if a Tensor has been written to the index.
    bool read;  // True if a Tensor has been written to and read from the index.
    bool cleared;  // True if a tensor has been read with
                   // clear_after_read = true;

    // Used by writes when multiple_writes_aggregate is true.  In this
    // case, the first time a value is written, it is a shallow copy.
    // The second time a value is written, it is aggregated.  However,
    // in this case a new Tensor must be constructed to hold the
    // aggregated value.  This flag marks that such a Tensor is being
    // used.  All future writes will aggregate to the existing local Tensor.
    bool local_copy;
  };
  // The list of underlying Tensors and states.
  std::vector<TensorAndState> tensors_ TF_GUARDED_BY(mu_);
};

template <typename Device, typename T>
Status TensorArray::LockedWriteOrAggregate(OpKernelContext* ctx,
                                           const int32_t index,
                                           const Tensor* value) {
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  size_t index_size = static_cast<size_t>(index);
  if (index < 0 || (!dynamic_size_ && index_size >= tensors_.size())) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<tstring>()(1), ": Tried to write to index ",
        index, " but array is not resizeable and size is: ", tensors_.size());
  }
  if (dynamic_size_) {
    // We must grow the internal TensorArray
    if (index_size >= tensors_.capacity()) {
      tensors_.reserve(2 * (index_size + 1));
    }
    if (index_size >= tensors_.size()) {
      tensors_.resize(index_size + 1);
    }
  }
  TensorAndState& t = tensors_[index];

  if (value->dtype() != dtype_) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<tstring>()(1),
        ": Could not write to TensorArray index ", index,
        " because the value dtype is ", DataTypeString(value->dtype()),
        " but TensorArray dtype is ", DataTypeString(dtype_), ".");
  }
  if (!element_shape_.IsCompatibleWith(value->shape())) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<tstring>()(1),
        ": Could not write to TensorArray index ", index,
        " because the value shape is ", value->shape().DebugString(),
        " which is incompatible with the TensorArray's inferred element "
        "shape: ",
        element_shape_.DebugString(), " (consider setting infer_shape=False).");
  } else if (identical_element_shapes_ && !element_shape_.IsFullyDefined()) {
    element_shape_ = PartialTensorShape(value->shape().dim_sizes());
  }

  if (t.read) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                   ": Could not write to TensorArray index ",
                                   index, " because it has already been read.");
  }

  if (!multiple_writes_aggregate_ && t.written) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                   ": Could not write to TensorArray index ",
                                   index,
                                   " because it has already been written to.");
  }

  if (t.written) {
    DCHECK(multiple_writes_aggregate_);

    // Check that value shape matches t.shape
    if (value->shape() != t.shape) {
      return errors::InvalidArgument(
          "TensorArray ", handle_.vec<tstring>()(1),
          ": Could not aggregate to TensorArray index ", index,
          " because the existing shape is ", t.shape.DebugString(),
          " but the new input shape is ", value->shape().DebugString(), ".");
    }

    if (!t.tensor.IsInitialized() || t.tensor.NumElements() == 0) {
      // If existing_t == nullptr but written == true, then what was stored
      // was just a shape, which just means zeros.  So all we must do in this
      // case is copy the reference over and return early.
      t.tensor = *value;
      return OkStatus();
    }

    Tensor* existing_t = &t.tensor;

    if (t.local_copy) {
      Status s = tensor_array::AddToTensor<Device, T>(ctx, existing_t,
                                                      existing_t, value);
      TF_RETURN_IF_ERROR(s);
    } else {
      Tensor local_tensor;
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(dtype_, existing_t->shape(), &local_tensor));
      Status s = tensor_array::AddToTensor<Device, T>(ctx, &local_tensor,
                                                      existing_t, value);
      TF_RETURN_IF_ERROR(s);
      t.tensor = local_tensor;
      t.local_copy = true;
    }

    // We've aggregated the values, so disallow backprop on this
    // TensorArray.
    gradients_disallowed_ = true;
  } else {
    t.tensor = *value;
    t.shape = value->shape();
    t.written = true;
  }
  return OkStatus();
}

template <typename Device, typename T>
Status TensorArray::LockedRead(OpKernelContext* ctx, const int32_t index,
                               Tensor* value) {
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  if ((index < 0) ||
      (!is_grad_ && (static_cast<size_t>(index) >= tensors_.size()))) {
    return errors::InvalidArgument("Tried to read from index ", index,
                                   " but array size is: ", tensors_.size());
  }
  size_t index_t = static_cast<size_t>(index);
  if ((is_grad_ && (index_t >= tensors_.size() || !tensors_[index].written)) ||
      (!is_grad_ && (index_t < tensors_.size() && !tensors_[index].written))) {
    // Special case returning zeros if this is a gradient read that happens
    // after a stop_gradients call with dynamic forward TensorArrays.
    // There is sometimes a race condition where the gradient is not
    // written due to stop_gradients, but is later read.
    TensorShape element_shape;
    if (is_grad_ && index_t < tensors_.size() &&
        tensors_[index].shape.dims() > 0) {
      // A gradient TensorArray has more specific gradient information
      // available for each entry.  A forward TensorArray must rely on
      // the global element_shape_ to fill in zeros on read.
      element_shape = tensors_[index].shape;
    } else if (!element_shape_.IsFullyDefined()) {
      return errors::InvalidArgument(
          "TensorArray ", handle_.vec<tstring>()(1),
          ": Could not read from TensorArray index ", index,
          ".  Furthermore, the element shape is not fully defined: ",
          element_shape_.DebugString(),
          ".  It is possible you are working with a resizeable TensorArray and "
          "stop_gradients is not allowing the gradients to be written.  If you "
          "set the full "
          "element_shape property on the forward TensorArray, the proper "
          "all-zeros tensor "
          "will be returned instead of incurring this error.");
    } else {
      element_shape_.AsTensorShape(&element_shape);  // Always succeeds.
    }
    if (index_t >= tensors_.size()) {
      // Fill in tensors_ up to index to have known shape.
      size_t old_tensors_size = tensors_.size();
      tensors_.resize(index + 1);
      for (size_t i = old_tensors_size; i < index + 1; ++i) {
        tensors_[i].shape = element_shape;
        tensors_[i].written = true;
      }
    } else {
      tensors_[index].shape = element_shape;
      tensors_[index].written = true;
    }
  }

  TensorAndState& t = tensors_[index];

  if (t.cleared) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<tstring>()(1),
                                   ": Could not read index ", index,
                                   " twice because it was cleared after a "
                                   "previous read (perhaps try setting "
                                   "clear_after_read = false?).");
  }

  if (!t.tensor.IsInitialized() || t.tensor.NumElements() == 0) {
    // We stored just a shape, but no value.  This means create and
    // return zeros of the appropriate shape.
    TF_RETURN_IF_ERROR(ctx->allocate_temp(dtype_, t.shape, &t.tensor));
    if (t.shape.num_elements() > 0) {
      Status s = tensor_array::TensorSetZero<Device, T>(ctx, &t.tensor);
      if (!s.ok()) return s;
    }
  }

  // Data is available inside the tensor, copy the reference over.
  *value = t.tensor;

  if (clear_after_read_) {
    t.tensor = Tensor();
    t.cleared = true;
  }
  t.read = true;
  return OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_ARRAY_H_
