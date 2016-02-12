/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_TENSOR_ARRAY_H_
#define TENSORFLOW_KERNELS_TENSOR_ARRAY_H_

#include <limits.h>
#include <vector>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// The TensorArray object keeps an array of PersistentTensors.  It
// allows reading from the array and writing to the array.
//
// Important properties:
//   * Reading and writing to a particular index in the TensorArray
//     is allowed at most once per index.
//   * Upon reading an entry, that entry is cleared from the array and
//     marked as read.  This allows removal of Tensor from memory
//     as soon as it is not needed.  Its shape is saved.
//   * No deep copies of any PersistentTensor are ever made.
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
//   * Read-Once semantics mean the TensorArray never sees
//     multiple writes to the same index as part of gradient aggregation.
//
class TensorArray : public ResourceBase {
 public:
  // Construct a TensorArray for holding Tensors of type 'dtype' with
  // 'N' elements.  While the underlying storage is a std::vector and
  // can hold more than MAX_INT entries, in practice we do not expect
  // users to construct this many Tensors for storage in a TensorArray.
  TensorArray(const DataType& dtype, const Tensor& handle, int32 N,
              bool dynamic_size)
      : dtype_(dtype),
        handle_(handle),
        closed_(false),
        dynamic_size_(dynamic_size),
        tensors_(N) {}

  // Write PersistentTensor 'value' to index 'index'.
  //
  // Preconditions:
  //  * The TensorArray is not closed
  //  * If the array has dynamic size:
  //      The index is >= 0
  //    Otherwise:
  //      The index is in [0, N) where N == Size()
  //  * The dtype of the Tensor in 'value' matches the TensorArray's dtype.
  //  * The Tensor at 'index' has not yet been written to.
  //
  // Side effects:
  //  * The underlying Tensor in 'value' has a new reference to it.
  //  * Index 'index' is marked as written.
  //
  // Note, value is passed as a pointer because we its underlying
  // Tensor's shape is accessed.  Otherwise it is not modified.
  Status Write(OpKernelContext* ctx, const int32 index,
               PersistentTensor* value) {
    mutex_lock l(mu_);
    return LockedWrite(ctx, index, value);
  }

  Status WriteMany(OpKernelContext* ctx,
                   std::vector<PersistentTensor>* values) {
    mutex_lock l(mu_);
    for (int32 i = values->size() - 1; i >= 0; --i) {
      TF_RETURN_IF_ERROR(LockedWrite(ctx, i, &(*values)[i]));
    }
    return Status::OK();
  }

  // Read from index 'index' into PersistentTensor 'value'.
  //
  // Preconditions:
  //  * The TensorArray is not closed
  //  * The index is in [0, N)
  //  * The Tensor at 'index' has been written to.
  //  * The Tensor at 'index' has not already been read.
  //
  // Side effects:
  //  * The PersistentTensor at 'index' is cleared from the given index.
  //  * The reference to the underlying Tensor at 'index' is shifted to
  //    the returned '*value'.
  //  * Index 'index' is marked as read.
  Status Read(const int32 index, PersistentTensor* value) {
    mutex_lock l(mu_);
    return LockedRead(index, value);
  }

  Status ReadMany(std::vector<PersistentTensor>* values) {
    mutex_lock l(mu_);
    values->clear();
    values->resize(tensors_.size());
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      TF_RETURN_IF_ERROR(LockedRead(i, &(*values)[i]));
    }
    return Status::OK();
  }

  DataType ElemType() const { return dtype_; }

  string DebugString() override {
    mutex_lock l(mu_);
    CHECK(!closed_);
    return strings::StrCat("TensorArray[", tensors_.size(), "]");
  }

  bool IsClosed() {
    mutex_lock l(mu_);
    return closed_;
  }

  // Return the Size of the TensorArray.
  Status Size(int32* size) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(LockedReturnIfClosed());
    *size = tensors_.size();
    return Status::OK();
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

  // Clear the TensorArray, including any Tensor references, and mark as closed.
  void ClearAndMarkClosed() {
    mutex_lock l(mu_);
    tensors_.clear();
    closed_ = true;
  }

  mutex* mu() { return &mu_; }
  Tensor* handle() { return &handle_; }

 private:
  Status LockedWrite(OpKernelContext* ctx, const int32 index,
                     PersistentTensor* value) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status LockedRead(const int32 index, PersistentTensor* value)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status LockedReturnIfClosed() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (closed_) {
      return errors::InvalidArgument("TensorArray ", handle_.vec<string>()(1),
                                     " has already been closed.");
    }
    return Status::OK();
  }

  const DataType dtype_;
  Tensor handle_;

  mutex mu_;

  bool closed_
      GUARDED_BY(mu_);  // Marks that the tensor_array_ has been cleared.

  bool dynamic_size_;  // Determines if Writes are allowed to grow the array.

  // TensorAndState is used to keep track of the PersistentTensors
  // stored in the TensorArray, along with their shapes, and a boolean
  // that determines whether they have already been read or not.
  struct TensorAndState {
    TensorAndState() : written(false), read(false) {}
    PersistentTensor tensor;
    TensorShape shape;
    bool written;  // True if a Tensor has been written to the index.
    bool read;  // True if a Tensor has been written to and read from the index.
  };
  // The list of underlying PersistentTensors and states.
  std::vector<TensorAndState> tensors_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TENSOR_ARRAY_H_
