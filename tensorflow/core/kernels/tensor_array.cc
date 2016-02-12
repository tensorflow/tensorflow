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

#include "tensorflow/core/kernels/tensor_array.h"

namespace tensorflow {

Status TensorArray::LockedWrite(OpKernelContext* ctx, const int32 index,
                                PersistentTensor* value) {
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  size_t index_size = static_cast<size_t>(index);
  if (index < 0 ||
      (!dynamic_size_ && index_size >= tensors_.size())) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<string>()(1), ": Tried to write to index ",
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
  if (t.written) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<string>()(1),
                                   ": Could not write to TensorArray index ",
                                   index,
                                   " because it has already been written to.");
  }
  Tensor* value_t = value->AccessTensor(ctx);
  if (value_t->dtype() != dtype_) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<string>()(1),
        ": Could not write to TensorArray index ", index,
        " because the value dtype is ", DataTypeString(value_t->dtype()),
        " but TensorArray dtype is ", DataTypeString(dtype_), ".");
  }
  t.tensor = *value;
  t.shape = value_t->shape();
  t.written = true;
  return Status::OK();
}

Status TensorArray::LockedRead(const int32 index, PersistentTensor* value) {
  TF_RETURN_IF_ERROR(LockedReturnIfClosed());
  if (index < 0 || static_cast<size_t>(index) >= tensors_.size()) {
    return errors::InvalidArgument("Tried to read from index ", index,
                                   " but array size is: ", tensors_.size());
  }
  TensorAndState& t = tensors_[index];
  if (t.read) {
    return errors::InvalidArgument(
        "TensorArray ", handle_.vec<string>()(1), ": Could not read index ",
        index, " twice because TensorArray a read-once object.");
  }
  if (!t.written) {
    return errors::InvalidArgument("TensorArray ", handle_.vec<string>()(1),
                                   ": Could not read from TensorArray index ",
                                   index,
                                   " because it has not yet been written to.");
  }
  *value = t.tensor;
  t.read = true;
  t.tensor = PersistentTensor();
  return Status::OK();
}

}  // namespace tensorflow
