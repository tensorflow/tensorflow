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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/aggregate_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensor_array {

// Full implementations are in tensor_array.cc
template <typename T, typename Device>
Status TensorArrayWriteOrAdd(OpKernelContext* ctx, Tensor* sum,
                             const Tensor* current, const Tensor* add) {
  return errors::InvalidArgument("TensorArrayWriteOrAdd type not supported: ",
                                 DataTypeString(DataTypeToEnum<T>::value));
};

#define TENSOR_ARRAY_WRITE_OR_ADD(Device, T)                                   \
  template <>                                                                  \
  Status TensorArrayWriteOrAdd<T, Device>(OpKernelContext * ctx, Tensor * sum, \
                                          const Tensor* current,               \
                                          const Tensor* add);

#define TENSOR_ARRAY_WRITE_OR_ADD_CPU(T) TENSOR_ARRAY_WRITE_OR_ADD(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_CPU)
#undef TENSOR_ARRAY_WRITE_OR_ADD_CPU

#if GOOGLE_CUDA

#define TENSOR_ARRAY_WRITE_OR_ADD_GPU(T) TENSOR_ARRAY_WRITE_OR_ADD(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
#undef TENSOR_ARRAY_WRITE_OR_ADD_GPU

#endif  // GOOGLE_CUDA

#undef TENSOR_ARRAY_WRITE_OR_ADD

}  // namespace tensor_array

class TensorArray : public ResourceBase {
 public:
  TensorArray(const DataType& dtype, const Tensor& handle, int32 N)
      : dtype_(dtype), handle_(handle), tensor_array_(N) {}

  Status Write(const int32 index, const PersistentTensor& value) {
    mutex_lock l(mu_);
    if (index < 0 || index >= tensor_array_.size()) {
      return errors::InvalidArgument("Tried to write to index ", index,
                                     " but array size is: ",
                                     tensor_array_.size());
    }
    if (tensor_array_[index].IsInitialized()) {
      return errors::InvalidArgument(
          "Could not write to TensorArray index ", index,
          " because it has already been written to.");
    }
    tensor_array_[index] = value;
    return Status::OK();
  }

  template <typename Device, typename T>
  Status WriteOrAdd(OpKernelContext* ctx, const int32 index,
                    PersistentTensor value) {
    mutex_lock l(mu_);
    if (index < 0 || index >= tensor_array_.size()) {
      return errors::InvalidArgument("Tried to write to index ", index,
                                     " but array size is: ",
                                     tensor_array_.size());
    }
    if (tensor_array_[index].IsInitialized()) {
      Tensor* sum = tensor_array_[index].AccessTensor(ctx);
      const Tensor* current = tensor_array_[index].AccessTensor(ctx);
      const Tensor* add = value.AccessTensor(ctx);
      if (!current->shape().IsSameSize(add->shape())) {
        return errors::InvalidArgument(
            "Cannot add to index ", index, " because shapes are inconsistent: ",
            current->shape().ShortDebugString(), " vs. ",
            add->shape().ShortDebugString());
      }
      return tensor_array::TensorArrayWriteOrAdd<T, Device>(ctx, sum, current,
                                                            add);
    } else {
      tensor_array_[index] = value;
    }
    return Status::OK();
  }

  Status Read(const int32 index, PersistentTensor* value) {
    mutex_lock l(mu_);
    if (index < 0 || index >= tensor_array_.size()) {
      return errors::InvalidArgument("Tried to read from index ", index,
                                     " but array size is: ",
                                     tensor_array_.size());
    }
    if (!tensor_array_[index].IsInitialized()) {
      return errors::InvalidArgument(
          "Could not read from TensorArray index ", index,
          " because it has not yet been written to.");
    }
    *value = tensor_array_[index];
    return Status::OK();
  }

  inline int32 Size() {
    mutex_lock l(mu_);
    return tensor_array_.size();
  }

  DataType ElemType() { return dtype_; }

  string DebugString() override {
    mutex_lock l(mu_);
    return strings::StrCat("TensorArray[", tensor_array_.size(), "]");
  }

  mutex* mu() { return &mu_; }
  Tensor* handle() { return &handle_; }

 private:
  mutex mu_;
  DataType dtype_;
  Tensor handle_;
  std::vector<PersistentTensor> tensor_array_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TENSOR_ARRAY_H_
