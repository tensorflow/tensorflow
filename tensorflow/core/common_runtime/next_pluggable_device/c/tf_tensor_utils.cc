/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_tensor_utils.h"

#include <stdint.h>

#include <cstring>
#include <memory>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

void CopyTF_TensorToTensor(const TF_Tensor* src, Tensor* dst) {
  // TODO: Convert through a lookup table for better API compatibility.
  DataType dtype = static_cast<DataType>(TF_TensorType(src));
  TensorShape tensor_shape;
  int dim = TF_NumDims(src);
  for (int i = 0; i < dim; ++i) {
    tensor_shape.AddDim(TF_Dim(src, i));
  }
  *dst = Tensor(dtype, tensor_shape);

  std::memcpy(dst->data(), TF_TensorData(src), TF_TensorByteSize(src));
}

TF_Tensor* CopyTensorToTF_Tensor(const Tensor& src) {
  // TODO: Convert through a lookup table for better API compatibility.
  TF_DataType dtype = static_cast<TF_DataType>(src.dtype());
  const TensorShape& shape = src.shape();
  auto dims = std::make_unique<int64_t[]>(shape.dims());
  size_t len = TF_DataTypeSize(dtype);
  for (int i = 0; i < shape.dims(); ++i) {
    dims[i] = shape.dim_size(i);
    len *= dims[i];
  }
  TF_Tensor* tf_tensor =
      TF_AllocateTensor(dtype, dims.get(), shape.dims(), len);
  void* data = TF_TensorData(tf_tensor);
  std::memcpy(data, src.data(), len);
  return tf_tensor;
}

}  // namespace tensorflow
