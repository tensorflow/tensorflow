/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

struct TFTensorDeleter {
  void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
};
using OwnedTFTensor = std::unique_ptr<TF_Tensor, TFTensorDeleter>;

// Moves one ref on HostBuffer to tensorflow::Tensor.
tensorflow::Tensor MoveHostBufferToTfTensor(
    tfrt::RCReference<tfrt::HostBuffer> host_buffer, tfrt::DType dtype,
    const tfrt::TensorShape& shape);

// Creates a tensorflow::Tensor based on StringHostTensor.
tensorflow::Tensor CopyShtToTfTensor(const tfrt::StringHostTensor& sht);

// Converts tfrt shape to tensorflow shape.
inline tensorflow::TensorShape GetTfShape(const tfrt::TensorShape& shape) {
  llvm::SmallVector<tfrt::Index, 4> dimensions;
  shape.GetDimensions(&dimensions);
  llvm::SmallVector<int64_t, 4> dims(dimensions.begin(), dimensions.end());
  return tensorflow::TensorShape(dims);
}

// Retrieves TFRT TensorMetadata from a tensorflow::Tensor.
inline tfrt::TensorMetadata GetTensorMetadata(
    const tensorflow::Tensor& tf_tensor) {
  auto dtype = tfd::GetTfrtDtype(tf_tensor.dtype());
  auto dim_sizes = tf_tensor.shape().dim_sizes();
  static_assert(sizeof(tfrt::Index) == sizeof(dim_sizes.front()),
                "Invalid dimension type size");
  auto shape = llvm::makeArrayRef(
      reinterpret_cast<tfrt::Index*>(dim_sizes.data()), dim_sizes.size());
  return tfrt::TensorMetadata(dtype, shape);
}

inline void CheckBoolCompatibility() {
  // sizeof(bool) is implementation defined. The following may only work when
  // sizeof(bool) is 1.
  //
  // TODO(tfrt-devs): It is still undefined behavior to directly cast char*
  // between bool* and access the data. Consider allocating target objects and
  // using memcpy instead.
  static_assert(sizeof(bool) == 1, "Only support when bool is 1 byte.");
}

}  // namespace tfd
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
