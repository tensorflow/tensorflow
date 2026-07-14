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

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_METADATA_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_METADATA_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace tensorflow::tfd {

// Retrieves TFRT TensorMetadata from a tensorflow::Tensor.
inline tfrt::TensorMetadata GetTensorMetadata(
    const tensorflow::Tensor& tf_tensor) {
  auto dtype = tfd::GetTfrtDtype(tf_tensor.dtype());
  auto dim_sizes = tf_tensor.shape().dim_sizes();
  static_assert(sizeof(tfrt::Index) == sizeof(dim_sizes.front()),
                "Invalid dimension type size");
  auto shape = llvm::ArrayRef(reinterpret_cast<tfrt::Index*>(dim_sizes.data()),
                              dim_sizes.size());
  return tfrt::TensorMetadata(dtype, shape);
}

}  // namespace tensorflow::tfd

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_METADATA_H_
