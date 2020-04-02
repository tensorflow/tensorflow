/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

namespace tensorflow {

constexpr const char CSRSparseMatrix::kTypeName[];

// Register variant decoding function for TF's RPC.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CSRSparseMatrix,
                                       CSRSparseMatrix::kTypeName);

#define REGISTER_CSR_COPY(DIRECTION)                    \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      CSRSparseMatrix, DIRECTION, CSRSparseMatrix::DeviceCopy)

REGISTER_CSR_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_CSR_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_CSR_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

#undef REGISTER_CSR_COPY

}  // namespace tensorflow
