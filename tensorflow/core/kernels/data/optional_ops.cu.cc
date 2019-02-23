/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/data/optional_ops.h"

#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {
namespace data {

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_GPU, OptionalVariant,
                                         OptionalZerosLike<GPUDevice>);

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_GPU,
                                          OptionalVariant,
                                          OptionalBinaryAdd<GPUDevice>);

}  // namespace data
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
