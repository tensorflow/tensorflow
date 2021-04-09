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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

// REGISTER# macros ignore all but first type (assumed to be float) when
// __ANDROID_TYPES_SLIM__ is defined.  Since this file is the second of two
// sharded files, only make its register calls when not __ANDROID_TYPES_SLIM__.
#if !defined(__ANDROID_TYPES_SLIM__)

REGISTER6(BinaryOp, CPU, "NotEqual", functor::not_equal_to, int32, int64,
          complex64, complex128, tstring, bool);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER6(BinaryOp, GPU, "NotEqual", functor::not_equal_to, int8, int16, int64,
          complex64, complex128, bool);
#else
REGISTER2(BinaryOp, GPU, "NotEqual", functor::not_equal_to, complex64,
          complex128);
#endif

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // !defined(__ANDROID_TYPES_SLIM__)
}  // namespace tensorflow
