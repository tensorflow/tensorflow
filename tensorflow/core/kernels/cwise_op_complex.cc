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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
#define REGISTER_COMPLEX(D, R, C)                         \
  REGISTER_KERNEL_BUILDER(Name("Complex")                 \
                              .Device(DEVICE_##D)         \
                              .TypeConstraint<R>("T")     \
                              .TypeConstraint<C>("Tout"), \
                          BinaryOp<D##Device, functor::make_complex<R>>);

REGISTER_COMPLEX(CPU, float, complex64);
REGISTER_COMPLEX(CPU, double, complex128);

#if GOOGLE_CUDA
REGISTER_COMPLEX(GPU, float, complex64);
REGISTER_COMPLEX(GPU, double, complex128);
#endif

#undef REGISTER_COMPLEX
}  // namespace tensorflow
