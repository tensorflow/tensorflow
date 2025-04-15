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

#ifndef TENSORFLOW_CORE_KERNELS_ARGMAX_OP_H_
#define TENSORFLOW_CORE_KERNELS_ARGMAX_OP_H_
// Generator definition for ArgMaxOp, must be compilable by nvcc.

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename Tout>
struct ArgMax {
#define DECLARE_COMPUTE_SPEC(Dims)                                             \
  EIGEN_ALWAYS_INLINE static void Reduce##Dims(                                \
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,            \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output) { \
    output.device(d) = input.argmax(dimension).template cast<Tout>();          \
  }

  DECLARE_COMPUTE_SPEC(1);
  DECLARE_COMPUTE_SPEC(2);
  DECLARE_COMPUTE_SPEC(3);
  DECLARE_COMPUTE_SPEC(4);
  DECLARE_COMPUTE_SPEC(5);
  DECLARE_COMPUTE_SPEC(6);
  DECLARE_COMPUTE_SPEC(7);

#undef DECLARE_COMPUTE_SPEC
};

template <typename Device, typename T, typename Tout>
struct ArgMin {
#define DECLARE_COMPUTE_SPEC(Dims)                                             \
  EIGEN_ALWAYS_INLINE static void Reduce##Dims(                                \
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,            \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output) { \
    output.device(d) = input.argmin(dimension).template cast<Tout>();          \
  }

  DECLARE_COMPUTE_SPEC(1);
  DECLARE_COMPUTE_SPEC(2);
  DECLARE_COMPUTE_SPEC(3);
  DECLARE_COMPUTE_SPEC(4);
  DECLARE_COMPUTE_SPEC(5);
  DECLARE_COMPUTE_SPEC(6);
  DECLARE_COMPUTE_SPEC(7);

#undef DECLARE_COMPUTE_SPEC
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ARGMAX_OP_H_
