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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/kernels/transpose_op_functor.h"

namespace tensorflow {
namespace functor {

template <typename T, int NDIMS>
struct TransposeFunctor<Eigen::GpuDevice, T, NDIMS> {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, NDIMS>::Tensor out,
                  typename TTypes<T, NDIMS>::ConstTensor in, const int* perm) {
    Transpose<Eigen::GpuDevice, T, NDIMS>(d, out, in, perm);
  }
};

#define DEFINE(T, N) template struct TransposeFunctor<Eigen::GpuDevice, T, N>;
#define DEFINE_DIM(T) \
  DEFINE(T, 1);       \
  DEFINE(T, 2);       \
  DEFINE(T, 3);       \
  DEFINE(T, 4);       \
  DEFINE(T, 5);       \
  DEFINE(T, 6);       \
  DEFINE(T, 7);       \
  DEFINE(T, 8);
DEFINE_DIM(uint8);
DEFINE_DIM(int8);
DEFINE_DIM(int16);
DEFINE_DIM(int32);
DEFINE_DIM(int64);
DEFINE_DIM(float);
DEFINE_DIM(double);
#undef DEFINE_DIM
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
