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

#include "tensorflow/core/kernels/tile_ops.h"
#include <stdio.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_TYPE(T) \
  DEFINE_DIM(T, 1)     \
  DEFINE_DIM(T, 2)     \
  DEFINE_DIM(T, 3)     \
  DEFINE_DIM(T, 4)     \
  DEFINE_DIM(T, 5)

#define DEFINE_DIM(T, NDIM)                     \
  template struct Tile<GPUDevice, T, NDIM>;     \
  template struct TileGrad<GPUDevice, T, NDIM>; \
  template struct ReduceAndReshape<GPUDevice, T, NDIM, 1>;

DEFINE_TYPE(float)
DEFINE_TYPE(double)
DEFINE_TYPE(int64)
DEFINE_TYPE(int32)
DEFINE_TYPE(int16)
// NOTE(keveman): Eigen's int8 and string versions don't compile yet with nvcc.

#undef DEFINE_DIM
#undef DEFINE_TYPE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
