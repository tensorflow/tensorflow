/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TILE_OPS_GPU_IMPL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TILE_OPS_GPU_IMPL_H_

// Header used to split up compilation of GPU tile ops.  For each type you want
// to have tile ops, create a .cu.cc file containing
//
//   #if GOOGLE_CUDA
//   #include "tensorflow/core/kernels/tile_ops_gpu_impl.h"
//   DEFINE_TILE_OPS(NDIM)
//   #endif  // GOGLE_CUDA
//
// where NDIM is an integer.
//
// NOTE(keveman): Eigen's int8 and string versions don't compile yet with nvcc.

#ifndef GOOGLE_CUDA
#error "This header must be included inside #ifdef GOOGLE_CUDA"
#endif

#define EIGEN_USE_GPU

#include <stdio.h>
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/kernels/tile_ops_impl.h"

#define DEFINE_DIM(T, NDIM)                            \
  template struct Tile<Eigen::GpuDevice, T, NDIM>;     \
  template struct TileGrad<Eigen::GpuDevice, T, NDIM>; \
  template struct ReduceAndReshape<Eigen::GpuDevice, T, NDIM, 1>;

#define DEFINE_TILE_OPS(NDIM)   \
  namespace tensorflow {        \
  namespace functor {           \
  DEFINE_DIM(int16, NDIM)       \
  DEFINE_DIM(int32, NDIM)       \
  DEFINE_DIM(int64, NDIM)       \
  DEFINE_DIM(Eigen::half, NDIM) \
  DEFINE_DIM(float, NDIM)       \
  DEFINE_DIM(double, NDIM)      \
  DEFINE_DIM(complex64, NDIM)   \
  DEFINE_DIM(complex128, NDIM)  \
  }                             \
  }

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TILE_OPS_GPU_IMPL_H_
