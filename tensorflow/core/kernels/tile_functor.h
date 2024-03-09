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

#ifndef TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace internal {

// Device-specific naive implementation for Tile.

template <typename T>
void TileSimple(const Eigen::ThreadPoolDevice& d, Tensor* out,
                const Tensor& in);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
void TileSimple(const Eigen::GpuDevice& d, Tensor* out, const Tensor& in);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T, typename Tmultiples, int NDIM>
void TileUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                    const gtl::ArraySlice<Tmultiples> broadcast_array) {
  Eigen::array<Tmultiples, NDIM> b;
  for (int i = 0; i < NDIM; ++i) b[i] = broadcast_array[i];
  MaybeWith32BitIndexing<Device>(
      [&](auto out32, auto in32) { out32.device(d) = in32.broadcast(b); },
      out->tensor<T, NDIM>(), in.tensor<T, NDIM>());
}

template <typename Device, typename T, typename Tmultiples>
void TileUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                    const gtl::ArraySlice<Tmultiples>) {
  auto x = in.tensor<T, 0>();
  auto y = out->tensor<T, 0>();
  // In the scalar case we simply copy the input.
  y.device(d) = x;
}

}  // end namespace internal

namespace functor {

template <typename Device, typename T, typename Tmultiples>
struct Tile {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<Tmultiples> broadcast_array) const {
    switch (in.dims()) {
      case 0:
        internal::TileUsingEigen<Device, T, Tmultiples>(d, out, in,
                                                        broadcast_array);
        break;
      case 1:
        internal::TileUsingEigen<Device, T, Tmultiples, 1>(d, out, in,
                                                           broadcast_array);
        break;
      case 2:
        internal::TileUsingEigen<Device, T, Tmultiples, 2>(d, out, in,
                                                           broadcast_array);
        break;
      case 3:
        internal::TileUsingEigen<Device, T, Tmultiples, 3>(d, out, in,
                                                           broadcast_array);
        break;
      case 4:
        internal::TileUsingEigen<Device, T, Tmultiples, 4>(d, out, in,
                                                           broadcast_array);
        break;
      case 5:
        internal::TileUsingEigen<Device, T, Tmultiples, 5>(d, out, in,
                                                           broadcast_array);
        break;
      case 6:
        internal::TileUsingEigen<Device, T, Tmultiples, 6>(d, out, in,
                                                           broadcast_array);
        break;
      case 7:
        internal::TileUsingEigen<Device, T, Tmultiples, 7>(d, out, in,
                                                           broadcast_array);
        break;
      default:
        internal::TileSimple<T>(d, out, in);
        break;
    }
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_H_
