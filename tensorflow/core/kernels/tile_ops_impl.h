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

#ifndef TENSORFLOW_KERNELS_TILE_IMPL_OPS_H_
#define TENSORFLOW_KERNELS_TILE_IMPL_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace internal {

template <typename Device, typename T>
void TileSimple(const Device& d, const Tensor& in, Tensor* out,
                const gtl::ArraySlice<int32> broadcast_array) {
  // need to be done
}

template <typename Device, typename T, int NDIMs>
void TileUsingEigen(const Device& d, const Tensor& in, Tensor* out,
                    const gtl::ArraySlice<int32> broadcast_array) {
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  Eigen::array<int32, NDIM> b;
  for (int i = 0; i < NDIMS; ++i) b[i] = broadcast_array[i];
  if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    // Use 32bit indexing to speed up the computations
    To32Bit(y).device(d) = To32Bit(x).broadcast(b);
  } else {
    out.device(d) = in.broadcast(b);
  }
}

} // end namespace internal

namespace functor {

template <typename Device, typename T>
struct Tile {
  void operator()(const Device& d, const Tensor& in, Tensor* out,
                  const gtl::ArraySlice<int32> broadcast_array) const {
    switch (in.dims()) {
      case 0:
        internal::TileUsingEigen<Device, T, 0>(d, in, out, broadcast_array);
        break;
      case 1:
        internal::TileUsingEigen<Device, T, 1>(d, in, out, broadcast_array);
        break;
      case 2:
        internal::TileUsingEigen<Device, T, 2>(d, in, out, broadcast_array);
        break;
      default:
        internal::TileSimple<Device, T>(d, in, out, broadcast_array);
        break;
    }
  }
};

template <typename Device, typename T>
struct Tile<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor out,
                  typename TTypes<T, 0>::ConstTensor in,
                  const Eigen::array<int32, 0>&) const {
    // In the scalar case we simply copy the input.
    out.device(d) = in;
  }
};

template <typename Device, typename T, int NDIM>
struct TileGrad {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes,
                  bool first) const {
    if (first) {
      out.device(d) = in.slice(indices, sizes);
    } else {
      out.device(d) += in.slice(indices, sizes);
    }
  }
};

template <typename Device, typename T>
struct TileGrad<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor out,
                  typename TTypes<T, 0>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&,
                  bool first) const {
    if (first) {
      out.device(d) = in;
    } else {
      out.device(d) += in;
    }
  }
};

template <typename Device, typename T, int NDIM, int REDUCEDNDIM>
struct ReduceAndReshape {
  void operator()(
      const Device& d, typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<T, NDIM>::ConstTensor in,
      const Eigen::DSizes<Eigen::DenseIndex, REDUCEDNDIM>& reduce_dim,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& reshape_dim) const {
    out.device(d) = in.sum(reduce_dim).reshape(reshape_dim);
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TILE_OPS_IMPL_H_
