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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

template <typename T>
void MemCpy (T* dst, const T* src, size_t n) {
  if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (size_t k = 0; k < n; ++k) {
      *dst++ = *src++;
    }
  }
};

template <typename T>
T* TileBlock(const std::vector<int>& dim_stat, int d, int ndims,
             const TensorShape& in_shape, const TensorShape& out_shape,
             const T* src, T* dst, const gtl::ArraySlice<int32>& multiples_array) {
  int64 block_size= 1;
  for (int i = ndims - 1; i > d; --i) {
    block_size *= out_shape.dim_size(i);
  }
  block_size *= in_shape.dim_size(d);
  std::cout << "blocksize: " << block_size << std::endl;
  int64 block_num = 0;
  int64 product = 1;
  for (int i = d; i > 0; --i) {
    product *= multiples_array[i];
    block_num += dim_stat[i - 1] * product;
  }
  std::cout << "blocknum: " << block_num << std::endl;
  src += block_num * block_size;
  for (int32 i = 1; i < multiples_array[d]; ++i) {
    MemCpy(dst, src, block_size);
    dst += block_size;
  }
  return dst;
}

template <typename T>
T* CopyFromSrc(const std::vector<int>& dim_stat, int d,
               const TensorShape& in_shape, const T* src, T* dst, int32 multiple) {
  for (int i = 0; i < multiple; ++i) {
    MemCpy(dst, src, in_shape.dim_size(d + 1));
    dst += in_shape.dim_size(d + 1);
  }
  return dst;
}

} // namespace

namespace internal {

template <typename Device, typename T>
void TileSimple(const Device& device, Tensor* out, const Tensor& in,
                const gtl::ArraySlice<int32>& multiples_array) {
  const int ndims = in.dims();
  const T* src = reinterpret_cast<const T*>(in.tensor_data().data());
  T* dst = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));
  std::vector<int> dim_stat = std::vector<int>(ndims - 1, 0);
  if (ndims == 1) {
    CopyFromSrc<T>(dim_stat, -1, in.shape(), src, dst, multiples_array[0]);
  } else {
    T* p = dst;
    bool is_end = false;
    while (!is_end) {
      int d = ndims - 2;
      p = CopyFromSrc<T>(dim_stat, d, in.shape(), src, p, multiples_array[d + 1]);
      src += in.dim_size(d + 1);
      dim_stat[d]++;
      std::cout << "d: " << d << " stat: ";
      for (int i = 0; i < dim_stat.size(); ++i) {std::cout << dim_stat[i] << " ";};
      std::cout << std::endl;
      while (dim_stat[d] == in.dim_size(d)) {
        p = TileBlock<T>(dim_stat, d, ndims, in.shape(), out->shape(),
                      dst, p, multiples_array);
        if (d == 0) {
          is_end = true;
          break;
        } else {
          dim_stat[d] = 0;
          dim_stat[d-1]++;
          --d;
          std::cout << "tileblock d: " << d << " stat: ";
          for (int i = 0; i < dim_stat.size(); ++i) {std::cout << dim_stat[i] << " ";};
          std::cout << std::endl;
        }
      }
    }
    for (int i = 0; i < out->NumElements(); ++i) {
      std::cout << *dst++ << " ";
    }
    std::cout << std::endl;
  }
}

template <typename Device, typename T, int NDIM>
void TileUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                    const gtl::ArraySlice<int32>& broadcast_array) {
  auto x = typename TTypes<T, NDIM>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIM>());
  auto y = typename TTypes<T, NDIM>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIM>());
  Eigen::array<int32, NDIM> b;
  for (int i = 0; i < NDIM; ++i) b[i] = broadcast_array[i];
  if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    // Use 32bit indexing to speed up the computations
    To32Bit(y).device(d) = To32Bit(x).broadcast(b);
  } else {
    y.device(d) = x.broadcast(b);
  }
}

template <typename Device, typename T>
void TileUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                    const gtl::ArraySlice<int32>&) {
  auto x = typename TTypes<T, 0>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<0>());
  auto y = typename TTypes<T, 0>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<0>());
  // In the scalar case we simply copy the input.
  y.device(d) = x;
}

} // end namespace internal

namespace functor {

template <typename Device, typename T>
struct Tile {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<int32> broadcast_array) const {
    switch (in.dims()) {
      case 0:
        internal::TileUsingEigen<Device, T>(d, out, in, broadcast_array);
        break;
      case 1:
        internal::TileUsingEigen<Device, T, 1>(d, out, in, broadcast_array);
        break;
      case 2:
        internal::TileUsingEigen<Device, T, 2>(d, out, in, broadcast_array);
        break;
      case 3:
        internal::TileUsingEigen<Device, T, 3>(d, out, in, broadcast_array);
        break;
      case 4:
        internal::TileUsingEigen<Device, T, 4>(d, out, in, broadcast_array);
        break;
      case 5:
        internal::TileUsingEigen<Device, T, 5>(d, out, in, broadcast_array);
        break;
      case 6:
        internal::TileUsingEigen<Device, T, 6>(d, out, in, broadcast_array);
        break;
      case 7:
        internal::TileUsingEigen<Device, T, 7>(d, out, in, broadcast_array);
        break;
      case 8:
        internal::TileUsingEigen<Device, T, 8>(d, out, in, broadcast_array);
        break;
      default:
        internal::TileSimple<Device, T>(d, out, in, broadcast_array);
        break;
    }
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
