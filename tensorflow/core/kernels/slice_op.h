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

#ifndef TENSORFLOW_KERNELS_SLICE_OP_H_
#define TENSORFLOW_KERNELS_SLICE_OP_H_

// Functor definition for SliceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

namespace internal {

// Helper to compute 'strides' given a tensor 'shape'. I.e.,
// strides[i] = prod(shape.dim_size[(i+1):])
template <typename T>
gtl::InlinedVector<T, 8> ComputeStride(const TensorShape& shape) {
  const int ndims = shape.dims();
  gtl::InlinedVector<T, 8> strides(ndims);
  T stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<T>(shape.dim_size(i));
  }
  return strides;
}

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices);

template <typename Device, typename T, int NDIMS>
void SliceUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices,
                 const gtl::ArraySlice<int64>& slice_sizes) {

  auto input = in.tensor<T, NDIMS>();
  auto output = out->tensor<T, NDIMS>();
  Eigen::DSizes<int, NDIMS> indices;
  for (int i = 0; i < NDIMS; ++i) {
    indices[i] = slice_indices[i];
  }
  Eigen::DSizes<int, NDIMS> sizes;
  for (int i = 0; i < NDIMS; ++i) {
    sizes[i] = slice_sizes[i];
  }
  bool use_64bit = (input.size() > Eigen::NumTraits<int>::highest());
  if (!use_64bit &&
      Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
  } else {
    output.device(d) = input.slice(indices, sizes);
  }
}

} // namespace internal

namespace functor {

template <typename Device, typename T>
struct Slice {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<int64>& slice_indices,
                  const gtl::ArraySlice<int64>& slice_sizes) {
    switch (in.dims()) {
      case 1:
        internal::SliceUsingEigen<Device, T, 1>(d, out, in, slice_indices, slice_sizes);
        break;
      case 2:
        internal::SliceUsingEigen<Device, T, 2>(d, out, in, slice_indices, slice_sizes);
        break;
      case 3:
        internal::SliceUsingEigen<Device, T, 3>(d, out, in, slice_indices, slice_sizes);
        break;
      case 4:
        internal::SliceUsingEigen<Device, T, 4>(d, out, in, slice_indices, slice_sizes);
        break;
      case 5:
        internal::SliceUsingEigen<Device, T, 5>(d, out, in, slice_indices, slice_sizes);
        break;
      case 6:
        internal::SliceUsingEigen<Device, T, 6>(d, out, in, slice_indices, slice_sizes);
        break;
      case 7:
        internal::SliceUsingEigen<Device, T, 7>(d, out, in, slice_indices, slice_sizes);
        break;
      default:
        internal::SliceSimple<Device, T>(d, out, in, slice_indices);
        break;
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
