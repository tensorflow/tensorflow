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

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices,
                 const gtl::ArraySlice<int64>& slice_sizes);

template <typename Device, typename T, int NDIMS>
void SliceUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices,
                 const gtl::ArraySlice<int64>& slice_sizes) {
    bool use_64bit = (input.size() > Eigen::NumTraits<int>::highest());
    if (!use_64bit &&
        Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      typename TTypes<T, NDIMS>::ConstTensor input = in.tensor<T, NDIMS>();
      typename TTypes<T, NDIMS>::Tensor output = out->tensor<T, NDIMS>();
      Eigen::DSizes<int, NDIMS> indices;
      for (int i = 0; i < NDIMS; ++i) {
        indices[i] = slice_indices[i];
      }
      Eigen::DSizes<int, NDIMS> sizes;
      for (int i = 0; i < NDIMS; ++i) {
        sizes[i] = slice_sizes[i];
      }
      To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
    } else {
      output.device(d) = input.slice(slice_indices, slice_sizes);
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
        internal::SliceUsingEigen<d, T, 1>(d, out, in, slice_indices, slice_sizes);
        break;
      case 2:
        internal::SliceUsingEigen<d, T, 2>(d, out, in, slice_indices, slice_sizes);
        break;
      case 3:
        internal::SliceUsingEigen<d, T, 3>(d, out, in, slice_indices, slice_sizes);
        break;
      case 4:
        internal::SliceUsingEigen<d, T, 4>(d, out, in, slice_indices, slice_sizes);
        break;
      case 5:
        internal::SliceUsingEigen<d, T, 5>(d, out, in, slice_indices, slice_sizes);
        break;
      case 6:
        internal::SliceUsingEigen<d, T, 6>(d, out, in, slice_indices, slice_sizes);
        break;
      case 7:
        internal::SliceUsingEigen<d, T, 7>(d, out, in, slice_indices, slice_sizes);
        break;
      default:
        internal::SliceSimple<d, T>(d, out, in, slice_indices);
        break;
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
