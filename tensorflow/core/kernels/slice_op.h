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
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {

namespace internal {

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices);
template <typename Device, typename T>
void SliceSimpleGpu(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices);

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices) {
  const int ndims = in.dims();
  const int64 nelem = out->NumElements();
  const gtl::InlinedVector<int64, 8> in_strides = ComputeStride<int64>(in.shape());
  const gtl::InlinedVector<int64, 8> out_strides = ComputeStride<int64>(out->shape());
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();

  std::vector<int64> i_idx(nelem, 0);
  std::vector<int64> t(nelem, 0);

  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    t[o_idx] = o_idx;
  }
  for (int i = 0; i < ndims; ++i) {
    int64 n = (nelem + 7) / 8;
    int64 o_idx = 0;
    switch (nelem % 8) {
#define CALC_INPUT_IDX                                                            \
  i_idx[o_idx] += (t[o_idx] / out_strides[i] + slice_indices[i]) * in_strides[i]; \
  t[o_idx] %= out_strides[i];                                                     \
  ++o_idx;
      case 0: do { CALC_INPUT_IDX;
      case 7:      CALC_INPUT_IDX;
      case 6:      CALC_INPUT_IDX;
      case 5:      CALC_INPUT_IDX;
      case 4:      CALC_INPUT_IDX;
      case 3:      CALC_INPUT_IDX;
      case 2:      CALC_INPUT_IDX;
      case 1:      CALC_INPUT_IDX;
#undef CALC_INPUT_IDX
              } while (--n > 0);
    }
  }
  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    q[o_idx] = p[i_idx[o_idx]];
  }
}

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
  const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
  if (!use_64bit &&
      Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
  } else {
    output.device(d) = input.slice(indices, sizes);
  }
}

} // namespace internal

namespace functor {

// Template parameter NDIM is not neccesary here. The aim of keeping it
// is to compile struct slice separately which minimizes the compiling time.
template <typename Device, typename T, int NDIM>
struct Slice {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<int64>& slice_indices,
                  const gtl::ArraySlice<int64>& slice_sizes) {
    if (in.dims() == NDIM) {
        internal::SliceUsingEigen<Device, T, NDIM>(d, out, in, slice_indices, slice_sizes);
    } else {
        if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
          internal::SliceSimpleGpu<Device, T>(d, out, in, slice_indices);
        } else {
          internal::SliceSimple<Device, T>(d, out, in, slice_indices);
        }
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
