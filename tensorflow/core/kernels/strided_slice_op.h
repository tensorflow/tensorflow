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

#ifndef TENSORFLOW_KERNELS_STRIDED_SLICE_OP_H_
#define TENSORFLOW_KERNELS_STRIDED_SLICE_OP_H_

// Functor definition for StridedSliceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct StridedSlice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
    if (!use_64bit &&
        Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      Eigen::DSizes<int, NDIMS> start_i, stop_i, strides_i;
      for (int i = 0; i < NDIMS; ++i) {
        start_i[i] = start_indices[i];
        stop_i[i] = stop_indices[i];
        strides_i[i] = strides[i];
      }
      To32Bit(output).device(d) =
          To32Bit(input).stridedSlice(start_i, stop_i, strides_i);
    } else {
      output.device(d) =
          input.stridedSlice(start_indices, stop_indices, strides);
    }
  }
};

template <typename T, int NDIMS, typename Device>
struct InitOutput {
  static void run(const Device& d, typename TTypes<T, NDIMS>::Tensor output) {
    output.device(d) = output.constant(T(0));
  }
};

template <int NDIMS, typename Device>
struct InitOutput<string, NDIMS, Device> {
  static void run(const Device& d,
                  typename TTypes<string, NDIMS>::Tensor output) {
    output.device(d) = output.constant(string());
  }
};

template <typename Device, typename T, int NDIMS>
struct StridedSliceGrad {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    InitOutput<T, NDIMS, Device>::run(d, output);
    const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
    if (!use_64bit &&
        Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      Eigen::DSizes<int, NDIMS> start_i, stop_i, strides_i;
      for (int i = 0; i < NDIMS; ++i) {
        start_i[i] = start_indices[i];
        stop_i[i] = stop_indices[i];
        strides_i[i] = strides[i];
      }
      To32Bit(output).stridedSlice(start_i, stop_i, strides_i).device(d) =
          input;
    } else {
      output.stridedSlice(start_indices, stop_indices, strides).device(d) =
          input;
    }
  }
};

template <typename Device, typename T, int NDIMS>
struct StridedSliceAssign {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
    if (!use_64bit &&
        Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      Eigen::DSizes<int, NDIMS> start_i, stop_i, strides_i;
      for (int i = 0; i < NDIMS; ++i) {
        start_i[i] = start_indices[i];
        stop_i[i] = stop_indices[i];
        strides_i[i] = strides[i];
      }
      To32Bit(output).stridedSlice(start_i, stop_i, strides_i).device(d) =
          To32Bit(input);
    } else {
      output.stridedSlice(start_indices, stop_indices, strides).device(d) =
          input;
    }
  }
};

template <typename Device, typename T>
struct StridedSliceAssignScalar {
  void operator()(const Device& d, typename TTypes<T, 1>::Tensor output,
                  typename TTypes<T, 1>::ConstTensor input) {
    output.device(d) = input;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
