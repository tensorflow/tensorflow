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

#ifndef TENSORFLOW_CORE_KERNELS_STRIDED_SLICE_OP_H_
#define TENSORFLOW_CORE_KERNELS_STRIDED_SLICE_OP_H_

// Functor definition for StridedSliceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct StridedSlice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    MaybeWith32BitIndexing<Device>(
        [&](auto output32, auto input32, const auto& start_indices32,
            const auto& stop_indices32, const auto& strides32) {
          output32.device(d) =
              input32.stridedSlice(start_indices32, stop_indices32, strides32);
        },
        output, input, start_indices, stop_indices, strides);
  }
};

template <typename T, int NDIMS, typename Device>
struct InitOutput {
  static void run(const Device& d, typename TTypes<T, NDIMS>::Tensor output) {
    output.device(d) = output.constant(T(0));
  }
};

template <int NDIMS, typename Device>
struct InitOutput<ResourceHandle, NDIMS, Device> {
  static void run(const Device& d,
                  typename TTypes<ResourceHandle, NDIMS>::Tensor output) {
    output.device(d) = output.constant(ResourceHandle());
  }
};

template <int NDIMS, typename Device>
struct InitOutput<tstring, NDIMS, Device> {
  static void run(const Device& d,
                  typename TTypes<tstring, NDIMS>::Tensor output) {
    output.device(d) = output.constant(tstring());
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
    MaybeWith32BitIndexing<Device>(
        [&](auto output32, const auto& start_indices32,
            const auto& stop_indices32, const auto& strides32) {
          output32.stridedSlice(start_indices32, stop_indices32, strides32)
              .device(d) = input;
        },
        output, start_indices, stop_indices, strides);
  }
};

template <typename Device, typename T, int NDIMS>
struct StridedSliceAssign {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides,
                  const StridedSliceAssignBCast& bcast) {
    MaybeWith32BitIndexing<Device>(
        [&](auto output32, auto input32, const auto& start_indices32,
            const auto& stop_indices32, const auto& strides32) {
          if (bcast.IsBroadcastingRequired()) {
            output32.stridedSlice(start_indices32, stop_indices32, strides32)
                .device(d) = input32.broadcast(bcast.bcast());
          } else {
            output32.stridedSlice(start_indices32, stop_indices32, strides32)
                .device(d) = input32;
          }
        },
        output, input, start_indices, stop_indices, strides);
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

#endif  // TENSORFLOW_CORE_KERNELS_STRIDED_SLICE_OP_H_
