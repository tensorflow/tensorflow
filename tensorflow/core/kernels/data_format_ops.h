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

#ifndef TENSORFLOW_KERNELS_DATA_FORMAT_OPS_H_
#define TENSORFLOW_KERNELS_DATA_FORMAT_OPS_H_
// Functor definition for data format dim mapping ops, must be compilable
// by nvcc.
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by DataFormatDimMapOP to do the computations.
template <typename Device, typename T>
struct DataFormatDimMap {
  void operator()(const Device& d, typename TTypes<T>::ConstScalar x,
                  typename TTypes<T>::Scalar y) {
    auto zero = x.constant(0);
    auto one = x.constant(1);
    auto three = x.constant(3);
    auto four = x.constant(4);
    auto x_mod = (x + four) % 4;
    auto is_zero = (x_mod == zero);
    auto is_three = (x_mod == three);
    y.device(d) = is_zero.select(zero, is_three.select(one, x_mod + one));
  }
};

template <typename T>
struct VecPermuteNHWCToNCHW {
  Eigen::DSizes<Eigen::DenseIndex, 1> dimensions(
      typename TTypes<T>::ConstVec input) const {
    Eigen::DSizes<Eigen::DenseIndex, 1> result;
    result[0] = input.dimension(0);
    return result;
  }
  template <typename Output, typename Device>
  void eval(typename TTypes<T>::ConstVec input, Output& output,
            const Device& d) const {
    output.template chip<0>(0).device(d) = input.template chip<0>(0);
    output.template chip<0>(1).device(d) = input.template chip<0>(3);
    output.template chip<0>(2).device(d) = input.template chip<0>(1);
    output.template chip<0>(3).device(d) = input.template chip<0>(2);
  }
};

template <typename T>
struct VecPermuteNCHWToNHWC {
  Eigen::DSizes<Eigen::DenseIndex, 1> dimensions(
      typename TTypes<T>::ConstVec input) const {
    Eigen::DSizes<Eigen::DenseIndex, 1> result;
    result[0] = input.dimension(0);
    return result;
  }
  template <typename Output, typename Device>
  void eval(typename TTypes<T>::ConstVec input, Output& output,
            const Device& d) const {
    output.template chip<0>(0).device(d) = input.template chip<0>(0);
    output.template chip<0>(1).device(d) = input.template chip<0>(2);
    output.template chip<0>(2).device(d) = input.template chip<0>(3);
    output.template chip<0>(3).device(d) = input.template chip<0>(1);
  }
};

// Functor used by DataFormatVecPermuteOp to do the computations.
template <typename Device, typename T>
struct DataFormatVecPermute {
  void operator()(const Device& d, typename TTypes<T>::ConstVec x,
                  typename TTypes<T>::Vec y, bool nhwc_to_nchw) {
    if (nhwc_to_nchw) {
      y.device(d) = x.customOp(VecPermuteNHWCToNCHW<T>());
    } else {
      y.device(d) = x.customOp(VecPermuteNCHWToNHWC<T>());
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DATA_FORMAT_OPS_H_
