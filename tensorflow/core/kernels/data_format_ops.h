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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_FORMAT_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_FORMAT_OPS_H_
// Functor definition for data format dim mapping ops, must be compilable
// by nvcc.
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by DataFormatDimMapOP to do the computations.
template <typename Device, typename T>
struct DataFormatDimMap {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::Flat y, const TTypes<int>::Vec dst) {
    if (dst.size() == 4) {
      auto zero = x.constant(0);
      auto one = x.constant(1);
      auto two = x.constant(2);

      auto f_zero = x.constant(dst(0));
      auto f_one = x.constant(dst(1));
      auto f_two = x.constant(dst(2));
      auto f_three = x.constant(dst(3));

      auto four = x.constant(4);
      auto x_mod = (x + four) % 4;

      auto is_zero = (x_mod == zero);
      auto is_one = (x_mod == one);
      auto is_two = (x_mod == two);

      y.device(d) = is_zero.select(
          f_zero, is_one.select(f_one, is_two.select(f_two, f_three)));
    } else {
      auto zero = x.constant(0);
      auto one = x.constant(1);
      auto two = x.constant(2);
      auto three = x.constant(3);

      auto f_zero = x.constant(dst(0));
      auto f_one = x.constant(dst(1));
      auto f_two = x.constant(dst(2));
      auto f_three = x.constant(dst(3));
      auto f_four = x.constant(dst(4));

      auto five = x.constant(5);
      auto x_mod = (x + five) % 5;

      auto is_zero = (x_mod == zero);
      auto is_one = (x_mod == one);
      auto is_two = (x_mod == two);
      auto is_three = (x_mod == three);

      y.device(d) = is_zero.select(
          f_zero,
          is_one.select(
              f_one, is_two.select(f_two, is_three.select(f_three, f_four))));
    }
  }
};

template <typename T>
struct VecPermute {
  VecPermute(const Eigen::DSizes<Eigen::DenseIndex, 8>& dst) : dst_(dst) {}
  Eigen::DSizes<Eigen::DenseIndex, 1> dimensions(
      typename TTypes<T>::ConstFlat input) const {
    Eigen::DSizes<Eigen::DenseIndex, 1> result;
    result[0] = input.dimension(0);
    return result;
  }
  template <typename Output, typename Device>
  void eval(typename TTypes<T>::ConstFlat input, Output& output,
            const Device& d) const {
    for (int i = 0; i < input.size(); ++i) {
      output.template chip<0>(dst_[i]).device(d) = input.template chip<0>(i);
    }
  }

 private:
  Eigen::DSizes<Eigen::DenseIndex, 8> dst_;
};

// Functor used by DataFormatVecPermuteOp to do the computations.
template <typename Device, typename T>
struct DataFormatVecPermute {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::Flat y,
                  const Eigen::DSizes<Eigen::DenseIndex, 8>& dst) {
    y.device(d) = x.customOp(VecPermute<T>(dst));
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_FORMAT_OPS_H_
