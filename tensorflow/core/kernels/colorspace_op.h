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

#ifndef TENSORFLOW_KERNELS_COLORSPACE_OP_H_
#define TENSORFLOW_KERNELS_COLORSPACE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct RGBToHSV {
  void operator()(const Device &d,
                  typename TTypes<T, 2>::ConstTensor input_data,
                  typename TTypes<T, 1>::Tensor range,
                  typename TTypes<T, 2>::Tensor output_data) {
    auto H = output_data.template chip<1>(0);
    auto S = output_data.template chip<1>(1);
    auto V = output_data.template chip<1>(2);

    auto R = input_data.template chip<1>(0);
    auto G = input_data.template chip<1>(1);
    auto B = input_data.template chip<1>(2);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> channel_axis{{1}};
#else
    Eigen::IndexList<Eigen::type2index<1> > channel_axis;
#endif

    V.device(d) = input_data.maximum(channel_axis);

    range.device(d) = V - input_data.minimum(channel_axis);

    S.device(d) = (V > T(0)).select(range / V, V.constant(T(0)));

    auto norm = range.inverse() * (T(1) / T(6));
    // TODO(wicke): all these assignments are only necessary because a combined
    // expression is larger than kernel parameter space. A custom kernel is
    // probably in order.
    H.device(d) = (R == V).select(norm * (G - B),
                                  (G == V).select(
                                      norm * (B - R) + T(2) / T(6),
                                      norm * (R - G) + T(4) / T(6)));
    H.device(d) = (range > T(0)).select(H, H.constant(T(0)));
    H.device(d) = (H < T(0)).select(H + T(1), H);
  }
};

template <typename Device, typename T>
struct HSVToRGB {
  void operator()(const Device &d,
                  typename TTypes<T, 2>::ConstTensor input_data,
                  typename TTypes<T, 2>::Tensor output_data) {
    auto H = input_data.template chip<1>(0);
    auto S = input_data.template chip<1>(1);
    auto V = input_data.template chip<1>(2);

    // TODO(wicke): compute only the fractional part of H for robustness
    auto dh = H * T(6);
    auto dr = ((dh - T(3)).abs() - T(1)).cwiseMax(T(0)).cwiseMin(T(1));
    auto dg = (-(dh - T(2)).abs() + T(2)).cwiseMax(T(0)).cwiseMin(T(1));
    auto db = (-(dh - T(4)).abs() + T(2)).cwiseMax(T(0)).cwiseMin(T(1));
    auto one_s = -S + T(1);

    auto R = output_data.template chip<1>(0);
    auto G = output_data.template chip<1>(1);
    auto B = output_data.template chip<1>(2);

    R.device(d) = (one_s + S * dr) * V;
    G.device(d) = (one_s + S * dg) * V;
    B.device(d) = (one_s + S * db) * V;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_COLORSPACE_OP_H_
