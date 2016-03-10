/* Copyright 2015 Google Inc. All Rights Reserved.

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

template <typename Device>
struct RGBToHSV {
  void operator()(const Device &d, TTypes<float, 2>::ConstTensor input_data,
                  TTypes<float, 1>::Tensor range,
                  TTypes<float, 2>::Tensor output_data) {
    auto H = output_data.chip<1>(0);
    auto S = output_data.chip<1>(1);
    auto V = output_data.chip<1>(2);

    auto R = input_data.chip<1>(0);
    auto G = input_data.chip<1>(1);
    auto B = input_data.chip<1>(2);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> channel_axis{{1}};
#else
    Eigen::IndexList<Eigen::type2index<1> > channel_axis;
#endif

    V.device(d) = input_data.maximum(channel_axis);

    range.device(d) = V - input_data.minimum(channel_axis);

    S.device(d) = (V > 0.f).select(range / V, V.constant(0.f));

    auto norm = range.inverse() * (1.f / 6.f);
    // TODO(wicke): all these assignments are only necessary because a combined
    // expression is larger than kernel parameter space. A custom kernel is
    // probably in order.
    H.device(d) = (R == V).select(norm * (G - B),
                                  (G == V).select(norm * (B - R) + 2.f / 6.f,
                                                  norm * (R - G) + 4.f / 6.f));
    H.device(d) = (range > 0.f).select(H, H.constant(0.f));
    H.device(d) = (H < 0.f).select(H + 1.f, H);
  }
};

template <typename Device>
struct HSVToRGB {
  void operator()(const Device &d, TTypes<float, 2>::ConstTensor input_data,
                  TTypes<float, 2>::Tensor output_data) {
    auto H = input_data.chip<1>(0);
    auto S = input_data.chip<1>(1);
    auto V = input_data.chip<1>(2);

    // TODO(wicke): compute only the fractional part of H for robustness
    auto dh = H * 6.f;
    auto dr = ((dh - 3.f).abs() - 1.f).cwiseMax(0.f).cwiseMin(1.f);
    auto dg = (-(dh - 2.f).abs() + 2.f).cwiseMax(0.f).cwiseMin(1.f);
    auto db = (-(dh - 4.f).abs() + 2.f).cwiseMax(0.f).cwiseMin(1.f);
    auto one_s = -S + 1.f;

    auto R = output_data.chip<1>(0);
    auto G = output_data.chip<1>(1);
    auto B = output_data.chip<1>(2);

    R.device(d) = (one_s + S * dr) * V;
    G.device(d) = (one_s + S * dg) * V;
    B.device(d) = (one_s + S * db) * V;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_COLORSPACE_OP_H_
