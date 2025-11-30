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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_COLORSPACE_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_COLORSPACE_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
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
    // Upcast to double for all calculations, then cast back to T at the end.
    const auto n = input_data.dimension(0);
    for (Eigen::Index i = 0; i < n; ++i) {
      double r = static_cast<double>(input_data(i, 0));
      double g = static_cast<double>(input_data(i, 1));
      double b = static_cast<double>(input_data(i, 2));
      double v = std::max({r, g, b});
      double minc = std::min({r, g, b});
      double rc = v - minc;
      double s = (v > 0.0) ? rc / v : 0.0;
      double h = 0.0;
      if (rc > 0.0) {
        if (v == r) {
          h = (g - b) / rc;
        } else if (v == g) {
          h = 2.0 + (b - r) / rc;
        } else {
          h = 4.0 + (r - g) / rc;
        }
        h /= 6.0;
        if (h < 0.0) h += 1.0;
      }
      output_data(i, 0) = static_cast<T>(h);
      output_data(i, 1) = static_cast<T>(s);
      output_data(i, 2) = static_cast<T>(v);
      range(i) = static_cast<T>(rc);
    }
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

#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_COLORSPACE_OP_H_
