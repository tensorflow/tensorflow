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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_ADJUST_CONTRAST_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_ADJUST_CONTRAST_OP_H_
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by AdjustContrastOp to do the computations.
template <typename Device, typename T>
struct AdjustContrast {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<float>::ConstScalar contrast_factor,
                  typename TTypes<float>::ConstScalar min_value,
                  typename TTypes<float>::ConstScalar max_value,
                  typename TTypes<float, 4>::Tensor mean_values,
                  typename TTypes<float, 4>::Tensor output) {
    const int batch = input.dimension(0);
    const int height = input.dimension(1);
    const int width = input.dimension(2);
    const int channels = input.dimension(3);

    Eigen::array<int, 4> scalar_broadcast;
    scalar_broadcast[0] = batch;
    scalar_broadcast[1] = height;
    scalar_broadcast[2] = width;
    scalar_broadcast[3] = channels;

    Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >
        reduction_axis;
    Eigen::IndexList<Eigen::type2index<1>, int, int, Eigen::type2index<1> >
        broadcast_dims;
    broadcast_dims.set(1, height);
    broadcast_dims.set(2, width);
    Eigen::IndexList<int, Eigen::type2index<1>, Eigen::type2index<1>, int>
        reshape_dims;
    reshape_dims.set(0, batch);
    reshape_dims.set(3, channels);

    Eigen::Sizes<1, 1, 1, 1> scalar;
    float num_reduced_coeffs = height * width;
    mean_values.device(d) =
        (input.template cast<float>().sum(reduction_axis).eval() /
         num_reduced_coeffs)
            .reshape(reshape_dims)
            .broadcast(broadcast_dims);

    auto contrast_factor_tensor =
        contrast_factor.reshape(scalar).broadcast(scalar_broadcast);
    auto adjusted =
        (input.template cast<float>() - mean_values) * contrast_factor_tensor +
        mean_values;
    auto min_bcast = min_value.reshape(scalar).broadcast(scalar_broadcast);
    auto max_bcast = max_value.reshape(scalar).broadcast(scalar_broadcast);
    // TODO(wicke): This is rather slow and should be re-written as pure cuda.
    output.device(d) = adjusted.cwiseMin(max_bcast).cwiseMax(min_bcast);
  }
};

// Functor used by AdjustContrastOpv2 to do the computations.
template <typename Device, typename T>
struct AdjustContrastv2 {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<float>::ConstScalar contrast_factor,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch = input.dimension(0);
    const int height = input.dimension(1);
    const int width = input.dimension(2);
    const int channels = input.dimension(3);

    Eigen::array<int, 4> scalar_broadcast;
    scalar_broadcast[0] = batch;
    scalar_broadcast[1] = height;
    scalar_broadcast[2] = width;
    scalar_broadcast[3] = channels;

    Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1> >
        reduction_axis;
    Eigen::IndexList<Eigen::type2index<1>, int, int, Eigen::type2index<1> >
        broadcast_dims;
    broadcast_dims.set(1, height);
    broadcast_dims.set(2, width);
    Eigen::IndexList<int, Eigen::type2index<1>, Eigen::type2index<1>, int>
        reshape_dims;
    reshape_dims.set(0, batch);
    reshape_dims.set(3, channels);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2>,
                     Eigen::type2index<0>, Eigen::type2index<3> >
        reduced_dims_first;

    Eigen::Sizes<1, 1, 1, 1> scalar;
    float num_reduced_coeffs = height * width;
    output.device(d) = (input.template cast<float>()
                            .shuffle(reduced_dims_first)
                            .sum(reduction_axis)
                            .eval() /
                        num_reduced_coeffs)
                           .template cast<T>()
                           .reshape(reshape_dims)
                           .broadcast(broadcast_dims);
    auto contrast_factor_tensor =
        contrast_factor.reshape(scalar).broadcast(scalar_broadcast);
    auto adjusted =
        (input - output).template cast<float>() * contrast_factor_tensor;
    output.device(d) += adjusted.template cast<T>();
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_ADJUST_CONTRAST_OP_H_
