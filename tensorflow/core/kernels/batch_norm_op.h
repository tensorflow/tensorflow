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

#ifndef TENSORFLOW_KERNELS_BATCH_NORM_OP_H_
#define TENSORFLOW_KERNELS_BATCH_NORM_OP_H_
// Functor definition for BatchNormOp, must be compilable by nvcc.
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by BatchNormOp to do the computations.
template <typename Device, typename T>
struct BatchNorm {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T>::ConstVec mean,
                  typename TTypes<T>::ConstVec var,
                  typename TTypes<T>::ConstVec beta,
                  typename TTypes<T>::ConstVec gamma, T variance_epsilon,
                  bool scale_after_normalization,
                  typename TTypes<T, 4>::Tensor output) {
    const int depth = mean.dimension(0);
    const int rest_size = input.size() / depth;

    Eigen::DSizes<int, 2> rest_by_depth(rest_size, depth);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<int, 2> rest_by_one(rest_size, 1);
    Eigen::DSizes<int, 2> one_by_depth(1, depth);
    Eigen::DSizes<int, 2> depth_by_one(depth, 1);
#else
    Eigen::IndexList<int, Eigen::type2index<1> > rest_by_one;
    rest_by_one.set(0, rest_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<int, Eigen::type2index<1> > depth_by_one;
    depth_by_one.set(0, depth);
#endif
    if (scale_after_normalization) {
      output.reshape(rest_by_depth).device(d) =
          (input.reshape(rest_by_depth) -
           mean.reshape(one_by_depth).broadcast(rest_by_one)) *
              ((var + var.constant(variance_epsilon)).rsqrt() * gamma)
                  .eval()
                  .reshape(one_by_depth)
                  .broadcast(rest_by_one) +
          beta.reshape(one_by_depth).broadcast(rest_by_one);
    } else {
      output.reshape(rest_by_depth).device(d) =
          (input.reshape(rest_by_depth) -
           mean.reshape(one_by_depth).broadcast(rest_by_one)) *
              ((var + var.constant(variance_epsilon)).rsqrt())
                  .eval()
                  .reshape(one_by_depth)
                  .broadcast(rest_by_one) +
          beta.reshape(one_by_depth).broadcast(rest_by_one);
    }
  }
};

template <typename Device, typename T>
struct BatchNormGrad {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T>::ConstVec mean,
                  typename TTypes<T>::ConstVec var,
                  typename TTypes<T>::ConstVec gamma,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  T variance_epsilon, bool scale_after_normalization,
                  typename TTypes<T, 4>::Tensor dx, typename TTypes<T>::Vec dm,
                  typename TTypes<T>::Vec dv, typename TTypes<T>::Vec db,
                  typename TTypes<T>::Vec dg, typename TTypes<T>::Vec scratch1,
                  typename TTypes<T>::Vec scratch2) {
    const int depth = mean.dimension(0);
    const int rest_size = input.size() / depth;

    typedef typename TTypes<T>::ConstVec::Index Index;

    Eigen::DSizes<Index, 2> rest_by_depth(rest_size, depth);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Index, 2> rest_by_one(rest_size, 1);
    Eigen::DSizes<Index, 2> one_by_depth(1, depth);
    Eigen::array<Index, 1> reduction_axis;
    reduction_axis[0] = 0;  // Reduces on first dimension.
#else
    Eigen::IndexList<Index, Eigen::type2index<1> > rest_by_one;
    rest_by_one.set(0, rest_size);
    Eigen::IndexList<Eigen::type2index<1>, Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0> > reduction_axis;
#endif

    // db = out_backprop
    //
    // dg = out_backprop * ((x - m) * rsqrt(v + epsilon))
    //
    // dv = sum_over_rest(out_backprop * gamma * (x - m)) *
    //      (-1/2) * (v + epsilon) ^ (-3/2)
    //
    // dm = sum_over_rest(out_backprop * gamma) * (-1 / rsqrt(v + epsilon))
    //
    // dx = out_backprop * (gamma * rsqrt(v + epsilon))
    db.device(d) = out_backprop.reshape(rest_by_depth).sum(reduction_axis);

    // scratch1 = rsqrt(v + epsilon)
    scratch1.device(d) = (var + var.constant(variance_epsilon)).rsqrt();

    // scratch2 = sum_over_rest(out_backprop * (x - m))
    scratch2.device(d) = (out_backprop.reshape(rest_by_depth) *
                          (input.reshape(rest_by_depth) -
                           mean.reshape(one_by_depth).broadcast(rest_by_one)))
                             .sum(reduction_axis);

    if (scale_after_normalization) {
      dx.reshape(rest_by_depth).device(d) =
          out_backprop.reshape(rest_by_depth) * ((scratch1 * gamma)
                                                     .eval()
                                                     .reshape(one_by_depth)
                                                     .broadcast(rest_by_one));
      dm.device(d) = -db * (scratch1 * gamma).eval();
      dg.device(d) = scratch2 * scratch1;
    } else {
      dx.reshape(rest_by_depth).device(d) =
          out_backprop.reshape(rest_by_depth) *
          scratch1.reshape(one_by_depth).broadcast(rest_by_one);
      dm.device(d) = -db * scratch1;
      dg.device(d) = dg.constant(static_cast<T>(0.0));  // Gamma is not learned.
    }

    // scratch1 = - 1/2 * (var + epsilon) ^ (-3/2)
    scratch1.device(d) = scratch1 * scratch1.constant(static_cast<T>(-0.5f)) /
                         (var + var.constant(variance_epsilon));

    if (scale_after_normalization) {
      dv.device(d) = scratch2 * (scratch1 * gamma).eval();
    } else {
      dv.device(d) = scratch2 * scratch1;
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_BATCH_NORM_OP_H_
