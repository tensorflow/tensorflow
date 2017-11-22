/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_FUSED_BATCH_NORM_OP_H_
#define TENSORFLOW_KERNELS_FUSED_BATCH_NORM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

#if GOOGLE_CUDA

// There is a behavior difference between cuDNN v4 and v5 with regard to the
// scaling factor for function cudnnBatchNormalizationForwardInference.
// This function corrects the scaling factor if cuDNN v4 is used, so that
// this behavior inconsistency is hidden from TensorFlow users.
// Details: in cuDNN v4, y = bnScale * (x - mean) * variance + bnBias;
// in v5, y = bnScale * (x - mean) / sqrt(variance + epsilon) + bnBias
// The template is instantiated with T as float in batch_norm_ops.cu.cc; for
// other types, the instantiation needs to be added accordingly.
template <class T>
struct VarianceToInvVariance {
  void operator()(const Eigen::GpuDevice& d, const T* variance, double epsilon,
                  int channels, T* inv_variance);
};

// This function converts the inverted variance of the cuDNN forward training
// output to variance for TensorFlow to calculate the running variance.
// The template is instantiated with T as float in batch_norm_ops.cu.cc; for
// other types, the instantiation needs to be added accordingly.
template <class T>
struct InvVarianceToVariance {
  void operator()(const Eigen::GpuDevice& d, double epsilon, int sample_size,
                  int channels, T* variance);
};

#endif  // GOOGLE_CUDA

// Functor used by FusedBatchNormGradOp to do the computations when
// is_training=False. Both CPU and GPU will use this functor.
template <typename Device, typename T, typename U>
struct FusedBatchNormFreezeGrad {
  void operator()(const Device& d, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& pop_mean_input,
                  const Tensor& pop_variance_input, U epsilon,
                  Tensor* x_backprop_output, Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output,
                  typename TTypes<U>::Vec scratch1,
                  typename TTypes<U>::Vec scratch2) {
    typename TTypes<T, 4>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 4>());
    typename TTypes<T, 4>::ConstTensor input(x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec pop_mean(pop_mean_input.vec<U>());
    typename TTypes<U>::ConstVec pop_var(pop_variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor x_backprop(x_backprop_output->tensor<T, 4>());
    typename TTypes<U>::Vec scale_backprop(scale_backprop_output->vec<U>());
    typename TTypes<U>::Vec offset_backprop(offset_backprop_output->vec<U>());

    const int depth = pop_mean.dimension(0);
    const int rest_size = input.size() / depth;

    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduction_axis{0};
    Eigen::array<int, 2> rest_by_one({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0> > reduction_axis;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > rest_by_one;
    rest_by_one.set(0, rest_size);
#endif

    // offset_backprop  = sum(y_backprop)
    // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var + epsilon))
    // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))

    auto y_backprop_rest_by_depth =
        y_backprop.reshape(rest_by_depth).template cast<U>();
    auto input_rest_by_depth = input.reshape(rest_by_depth).template cast<U>();

    offset_backprop.device(d) = y_backprop_rest_by_depth.sum(reduction_axis);

    // scratch1 = rsqrt(pop_var + epsilon)
    scratch1.device(d) = (pop_var + pop_var.constant(epsilon)).rsqrt();

    // scratch2 = sum(y_backprop * (x - mean))
    scratch2.device(d) =
        (y_backprop_rest_by_depth *
         (input_rest_by_depth -
          pop_mean.reshape(one_by_depth).broadcast(rest_by_one)))
            .sum(reduction_axis);

    x_backprop.reshape(rest_by_depth).device(d) =
        (y_backprop_rest_by_depth * ((scratch1 * scale)
                                         .eval()
                                         .reshape(one_by_depth)
                                         .broadcast(rest_by_one)))
            .template cast<T>();
    scale_backprop.device(d) = scratch2 * scratch1;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FUSED_BATCH_NORM_OP_H_
