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

#ifndef TENSORFLOW_KERNELS_CUDNN_BATCH_NORM_OP_H_
#define TENSORFLOW_KERNELS_CUDNN_BATCH_NORM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by BatchNormTrainingOp to do the computations.
template <typename Device, typename T>
struct BatchNormTraining {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T>::ConstVec mean,
                  typename TTypes<T>::ConstVec var,
                  typename TTypes<T>::ConstVec beta,
                  typename TTypes<T>::ConstVec gamma,
                  float variance_epsilon,
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
    //MAKE the call to the stream executor
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_BATCH_NORM_OP_H_
