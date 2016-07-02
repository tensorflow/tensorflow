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

#ifndef TENSORFLOW_KERNELS_DENSE_UPDATE_OPS_H_
#define TENSORFLOW_KERNELS_DENSE_UPDATE_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

enum DenseUpdateType { ADD, SUB, ASSIGN };

namespace functor {

template <typename Device, typename T, DenseUpdateType OP>
struct DenseUpdate;

template <typename Device, typename T>
struct DenseUpdate<Device, T, ADD> {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename Device, typename T>
struct DenseUpdate<Device, T, SUB> {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

template <typename Device, typename T>
struct DenseUpdate<Device, T, ASSIGN> {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DENSE_UPDATE_OPS_H_
