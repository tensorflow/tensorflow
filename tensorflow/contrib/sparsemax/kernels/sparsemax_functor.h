/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_SPARSEMAX_FUNCTOR_OP_H_
#define TENSORFLOW_KERNELS_SPARSEMAX_FUNCTOR_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct Sparsemax {
  void operator()(const Device& d,
                  typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Vec temp_vec,
                  typename TTypes<T>::Matrix temp_mat,
                  typename TTypes<T>::Matrix output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SPARSEMAX_FUNCTOR_OP_H_
