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

template <typename Device, typename Type>
struct Cross {
  void operator()(const Device &d,
                  typename TTypes<Type, 2>::ConstTensor in0_data,
                  typename TTypes<Type, 2>::ConstTensor in1_data,
                  typename TTypes<Type, 2>::Tensor output_data) {
    auto s1 = output_data.template chip<1>(0);
    auto s2 = output_data.template chip<1>(1);
    auto s3 = output_data.template chip<1>(2);

    auto u1 = in0_data.template chip<1>(0);
    auto u2 = in0_data.template chip<1>(1);
    auto u3 = in0_data.template chip<1>(2);

    auto v1 = in1_data.template chip<1>(0);
    auto v2 = in1_data.template chip<1>(1);
    auto v3 = in1_data.template chip<1>(2);

    s1.device(d) = u2 * v3 - u3 * v2;
    s2.device(d) = u3 * v1 - u1 * v3;
    s3.device(d) = u1 * v2 - u2 * v1;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_COLORSPACE_OP_H_
