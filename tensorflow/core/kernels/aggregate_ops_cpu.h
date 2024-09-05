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

#ifndef TENSORFLOW_CORE_KERNELS_AGGREGATE_OPS_CPU_H_
#define TENSORFLOW_CORE_KERNELS_AGGREGATE_OPS_CPU_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/core/kernels/aggregate_ops.h"

typedef Eigen::ThreadPoolDevice CPUDevice;


namespace tensorflow {

// Partial specializations for a CPUDevice, that uses the Eigen implementation
// from AddNEigenImpl.
namespace functor {
template <typename T>
struct Add2Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2) {
    Add2EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2);
  }
};
template <typename T>
struct Add3Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3) {
    Add3EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3);
  }
};
template <typename T>
struct Add4Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4) {
    Add4EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4);
  }
};
template <typename T>
struct Add5Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5) {
    Add5EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5);
  }
};
template <typename T>
struct Add6Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6) {
    Add6EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6);
  }
};
template <typename T>
struct Add7Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6,
                  typename TTypes<T>::ConstFlat in7) {
    Add7EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7);
  }
};

template <typename T>
struct Add8Functor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8);
  }
};

template <typename T>
struct Add8pFunctor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8pEigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                          in7, in8);
  }
};

template <typename T>
struct Add9Functor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9) {
    Add9EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8, in9);
  }
};


}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_AGGREGATE_OPS_CPU_H_
