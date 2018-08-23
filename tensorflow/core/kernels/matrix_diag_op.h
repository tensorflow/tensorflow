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

#ifndef TENSORFLOW_CORE_KERNELS_MATRIX_DIAG_OP_H_
#define TENSORFLOW_CORE_KERNELS_MATRIX_DIAG_OP_H_

// Generator definition for MatrixDiagOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

template <typename T>
class MatrixDiagPartGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  MatrixDiagPartGenerator(typename TTypes<T, 3>::ConstTensor input)
      : input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    Eigen::array<Eigen::DenseIndex, 3> diag_from_coords(
        {coords[0], coords[1], coords[1]});
    return input_(diag_from_coords);
  }

 private:
  typename TTypes<T, 3>::ConstTensor input_;
};

template <typename T>
class MatrixDiagGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  MatrixDiagGenerator(typename TTypes<T, 2>::ConstTensor input)
      : input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& coords) const {
    if (coords[2] != coords[1]) return T();

    Eigen::array<Eigen::DenseIndex, 2> diag_coords({coords[0], coords[1]});
    return input_(diag_coords);
  }

 private:
  typename TTypes<T, 2>::ConstTensor input_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T>
struct MatrixDiagPart {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, 3>::ConstTensor input,
      typename TTypes<T, 2>::Tensor output) {
    generator::MatrixDiagPartGenerator<T> generator(input);
    output.device(d) = output.generate(generator);
  }
};

template <typename Device, typename T>
struct MatrixDiag {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, 2>::ConstTensor input,
      typename TTypes<T, 3>::Tensor output) {
    generator::MatrixDiagGenerator<T> generator(input);
    output.device(d) = output.generate(generator);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATRIX_DIAG_OP_H_
