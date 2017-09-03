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

#ifndef TENSORFLOW_KERNELS_MATRIX_SET_DIAG_OP_H_
#define TENSORFLOW_KERNELS_MATRIX_SET_DIAG_OP_H_

// Generator definition for MatrixSetDiagOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

template <typename T>
class OverwriteDiagGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OverwriteDiagGenerator(typename TTypes<T, 2>::ConstTensor diag,
                         typename TTypes<T, 3>::Tensor output)
      : diag_(diag), output_(output) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    Eigen::array<Eigen::DenseIndex, 3> diag_from_coords(
        {coords[0], coords[1], coords[1]});

    // This is the side effect we care about.
    output_(diag_from_coords) = diag_(coords);

    return T(0);
  }

 private:
  typename TTypes<T, 2>::ConstTensor diag_;
  mutable typename TTypes<T, 3>::Tensor output_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T>
struct MatrixSetDiag {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, 3>::ConstTensor input,
      typename TTypes<T, 2>::ConstTensor diag,
      typename TTypes<T>::Scalar scratch,
      typename TTypes<T, 3>::Tensor output) {
    output.device(d) = input;
    generator::OverwriteDiagGenerator<T> generator(diag, output);
    // Use sum() to force the generation to aggregate to the scalar
    // output scratch.  This in turn forces each element of the
    // generator to execute.  The side effect of the execution is to
    // update the diagonal components of output with diag.
    scratch.device(d) = diag.generate(generator).sum();
  }
};

template <typename Device>
struct MatrixSetDiag<Device, bool> {
  EIGEN_ALWAYS_INLINE static void Compute(const Device& d,
                                          TTypes<bool, 3>::ConstTensor input,
                                          TTypes<bool, 2>::ConstTensor diag,
                                          TTypes<bool>::Scalar scratch,
                                          TTypes<bool, 3>::Tensor output) {
    output.device(d) = input;
    generator::OverwriteDiagGenerator<bool> generator(diag, output);
    // Use all() to force the generation to aggregate to the scalar
    // output scratch.  This in turn forces each element of the
    // generator to execute.  The side effect of the execution is to
    // update the diagonal components of output with diag.
    scratch.device(d) = diag.generate(generator).all();
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MATRIX_SET_DIAG_OP_H_
