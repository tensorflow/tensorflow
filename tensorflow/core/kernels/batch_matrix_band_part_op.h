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

#ifndef TENSORFLOW_KERNELS_BATCH_MATRIX_DIAG_OP_H_
#define TENSORFLOW_KERNELS_BATCH_MATRIX_DIAG_OP_H_

// Generator definition for BatchMatrixBandPartOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

template <typename T>
class BatchMatrixBandPartGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE BatchMatrixBandPartGenerator(
      Eigen::DenseIndex num_lower, Eigen::DenseIndex num_upper,
      typename TTypes<T, 3>::ConstTensor input)
      : num_lower_(num_lower), num_upper_(num_upper), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& coords) const {
    return (((num_lower_ < 0 || coords[1] - coords[2] <= num_lower_) &&
             (num_upper_ < 0 || coords[2] - coords[1] <= num_upper_))
                ? input_(coords)
                : T());
  }

 private:
  const Eigen::DenseIndex num_lower_;
  const Eigen::DenseIndex num_upper_;
  typename TTypes<T, 3>::ConstTensor input_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T>
struct BatchMatrixBandPart {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, Eigen::DenseIndex num_lower, Eigen::DenseIndex num_upper,
      typename TTypes<T, 3>::ConstTensor input,
      typename TTypes<T, 3>::Tensor output) {
    if ((num_lower < 0 || num_lower >= input.dimension(1)) &&
        (num_upper < 0 || num_upper >= input.dimension(2))) {
      output.device(d) = input;
    } else {
      generator::BatchMatrixBandPartGenerator<T> generator(num_lower, num_upper,
                                                           input);
      output.device(d) = output.generate(generator);
    }
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_BATCH_MATRIX_DIAG_OP_H_
