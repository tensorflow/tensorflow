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

#ifndef TENSORFLOW_KERNELS_WHERE_OP_H_
#define TENSORFLOW_KERNELS_WHERE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

template <typename Device>
struct NumTrue {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<bool>::ConstFlat input,
      TTypes<int64>::Scalar num_true) {
    num_true.device(d) = input.template cast<int64>().sum();
  }
};

template <typename Device, int NDIM>
struct Where {
  EIGEN_ALWAYS_INLINE static int64 Compute(
      const Device& d, typename TTypes<bool, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output) {
    Eigen::DenseIndex true_n = 0;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides;

    // Calculate strides for RowMajor order.
    EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                         static_cast<int>(Eigen::RowMajor)),
                        INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);

    strides[NDIM - 1] = 1;
    for (int i = NDIM - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }

    Eigen::DenseIndex output_size = output.dimension(0);
    for (Eigen::DenseIndex n = 0; n < input.size(); ++n) {
      if (input.data()[n]) {
        if (TF_PREDICT_TRUE(true_n < output_size)) {
          WriteIndexRowMajor(output, strides, true_n, n);
        }
        ++true_n;
      }
    }
    return true_n;
  }

  EIGEN_ALWAYS_INLINE static void WriteIndexRowMajor(
      typename TTypes<int64>::Matrix output,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides,
      Eigen::DenseIndex true_n, Eigen::DenseIndex index) {
    for (int i = 0; i < NDIM; ++i) {
      output(true_n, i) = index / strides[i];
      index %= strides[i];
    }
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_WHERE_OP_H_
