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

#ifndef TENSORFLOW_CORE_KERNELS_REVERSE_SEQUENCE_OP_H_
#define TENSORFLOW_CORE_KERNELS_REVERSE_SEQUENCE_OP_H_
// Generator definition for ReverseSequenceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

template <typename T, typename Tlen, size_t Dims>
class ReverseGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ReverseGenerator(typename TTypes<T, Dims>::ConstTensor input, int32 batch_dim,
                   int32 seq_dim, typename TTypes<Tlen>::ConstVec seq_lengths)
      : input_(input),
        batch_dim_(batch_dim),
        seq_dim_(seq_dim),
        seq_lengths_(seq_lengths) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, Dims>& coords) const {
    Eigen::array<Eigen::DenseIndex, Dims> new_coords = coords;
    if (coords[seq_dim_] < seq_lengths_(coords[batch_dim_])) {
      new_coords[seq_dim_] =
          seq_lengths_(coords[batch_dim_]) - coords[seq_dim_] - 1;
    }

    return input_(new_coords);
  }

 private:
  typename TTypes<T, Dims>::ConstTensor input_;
  int32 batch_dim_;
  int32 seq_dim_;
  typename TTypes<Tlen>::ConstVec seq_lengths_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T, typename Tlen, size_t Dims>
struct ReverseSequence {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,
      int32 batch_dim, int32 seq_dim,
      typename TTypes<Tlen>::ConstVec seq_lengths,
      typename TTypes<T, Dims>::Tensor output) {
    generator::ReverseGenerator<T, Tlen, Dims> generator(input, batch_dim,
                                                         seq_dim, seq_lengths);
    output.device(d) = input.generate(generator);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REVERSE_SEQUENCE_OP_H_
