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
#ifndef TENSORFLOW_KERNELS_FRACTIONAL_POOL_COMMON_H_
#define TENSORFLOW_KERNELS_FRACTIONAL_POOL_COMMON_H_

#include <algorithm>
#include <vector>

#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

// Shuffle a container randomly, copied from random_shuffle_op.cc
template <class Iter, class Random>
static inline void RandomShuffle(Iter first, Iter last, const Random& uniform) {
  if (first == last) {
    return;
  }
  const auto stop = last - 1;
  for (auto i = first; i != stop; ++i) {
    using std::iter_swap;
    iter_swap(i, i + uniform(last - i));
  }
}

// Generate pooling sequence for fractional pooling along one dimension.
//
// Regular max/avg pooling can be viewed as a special case, in which given the
//     * input_length: e.g. 10
//     * output_length: e.g. 5
// it will generate pooling sequence as
//     diff sequence: [2, 2, 2, 2, 2]
// or as
//     cumulative sequence: [0, 2, 4, 6, 8, 10]
//
// In the case of fractional pooling, input_length is not an integer multiple of
// output_length, randomness plays a role when generating pooling sequence.
// There are two type of randomness (random vs pseudo-random) defined in paper:
// http://arxiv.org/abs/1412.6071
// You can check the paper for the difference between these two types.
//
// In summary, the generated diff sequence satisfy the following properties for
// both types of randomness:
//     * length(generated_diff_pooling_sequence) = output_length
//     * sum(generated_diff_pooling_sequence) = input_length
//     * Let's define floor(input_length / output_length) = K, then
//       K <= generated_diff_pooling_sequence[i] <= K+1
// For example, when input_length = 10, output_length = 6, the following are
// valid pooling sequence:
//     * [1, 2, 2, 1, 2, 2]
//     * [1, 1, 2, 2, 2, 2]
// [1, 3, 2, 2, 2, 2] is not valid.
//
// Args:
//   input_length:  See above explanation
//   output_length:  See above explanation
//   generator:  Parallel version of random number generator
//   pseudo_random:  Whether or not use pseudo-random
// Returns:
//   pooling_sequence:  This is the cumulative pooling sequence.
std::vector<int64> GeneratePoolingSequence(int input_length, int output_length,
                                           GuardedPhiloxRandom* generator,
                                           bool pseudo_random);
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FRACTIONAL_POOL_COMMON_H_
