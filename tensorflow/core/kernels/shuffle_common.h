/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Common utilities for random shuffling.

#ifndef TENSORFLOW_CORE_KERNELS_SHUFFLE_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_SHUFFLE_COMMON_H_

#include <algorithm>
#include <functional>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

// TODO(irving): If performance is critical, generate output directly instead
// of an in-place shuffle using a pseudorandom permutation like
//
//   https://github.com/otherlab/geode/blob/master/geode/random/permute.cpp
//
// This is probably also the right thing if we want a GPU version of shuffling.

// We use our own version of std::random_shuffle to guarantee that exactly
// size - 1 samples are used.
template <class Iter, class Random>
static inline void ShuffleRange(Iter first, Iter last, Random& uniform) {
  if (first == last) return;
  const auto stop = last - 1;
  for (auto i = first; i != stop; ++i) {
    using std::iter_swap;
    iter_swap(i, i + uniform(last - i));
  }
}

template <class IntT, class InT, class OutT, class Random>
static void IndexedShuffle(const int64_t size, const InT& input_mat,
                           OutT output_mat, Random& uniform) {
  std::vector<IntT> permutation(size);
  for (IntT i = 0; i < size; i++) {
    permutation[i] = i;
  }
  ShuffleRange(permutation.begin(), permutation.end(), uniform);
  for (IntT i = 0; i < size; i++) {
    output_mat.template chip<0>(i) = input_mat.template chip<0>(permutation[i]);
  }
}

template <typename T>
Status RandomShuffle(OpKernelContext* context, const Tensor& input,
                     int output_idx,
                     std::function<random::PhiloxRandom(int64_t)> get_rng) {
  if (input.NumElements() <= 1 || input.dim_size(0) <= 1) {
    // No shuffling is required, so copy input directly to output
    context->set_output(output_idx, input);
  } else {
    // Reserve enough random samples for shuffling
    const int64_t size = input.dim_size(0);
    const int64_t samples = size - 1;
    auto rng = get_rng(samples);
    random::SingleSampleAdapter<random::PhiloxRandom> single(&rng);
    const auto uniform = [&single](uint32 n) { return single() % n; };

    if (input.dims() == 1) {
      // For 1D data, copy and then shuffle in place
      context->set_output(output_idx, tensor::DeepCopy(input));
      auto vec = context->mutable_output(output_idx)->vec<T>();
      ShuffleRange(vec.data(), vec.data() + size, uniform);
    } else {
      // For >= 2D, shuffle indices and then copy across
      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(
          context->allocate_output(output_idx, input.shape(), &output));
      const auto input_mat = input.flat_outer_dims<T>();
      auto output_mat = output->flat_outer_dims<T>();
      if (size < kint32max) {
        IndexedShuffle<int32>(size, input_mat, output_mat, uniform);
      } else {
        IndexedShuffle<int64_t>(size, input_mat, output_mat, uniform);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SHUFFLE_COMMON_H_
