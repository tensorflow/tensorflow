/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RANDOM_BINOMIAL_OP_H_
#define TENSORFLOW_CORE_KERNELS_RANDOM_BINOMIAL_OP_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {

// Sample a binomial random variable, with probs and counts for each batch.
// Uses binomial inversion and a transformed rejection sampling method as
// described in
// https://pdfs.semanticscholar.org/471b/c2726e25bbf8801ef781630a2c13f654268e.pdf.
// Two different algorithms are employed, depending on the size of
// counts * probs (or counts * (1 - probs) if probs > 0.5.
// If counts * probs < 10, we simply sum up Geometric random variables until
// they exceed count, and the number we used is binomially distributed.
// In expectation, this will take O(counts * probs) time, and requiring in
// expectation the same number of random variates.
// This can be much cheaper than summing bernoulli random variates, as we
// will always need O(counts) bernoulli random variates (so this requires fewer
// uniform r.v.s as well as can be faster).
//
// If counts * probs > 10, we use a transformed-rejection algorithm based on
// pairs of uniform random variates due to Hormann.
// https://pdfs.semanticscholar.org/471b/c2726e25bbf8801ef781630a2c13f654268e.pdf
// This algorithm has higher acceptance rates for counts * probs large, as the
// proposal distribution becomes quite tight, requiring approximately two
// uniform random variates as counts * probs becomes large.
template <typename Device, typename T, typename U>
struct RandomBinomialFunctor {
  void operator()(OpKernelContext* ctx, const Device& d, int64_t num_batches,
                  int64_t samples_per_batch, int64_t num_elements,
                  typename TTypes<T>::ConstFlat counts,
                  typename TTypes<T>::ConstFlat probs,
                  const random::PhiloxRandom& gen,
                  typename TTypes<U>::Flat output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RANDOM_BINOMIAL_OP_H_
