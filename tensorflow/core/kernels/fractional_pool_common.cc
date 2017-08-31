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
#include "tensorflow/core/kernels/fractional_pool_common.h"

#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
static std::vector<int64> GeneratePoolingSequencePseudoRandom(
    int input_length, int output_length, GuardedPhiloxRandom* generator) {
  std::vector<int64> cum_seq(output_length + 1, 0);
  std::vector<int64> diff(output_length, 0);

  double alpha = static_cast<double>(input_length) / output_length;
  int k = input_length / output_length;

  // In the paper [1], author proposes the following procedure to generate a
  // pseudo random pooling region:
  //   1) Set a_0 = 1, a_Nout = Nin;
  //   2) a_i = ceil(alpha*(u+i))
  //      in which, i = 1, 2, ... , Nout-1
  //                u is a random number in (0,1) for all i
  //                alpha = Nin/Nout in (1,2)
  // The sequence {a_i} should satisfy a_i-a_{i-1} = 1 or 2
  // Note: for step 1), it makes more sense to make a_Nout = Nin+1, that way,
  //    a_i-a_{i-1} = 1 or 2 is also true for i = Nout.
  //
  // However, there are choices of alpha and u that will make
  // a_i - a_{i-1} > 2. This happens at the left boundary. For example, with
  // alpha = 1.732, u = 0.8, then a_1 = 4, a_1-a_0 = 3.
  // This is why u_max1 is needed, i.e. u is a random number in (0,u_max1)
  // instead of (0,1).
  // Define k = ceil(alpha)-1, then we require:
  //   a_1 = alpha*(u+1) <= a_0+(k+1)
  // ===> This gives u_max1 = (k+2)/alpha - 1.
  //
  // In addition, when extending the pooling sequence generation process for
  // alpha beyond (1,2), e.g. (k,k+1); a check on the right boundary is also
  // needed to make sure the last gap a_Nout-a_{Nout-1} >= k. Solving it gives
  // u_max2 = (Nin+1-k)/alpha - (Nout-1)
  // Here is an example where u > u_max2, alpha = 2.3, u = 0.7, u_max2 = 0.565,
  // Nin = 23, Nout = 10; the sequence
  // from a_0 to a_10 is:
  // [1, 4, 7, 9, 11, 14, 16, 18, 21, 23, 24]
  // The last gap is only 1.
  //
  // [1]: https://arxiv.org/abs/1412.6071
  double u_max1 = (k + 2) / alpha - 1;
  double u_max2 = (input_length + 1 - k) / alpha - (output_length - 1);
  double max_u = std::min(u_max1, u_max2);

  // Generate random number in parallel.
  auto local_gen = generator->ReserveSamples32(2);
  random::SimplePhilox random(&local_gen);
  const double u = random.RandDouble() * max_u;

  cum_seq[0] = 1;
  cum_seq[output_length] = input_length + 1;
  for (int i = 1; i < output_length; ++i) {
    cum_seq[i] = static_cast<int>(ceil(alpha * (i + u)));
  }

  for (int i = 0; i < output_length; ++i) {
    diff[i] = cum_seq[i + 1] - cum_seq[i];
  }

  return diff;
}

static std::vector<int64> GeneratePoolingSequenceRandom(
    int input_length, int output_length, GuardedPhiloxRandom* generator) {
  int k = input_length / output_length;
  int num_random_spot = input_length % output_length;
  std::vector<int64> diff(output_length, k);

  for (int i = 0; i < num_random_spot; ++i) {
    diff[i] += 1;
  }

  // Randomly shuffle this vector.
  auto local_gen = generator->ReserveSamples32(diff.size());
  random::SingleSampleAdapter<random::PhiloxRandom> single(&local_gen);
  const auto uniform = [&single](uint32 n) { return single() % n; };
  RandomShuffle(diff.begin(), diff.end(), uniform);

  return diff;
}

std::vector<int64> GeneratePoolingSequence(int input_length, int output_length,
                                           GuardedPhiloxRandom* generator,
                                           bool pseudo_random) {
  std::vector<int64> diff;
  // This is a case that regular pooling can handle, just return diff with
  // each element input_length/output_length.
  if (input_length % output_length == 0) {
    diff = std::vector<int64>(output_length, input_length / output_length);
  }

  if (pseudo_random) {
    diff = GeneratePoolingSequencePseudoRandom(input_length, output_length,
                                               generator);
  } else {
    diff =
        GeneratePoolingSequenceRandom(input_length, output_length, generator);
  }

  // Sanity check.
  int k = input_length / output_length;
  for (int i = 0; i < output_length; ++i) {
    // k<= diff[i] <= k+1.
    DCHECK_GE(diff[i], k);
    DCHECK_LE(diff[i], k + 1);
  }

  // Return cumulative sequence.
  std::vector<int64> cum_seq(output_length + 1, 0);
  for (int i = 1; i < cum_seq.size(); ++i) {
    cum_seq[i] = cum_seq[i - 1] + diff[i - 1];
  }
  return cum_seq;
}

}  // namespace tensorflow
