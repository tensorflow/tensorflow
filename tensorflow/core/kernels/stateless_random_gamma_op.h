/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_GAMMA_OP_H_
#define TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_GAMMA_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct StatelessRandomGammaFunctor {
  static Status Fill(OpKernelContext* ctx, const T* alpha_flat,
                     int64 num_alphas, int64 samples_per_alpha,
                     const random::PhiloxRandom& random, T* samples_flat);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_GAMMA_OP_H_
