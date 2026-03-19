/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_OPS_H_
#define TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

// Generates a key and counter that can be used to seed a PhiloxRandom,
// generator, based on the seed value in `seed_t`.
//
// REQUIRES: `seed_t` must be a length-2 vector of type DT_INT{32,64}.
// `out_key` and `out_counter` must be non-null.
absl::Status GenerateKey(Tensor seed_t, random::PhiloxRandom::Key* out_key,
                         random::PhiloxRandom::ResultType* out_counter);

// A base class for kernels of stateless RNG ops that take shape and seed as the
// first 2 inputs.
class StatelessRandomOpBase : public OpKernel {
 public:
  explicit StatelessRandomOpBase(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 protected:
  // The part of Compute that depends on device, type, and distribution.
  // Must be a tail call because it doesn't report error via return value.
  virtual void Fill(OpKernelContext* context, random::PhiloxRandom random,
                    Tensor* output) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_OPS_H_
