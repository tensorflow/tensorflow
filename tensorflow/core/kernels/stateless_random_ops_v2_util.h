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

#ifndef TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_OPS_V2_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_OPS_V2_UTIL_H_

// Utilities for V2 stateless random ops' (non-XLA) kernels.

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/stateless_random_ops_v2.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
  auto dtype = DataTypeToEnum<T>::v();
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("input ", std::to_string(input_idx),
                                   " (0-based) must have shape [], not ",
                                   tensor.shape().DebugString());
  }
  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument("dtype of input ", std::to_string(input_idx),
                                   " (0-based) must be ", DataTypeString(dtype),
                                   ", not ", DataTypeString(tensor.dtype()));
  }
  *result = tensor.flat<T>()(0);
  return absl::OkStatus();
}

inline absl::StatusOr<std::tuple<Tensor, Tensor, Algorithm>>
GetKeyCounterAlgFromInputs(OpKernelContext* ctx, int key_input_idx,
                           int counter_input_idx, int alg_input_idx) {
  const Tensor& key_t = ctx->input(key_input_idx);
  const Tensor& counter_t = ctx->input(counter_input_idx);
  const Tensor& alg_t = ctx->input(alg_input_idx);

  int alg_id;
  TF_RETURN_IF_ERROR(GetScalar(alg_t, alg_input_idx, &alg_id));
  Algorithm alg = Algorithm(alg_id);
  if (alg == RNG_ALG_AUTO_SELECT) {
    alg = RNG_ALG_PHILOX;
  }

  TF_RETURN_IF_ERROR(
      CheckKeyCounterShape(alg, key_t.shape(), counter_t.shape()));
  return std::make_tuple(key_t, counter_t, alg);
}

template <typename Device, typename Distribution>
void FillRandomTensor(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
                      const Tensor& counter, Distribution dist,
                      Tensor* tensor) {
  typedef typename Distribution::ResultElementType T;
  auto flat = tensor->flat<T>();
  if (alg == RNG_ALG_PHILOX) {
    // Reuse the compute kernels from the stateful random ops
    auto key_data = key.flat<uint64>().data();
    auto counter_data = counter.flat<uint64>().data();
    functor::FillPhiloxRandom<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), key_data, counter_data,
        random::PhiloxRandom() /*dummy*/, flat.data(), flat.size(), dist);
  } else {
    OP_REQUIRES(ctx, false,
                errors::InvalidArgument("Unsupported algorithm id: ", alg));
  }
}
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATELESS_RANDOM_OPS_V2_UTIL_H_
