/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

template <typename T>
class SkipGramGenerateCandidatesOp : public OpKernel {
 public:
  explicit SkipGramGenerateCandidatesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input_tensor", &input_tensor));
    const auto input = input_tensor->flat<T>();

    const Tensor* min_skips_tensor;
    OP_REQUIRES_OK(context, context->input("min_skips", &min_skips_tensor));
    const int min_skips = *(min_skips_tensor->scalar<int>().data());
    const Tensor* max_skips_tensor;
    OP_REQUIRES_OK(context, context->input("max_skips", &max_skips_tensor));
    const int max_skips = *(max_skips_tensor->scalar<int>().data());

    OP_REQUIRES(
        context, min_skips >= 0 && max_skips >= 0,
        errors::InvalidArgument("Both min_skips and max_skips must be >= 0."));
    OP_REQUIRES(context, min_skips <= max_skips,
                errors::InvalidArgument("min_skips must be <= max_skips."));

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int start = *(start_tensor->scalar<int>().data());
    const Tensor* limit_tensor;
    OP_REQUIRES_OK(context, context->input("limit", &limit_tensor));
    const int limit = *(limit_tensor->scalar<int>().data());
    const int end =
        limit < 0 ? input.size()
                  : std::min(start + limit, static_cast<int>(input.size()));

    const Tensor* emit_self_tensor;
    OP_REQUIRES_OK(context,
                   context->input("emit_self_as_target", &emit_self_tensor));
    const bool emit_self_as_target = *(emit_self_tensor->scalar<bool>().data());

    std::vector<T> tokens;
    std::vector<T> labels;

    // Reserve the number of random numbers we will use - we use one for each
    // token between start and end.
    random::PhiloxRandom local_gen =
        generator_.ReserveSamples32(end - start + 1);
    random::SimplePhilox rng(&local_gen);

    // For each token in the sentence, pick a random skip, then generates
    // (token, label) pairs for all labels whose distances from the token are
    // within the range [-skip, skip].
    for (int i = start; i < end; ++i) {
      const int skips = min_skips + rng.Uniform(max_skips - min_skips + 1);
      for (int j = -skips; j <= skips; ++j) {
        if ((i + j < start) || (i + j >= end) ||
            (j == 0 && !emit_self_as_target)) {
          continue;
        }
        tokens.push_back(input(i));
        labels.push_back(input(i + j));
      }
    }

    Tensor* tokens_output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "tokens", TensorShape({static_cast<int>(tokens.size())}),
                       &tokens_output));
    Tensor* labels_output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "labels", TensorShape({static_cast<int>(labels.size())}),
                       &labels_output));
    OP_REQUIRES(
        context, tokens_output->IsSameSize(*labels_output),
        errors::Internal(strings::StrCat(
            "Mismatch between tokens_output shape of ",
            tokens_output->shape().DebugString(),
            " and labels_output shape of ",
            labels_output->shape().DebugString(),
            ". This should never happen - contact ami-team@ if it does.")));

    // Copies results to output tensors.
    for (int i = 0; i < tokens.size(); ++i) {
      tokens_output->vec<T>()(i) = tokens[i];
      labels_output->vec<T>()(i) = labels[i];
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("SkipGramGenerateCandidates") \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<type>("T"),    \
                          SkipGramGenerateCandidatesOp<type>)

REGISTER_KERNEL(string);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(int16);
// TODO(weiho): Add other types if the need arises.

#undef REGISTER_KERNEL

}  // namespace tensorflow
