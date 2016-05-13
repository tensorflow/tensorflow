/* Copyright 2016 Google Inc. All Rights Reserved.

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

// See docs in ../ops/sampling_ops.cc
#define EIGEN_USE_THREADS

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

template <typename T>
class BernoulliSampleOp : public OpKernel {
 public:
  explicit BernoulliSampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* p_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("p", &p_tensor));

    const Tensor* a_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("a", &a_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    OP_REQUIRES(ctx, a_tensor->shape() == b_tensor->shape(),
        errors::InvalidArgument("a_tensor.shape() != b_tensor.shape()"));

    Tensor* output_tensor = nullptr;
    ctx->allocate_output("output", a_tensor->shape(), &output_tensor);

    const int64 batch_size = a_tensor->dim_size(0);

    typedef random::UniformDistribution<random::PhiloxRandom, float>
        Distribution;
    Distribution dist;

    // First determine whether we want to sample or not sample from our distribution.
    // Sample our random numbers.
    const int kGroupSize = Distribution::kResultElementCount;
    auto local_generator = generator_.ReserveSamples32(batch_size);
    std::vector<float> sample_prob(batch_size + kGroupSize);
    for (int64 i = 0; i < batch_size; i += kGroupSize) {
      auto samples = dist(&local_generator);
      std::copy(&samples[0], &samples[0] + kGroupSize, &sample_prob[i]);
    }

    const float p = p_tensor->scalar<float>()();
    for (int64 b = 0; b < batch_size; ++b) {
      output_tensor->vec<T>()(b) = sample_prob[b] >= p
          ? a_tensor->vec<T>()(b)
          : b_tensor->vec<T>()(b);
    }
  }

 private:
  float sample_prob_;
  GuardedPhiloxRandom generator_;
};

#define REGISTER_BERNOULLI_SAMPLE(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("BernoulliSample").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BernoulliSampleOp<type>);

TF_CALL_NUMBER_TYPES(REGISTER_BERNOULLI_SAMPLE);

class SampleDistributionIndexOp : public OpKernel {
 public:
  explicit SampleDistributionIndexOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* p_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("p", &p_tensor));

    const int64 batch_size = p_tensor->dim_size(0);
    const int64 vocab_size = p_tensor->dim_size(1);

    Tensor* output_tensor = nullptr;
    ctx->allocate_output("output", TensorShape({batch_size}), &output_tensor);

    typedef random::UniformDistribution<random::PhiloxRandom, float>
        Distribution;
    Distribution dist;

    // First determine whether we want to sample or not sample from our distribution.
    // Sample our random numbers.
    const int kGroupSize = Distribution::kResultElementCount;
    auto local_generator = generator_.ReserveSamples32(batch_size);
    std::vector<float> sample_prob(batch_size + kGroupSize);
    for (int64 i = 0; i < batch_size; i += kGroupSize) {
      auto samples = dist(&local_generator);
      std::copy(&samples[0], &samples[0] + kGroupSize, &sample_prob[i]);
    }

    for (int64 b = 0; b < batch_size; ++b) {
      float prob_sum = 0.0f;
      int64 idx = 0;
      for (; idx < vocab_size; ++idx) {
        prob_sum += p_tensor->matrix<float>()(b, idx);
        if (prob_sum >= sample_prob[b]) break;
      }

      if (idx >= vocab_size) idx = vocab_size - 1;

      output_tensor->vec<int64>()(b) = idx;
    }
  }

 private:
  float sample_prob_;
  GuardedPhiloxRandom generator_;
};

REGISTER_KERNEL_BUILDER(Name("SampleDistributionIndex")
                             .Device(DEVICE_CPU),
                        SampleDistributionIndexOp);

}  // end namespace tensorflow
