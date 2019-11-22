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
#include "tensorflow/core/kernels/data/random_seed_ops.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {
namespace {

const char kNumRandomSamples[] = "num_random_samples";
const char kRandomSeedGenerator[] = "RandomSeedGenerator";
const char kSeed[] = "seed";
const char kSeed2[] = "seed2";

}  // namespace

string RandomSeedGenerator::DebugString() const { return kRandomSeedGenerator; }

void RandomSeedGenerator::GenerateRandomSeeds(int64* seed1, int64* seed2) {
  mutex_lock l(mu_);
  num_random_samples_++;
  *seed1 = generator_();
  num_random_samples_++;
  *seed2 = generator_();
}

int64 RandomSeedGenerator::num_random_samples() {
  tf_shared_lock l(mu_);
  return num_random_samples_;
}

void RandomSeedGenerator::set_num_random_samples(int64 num_random_samples) {
  mutex_lock l(mu_);
  num_random_samples_ = num_random_samples;
}

void RandomSeedGenerator::Reset() {
  mutex_lock l(mu_);
  // Reset the generators based on the current seeds.
  parent_generator_ = random::PhiloxRandom(seed_, seed2_);
  generator_ =
      random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
  generator_.Skip(num_random_samples_);
}

void RandomSeedGenerator::Serialize(OpKernelContext* ctx) {
  mutex_lock l(mu_);
  Tensor* num_random_samples;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(kNumRandomSamples, TensorShape({}),
                                           &num_random_samples));
  num_random_samples->scalar<int64>()() = num_random_samples_;
  Tensor* seed;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(kSeed, TensorShape({}), &seed));
  seed->scalar<int64>()() = seed_;
  Tensor* seed2;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(kSeed2, TensorShape({}), &seed2));
  seed2->scalar<int64>()() = seed2_;
}

AnonymousRandomSeedGeneratorHandleOp::AnonymousRandomSeedGeneratorHandleOp(
    OpKernelConstruction* ctx)
    : AnonymousResourceOp<RandomSeedGenerator>(ctx) {}

void AnonymousRandomSeedGeneratorHandleOp::Compute(OpKernelContext* ctx) {
  int64 seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kSeed, &seed));
  int64 seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kSeed2, &seed2));
  if (seed == 0 && seed2 == 0) {
    seed = random::New64();
    seed2 = random::New64();
  }
  seed_ = seed;
  seed2_ = seed2;
  AnonymousResourceOp<RandomSeedGenerator>::Compute(ctx);
}

string AnonymousRandomSeedGeneratorHandleOp::name() {
  return kRandomSeedGenerator;
}

Status AnonymousRandomSeedGeneratorHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, RandomSeedGenerator** resource) {
  *resource = new RandomSeedGenerator(seed_, seed2_);
  return Status::OK();
}

void DeleteRandomSeedGeneratorOp::Compute(OpKernelContext* ctx) {
  ResourceHandle handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The resource is guaranteed to exist because the variant tensor wrapping the
  // deleter is provided as an unused input to this op, which guarantees that it
  // has not run yet.
  OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete(handle));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("AnonymousRandomSeedGenerator").Device(DEVICE_CPU),
                        AnonymousRandomSeedGeneratorHandleOp);

REGISTER_KERNEL_BUILDER(Name("DeleteRandomSeedGenerator").Device(DEVICE_CPU),
                        DeleteRandomSeedGeneratorOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
