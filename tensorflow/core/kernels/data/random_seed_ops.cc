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
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace {

const char kSeedGenerator[] = "SeedGenerator";
const char kSeed[] = "seed";
const char kSeed2[] = "seed2";
const char kReshuffle[] = "reshuffle";

}  // namespace

string SeedGeneratorManager::DebugString() const { return kSeedGenerator; }

void FixedSeedGenerator::GenerateSeeds(int64_t* seed1, int64_t* seed2) {
  mutex_lock l(mu_);
  num_random_samples_++;
  *seed1 = seeds_.seed();
  *seed2 = seeds_.seed2();
}

void RandomSeedGenerator::GenerateSeeds(int64_t* seed1, int64_t* seed2) {
  mutex_lock l(mu_);
  num_random_samples_++;
  *seed1 = generator_();
  num_random_samples_++;
  *seed2 = generator_();
}

void RandomSeedGenerator::Reset() {
  mutex_lock l(mu_);
  // Reset the generators based on the current seeds.
  parent_generator_ = random::PhiloxRandom(seeds_.seed(), seeds_.seed2());
  generator_ =
      random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
  generator_.Skip(num_random_samples_);
}

AnonymousSeedGeneratorHandleOp::AnonymousSeedGeneratorHandleOp(
    OpKernelConstruction* ctx)
    : AnonymousResourceOp<SeedGeneratorManager>(ctx,
                                                /* ref_counting */ true,
                                                /* return_deleter */ true) {}

void AnonymousSeedGeneratorHandleOp::Compute(OpKernelContext* ctx) {
  int64_t seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
  int64_t seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
  // Seeds will be consumed by `CreateResource`, which is called via `Compute`.
  mutex_lock l(mu_);
  seeds_ = std::make_unique<RandomSeeds>(seed, seed2);
  OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, kReshuffle, &reshuffle_));
  AnonymousResourceOp<SeedGeneratorManager>::Compute(ctx);
}

std::string AnonymousSeedGeneratorHandleOp::name() { return kSeedGenerator; }

absl::Status AnonymousSeedGeneratorHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, SeedGeneratorManager** manager)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (reshuffle_) {
    *manager = new SeedGeneratorManager(new RandomSeedGenerator(*seeds_));
  } else {
    *manager = new SeedGeneratorManager(new FixedSeedGenerator(*seeds_));
  }
  seeds_ = nullptr;
  return absl::OkStatus();
}

void DeleteSeedGeneratorOp::Compute(OpKernelContext* ctx) {
  ResourceHandle handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The resource is guaranteed to exist because the variant tensor wrapping the
  // deleter is provided as an unused input to this op, which guarantees that it
  // has not run yet.
  OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete(handle));
}

namespace {
REGISTER_KERNEL_BUILDER(Name("AnonymousSeedGenerator").Device(DEVICE_CPU),
                        AnonymousSeedGeneratorHandleOp);

REGISTER_KERNEL_BUILDER(Name("DeleteSeedGenerator").Device(DEVICE_CPU),
                        DeleteSeedGeneratorOp);

REGISTER_KERNEL_BUILDER(Name("AnonymousRandomSeedGenerator").Device(DEVICE_CPU),
                        AnonymousSeedGeneratorHandleOp);

REGISTER_KERNEL_BUILDER(Name("DeleteRandomSeedGenerator").Device(DEVICE_CPU),
                        DeleteSeedGeneratorOp);

REGISTER_KERNEL_BUILDER(Name("DummySeedGenerator").Device(DEVICE_CPU),
                        DummyResourceOp<SeedGenerator>);

}  // namespace
}  // namespace data
}  // namespace tensorflow
