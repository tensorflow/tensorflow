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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {

// A random seed generator resource.
class RandomSeedGenerator : public ResourceBase {
 public:
  RandomSeedGenerator(int64 seed, int64 seed2)
      : seed_(seed),
        seed2_(seed2),
        parent_generator_(seed, seed2),
        generator_(&parent_generator_) {}

  int64 num_random_samples();
  void set_num_random_samples(int64 num_random_samples);

  string DebugString() const override;
  void GenerateRandomSeeds(int64* seed1, int64* seed2);
  void Reset();
  void Serialize(OpKernelContext* ctx);

 private:
  const int64 seed_;
  const int64 seed2_;
  mutex mu_;
  random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_ GUARDED_BY(mu_);
  int64 num_random_samples_ GUARDED_BY(mu_) = 0;
};

// Creates an instance of random seed generator resource and transfers ownership
// to the caller.
class AnonymousRandomSeedGeneratorHandleOp
    : public AnonymousResourceOp<RandomSeedGenerator> {
 public:
  explicit AnonymousRandomSeedGeneratorHandleOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  string name() override;
  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        RandomSeedGenerator** resource) override;

  int64 seed_;
  int64 seed2_;
};

// Deletes an instance of random seed generator resource.
class DeleteRandomSeedGeneratorOp : public OpKernel {
 public:
  explicit DeleteRandomSeedGeneratorOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_
