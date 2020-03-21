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

// Base class for seed generator resources. Subclasses customize how seeds are
// generated.
class SeedGenerator : public ResourceBase {
 public:
  virtual void GenerateSeeds(int64* seed1, int64* seed2) = 0;
  virtual void Reset() = 0;

  virtual int64 num_random_samples();
  virtual void set_num_random_samples(int64 num_random_samples);

 protected:
  mutex mu_;
  int64 num_random_samples_ TF_GUARDED_BY(mu_) = 0;
};

// Always generates the specified seed values.
class FixedSeedGenerator : public SeedGenerator {
 public:
  FixedSeedGenerator(int64 seed, int64 seed2) : seed_(seed), seed2_(seed2) {}

  std::string DebugString() const override;
  void GenerateSeeds(int64* seed1, int64* seed2) override;
  void Reset() override {}

 private:
  const int64 seed_;
  const int64 seed2_;
};

// Generates different (but deterministically chosen) seed values.
class RandomSeedGenerator : public SeedGenerator {
 public:
  RandomSeedGenerator(int64 seed, int64 seed2)
      : seed_(seed),
        seed2_(seed2),
        parent_generator_(seed, seed2),
        generator_(&parent_generator_) {}

  std::string DebugString() const override;
  void GenerateSeeds(int64* seed1, int64* seed2) override;
  void Reset() override;

 private:
  const int64 seed_;
  const int64 seed2_;
  random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_
      TF_GUARDED_BY(mu_);
};

// Creates an instance of seed generator resource and transfers ownership
// to the caller.
class AnonymousSeedGeneratorHandleOp
    : public AnonymousResourceOp<SeedGenerator> {
 public:
  explicit AnonymousSeedGeneratorHandleOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  string name() override;
  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        SeedGenerator** resource) override;

  int64 seed_;
  int64 seed2_;
  bool reshuffle_;
};

// Deletes an instance of seed generator resource.
class DeleteSeedGeneratorOp : public OpKernel {
 public:
  explicit DeleteSeedGeneratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_
