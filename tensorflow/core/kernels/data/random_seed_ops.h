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

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {

// Represents a pair of random seeds. By TensorFlow convention, if both seeds
// are 0, then pseudo-random values are used instead.
class RandomSeeds {
 public:
  RandomSeeds(int64_t seed, int64_t seed2)
      : input_seed_(seed),
        input_seed2_(seed2),
        seed_((seed | seed2) == 0 ? random::New64() : seed),
        seed2_((seed | seed2) == 0 ? random::New64() : seed2) {}

  int64_t input_seed() const { return input_seed_; }
  int64_t input_seed2() const { return input_seed2_; }
  int64_t seed() const { return seed_; }
  int64_t seed2() const { return seed2_; }

 private:
  const int64_t input_seed_;
  const int64_t input_seed2_;
  const int64_t seed_;
  const int64_t seed2_;
};

// Base class for seed generator resources. Subclasses customize how seeds are
// generated.
class SeedGenerator {
 public:
  virtual ~SeedGenerator() {}

  virtual int64_t seed() const = 0;
  virtual int64_t seed2() const = 0;
  virtual bool reshuffle_each_iteration() const = 0;

  virtual void GenerateSeeds(int64_t* seed1, int64_t* seed2) = 0;
  virtual void Reset() = 0;

  virtual int64_t num_random_samples() const {
    tf_shared_lock l(mu_);
    return num_random_samples_;
  }
  virtual void set_num_random_samples(int64_t num_random_samples) {
    mutex_lock l(mu_);
    num_random_samples_ = num_random_samples;
  }

 protected:
  mutable mutex mu_;
  int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
};

// A resource wrapping a shared instance of a seed generator.
class SeedGeneratorManager : public ResourceBase {
 public:
  explicit SeedGeneratorManager(SeedGenerator* seed_generator)
      : seed_generator_(seed_generator) {}

  std::string DebugString() const override;

  std::shared_ptr<SeedGenerator> get() { return seed_generator_; }

 private:
  std::shared_ptr<SeedGenerator> seed_generator_;
};

// Always generates the specified seed values.
class FixedSeedGenerator : public SeedGenerator {
 public:
  explicit FixedSeedGenerator(RandomSeeds seeds) : seeds_(std::move(seeds)) {}

  int64_t seed() const override { return seeds_.seed(); }
  int64_t seed2() const override { return seeds_.seed(); }
  bool reshuffle_each_iteration() const override { return false; }

  void GenerateSeeds(int64_t* seed1, int64_t* seed2) override;
  void Reset() override {}

 private:
  const RandomSeeds seeds_;
};

// Generates different (but deterministically chosen) seed values.
class RandomSeedGenerator : public SeedGenerator {
 public:
  explicit RandomSeedGenerator(RandomSeeds seeds)
      : seeds_(std::move(seeds)),
        parent_generator_(seeds_.seed(), seeds_.seed2()),
        generator_(&parent_generator_) {}

  int64_t seed() const override { return seeds_.seed(); }
  int64_t seed2() const override { return seeds_.seed2(); }
  bool reshuffle_each_iteration() const override { return true; }

  void GenerateSeeds(int64_t* seed1, int64_t* seed2) override;
  void Reset() override;

 private:
  const RandomSeeds seeds_;
  random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_
      TF_GUARDED_BY(mu_);
};

// Creates an instance of seed generator resource and transfers ownership
// to the caller.
class AnonymousSeedGeneratorHandleOp
    : public AnonymousResourceOp<SeedGeneratorManager> {
 public:
  explicit AnonymousSeedGeneratorHandleOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  string name() override;
  absl::Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, SeedGeneratorManager** manager) override;

  mutex mu_;
  std::unique_ptr<RandomSeeds> seeds_ TF_GUARDED_BY(mu_);
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
