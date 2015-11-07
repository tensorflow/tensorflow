#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

Status GuardedPhiloxRandom::Init(OpKernelConstruction* context) {
  // Grab seed Attrs.
  int64 seed, seed2;
  auto status = context->GetAttr("seed", &seed);
  if (!status.ok()) return status;
  status = context->GetAttr("seed2", &seed2);
  if (!status.ok()) return status;

  // Initialize with the given seeds
  Init(seed, seed2);
  return Status::OK();
}

void GuardedPhiloxRandom::Init(int64 seed, int64 seed2) {
  CHECK(!initialized_);
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = random::New64();
    seed2 = random::New64();
  }
  mutex_lock lock(mu_);
  generator_ = random::PhiloxRandom(seed, seed2);
  initialized_ = true;
}

random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples) {
  CHECK(initialized_);
  mutex_lock lock(mu_);
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}

}  // namespace tensorflow
