#ifndef TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_TEST_UTILS_H_
#define TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_TEST_UTILS_H_

#include <algorithm>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace random {

// Return a random seed.
inline uint64 GetTestSeed() { return New64(); }

// A utility function to fill the given array with samples from the given
// distribution.
template <class Distribution>
void FillRandoms(PhiloxRandom gen, typename Distribution::ResultElementType* p,
                 int64 size) {
  const int granularity = Distribution::kResultElementCount;

  CHECK(size % granularity == 0) << " size: " << size
                                 << " granularity: " << granularity;

  Distribution dist;
  for (int i = 0; i < size; i += granularity) {
    const auto sample = dist(&gen);
    std::copy(&sample[0], &sample[0] + granularity, &p[i]);
  }
}

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_TEST_UTILS_H_
