#ifndef TENSORFLOW_LIB_RANDOM_SIMPLE_PHILOX_H_
#define TENSORFLOW_LIB_RANDOM_SIMPLE_PHILOX_H_

#include <math.h>
#include <string.h>
#include <algorithm>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace random {

// A simple imperative interface to Philox
class SimplePhilox {
 public:
  PHILOX_DEVICE_INLINE
  explicit SimplePhilox(PhiloxRandom* gen) : single_(gen) {}

  // 32 random bits
  PHILOX_DEVICE_INLINE uint32 Rand32() { return single_(); }

  // 64 random bits
  PHILOX_DEVICE_INLINE uint64 Rand64() {
    const uint32 lo = single_(), hi = single_();
    return lo | static_cast<uint64>(hi) << 32;
  }

  // Uniform float in [0, 1)
  PHILOX_DEVICE_INLINE float RandFloat() { return Uint32ToFloat(single_()); }

  // Uniform double in [0, 1)
  PHILOX_DEVICE_INLINE double RandDouble() {
    const uint32 x0 = single_(), x1 = single_();
    return Uint64ToDouble(x0, x1);
  }

  // Uniform integer in [0, n).
  // Uses rejection sampling, so may need more than one 32-bit sample.
  uint32 Uniform(uint32 n);

  // Approximately uniform integer in [0, n).
  // Uses rejection sampling, so may need more than one 64-bit sample.
  uint64 Uniform64(uint64 n);

  // True with probability 1/n.
  bool OneIn(uint32 n) { return Uniform(n) == 0; }

  // Skewed: pick "base" uniformly from range [0,max_log] and then
  // return "base" random bits.  The effect is to pick a number in the
  // range [0,2^max_log-1] with bias towards smaller numbers.
  uint32 Skewed(int max_log);

 private:
  SingleSampleAdapter<PhiloxRandom> single_;
};

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_SIMPLE_PHILOX_H_
