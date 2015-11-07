#include "tensorflow/core/lib/random/random.h"

#include <random>
#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace random {

std::mt19937_64* InitRng() {
  std::random_device device("/dev/random");
  return new std::mt19937_64(device());
}

uint64 New64() {
  static std::mt19937_64* rng = InitRng();
  static mutex mu;
  mutex_lock l(mu);
  return (*rng)();
}

}  // namespace random
}  // namespace tensorflow
