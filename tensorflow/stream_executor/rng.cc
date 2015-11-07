#include "tensorflow/stream_executor/rng.h"

#include "tensorflow/stream_executor/platform/logging.h"

namespace perftools {
namespace gputools {
namespace rng {

bool RngSupport::CheckSeed(const uint8 *seed, uint64 seed_bytes) {
  CHECK(seed != nullptr);

  if (seed_bytes < kMinSeedBytes) {
    LOG(INFO) << "Insufficient RNG seed data specified: " << seed_bytes
              << ". At least " << RngSupport::kMinSeedBytes
              << " bytes are required.";
    return false;
  }

  if (seed_bytes > kMaxSeedBytes) {
    LOG(INFO) << "Too much RNG seed data specified: " << seed_bytes
              << ". At most " << RngSupport::kMaxSeedBytes
              << " bytes may be provided.";
    return false;
  }

  return true;
}

#if defined(__APPLE__)
const int RngSupport::kMinSeedBytes;
const int RngSupport::kMaxSeedBytes;
#endif

}  // namespace rng
}  // namespace gputools
}  // namespace perftools
