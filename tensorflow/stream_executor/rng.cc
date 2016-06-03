/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
