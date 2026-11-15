/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/rng_state_lib.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/numeric/int128.h"

namespace xla::cpu {
namespace {

TEST(RngStateTest, SharedStateAdvancesAcrossMultipleDeltas) {
  RngState rng;
  absl::int128 first;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&first), /*delta=*/131072);

  absl::int128 second;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&second), /*delta=*/65536);
  EXPECT_EQ(second, first + 131072);

  absl::int128 third;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&third), /*delta=*/0);
  EXPECT_EQ(third, second + 65536);
}

TEST(RngStateTest, ShardsStateByDeviceOrdinalAndClampsNegativeOrdinalToZero) {
  RngState rng;
  absl::int128 dev0_first;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&dev0_first),
                        /*delta=*/1024, /*device_ordinal=*/0);

  absl::int128 dev1_first;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&dev1_first),
                        /*delta=*/2048, /*device_ordinal=*/1);
  EXPECT_EQ(dev1_first, dev0_first);

  absl::int128 dev_neg_second;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&dev_neg_second),
                        /*delta=*/512, /*device_ordinal=*/-1);
  EXPECT_EQ(dev_neg_second, dev0_first + 1024);

  absl::int128 dev1_second;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&dev1_second), /*delta=*/0,
                        /*device_ordinal=*/1);
  EXPECT_EQ(dev1_second, dev1_first + 2048);
}

// PjRt CPU passes global device ids (2048 * process_index + local id) as the
// device ordinal, so large and sparse ordinals must get independent counters.
// 1 << 40 aliases ordinal 0 if the ordinal is narrowed to 32 bits.
TEST(RngStateTest, LargeDeviceOrdinalsGetIndependentCounters) {
  RngState rng;
  absl::int128 dev0_first;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&dev0_first),
                        /*delta=*/1024, /*device_ordinal=*/0);

  for (int64_t ordinal : {int64_t{2048} * 1000 + 3, int64_t{1} << 40}) {
    absl::int128 first;
    rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&first), /*delta=*/7,
                          ordinal);
    EXPECT_EQ(first, dev0_first) << "ordinal " << ordinal;

    absl::int128 second;
    rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&second), /*delta=*/0,
                          ordinal);
    EXPECT_EQ(second, first + 7) << "ordinal " << ordinal;
  }

  absl::int128 dev0_second;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&dev0_second),
                        /*delta=*/0, /*device_ordinal=*/0);
  EXPECT_EQ(dev0_second, dev0_first + 1024);
}

TEST(RngStateTest, LegacySingleArgumentOverloadUsesDefaultDelta) {
  RngState rng(/*default_delta=*/17);
  absl::int128 first;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&first));

  absl::int128 second;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&second));
  EXPECT_EQ(second, first + 17);
}

TEST(RngStateTest, LegacyForwardingWrappersShareTargetCounter) {
  RngState shared;
  RngState thunk0(/*default_delta=*/1, &shared);
  RngState thunk1(/*default_delta=*/3, &shared);
  RngState thunk2(/*default_delta=*/1, &shared);

  absl::int128 s0;
  thunk0.GetAndUpdateState(reinterpret_cast<uint64_t*>(&s0));

  absl::int128 s1;
  thunk1.GetAndUpdateState(reinterpret_cast<uint64_t*>(&s1));
  EXPECT_EQ(s1, s0 + 1);

  absl::int128 s2;
  thunk2.GetAndUpdateState(reinterpret_cast<uint64_t*>(&s2));
  EXPECT_EQ(s2, s1 + 3);

  absl::int128 s3;
  thunk0.GetAndUpdateState(reinterpret_cast<uint64_t*>(&s3));
  EXPECT_EQ(s3, s2 + 1);
}

}  // namespace
}  // namespace xla::cpu
