/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/rng_util.h"

#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace tflite {
namespace {

using tflite::rng::Philox4x32;
using tflite::rng::Threefry2x32;

// Test cases are from the test code of the original reference implementation of
// Threefry. For the values, see
// https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
TEST(RngUtilTest, Threefry2x32Test) {
  std::array<uint32_t, 2> results = Threefry2x32(0, 0, {0, 0});
  std::array<uint32_t, 2> expected = {0x6B200159u, 0x99BA4EFEu};
  ASSERT_EQ(results, expected);

  uint32_t u32_max = std::numeric_limits<uint32_t>::max();
  results = Threefry2x32(u32_max, u32_max, {u32_max, u32_max});
  expected = {0x1CB996FCu, 0xBB002BE7u};
  ASSERT_EQ(results, expected);

  results = Threefry2x32(0x13198A2Eu, 0x03707344u, {0x243F6A88u, 0x85A308D3u});
  expected = {0xC4923A9Cu, 0x483DF7A0u};
  ASSERT_EQ(results, expected);
}
// Test cases are from the test code of the original reference implementation of
// Philox. For the values, see
// https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_philox.cpp#L50
TEST(RngUtilTest, Philox4x32Test) {
  std::array<uint32_t, 4> results = Philox4x32(0, 0, {0, 0, 0, 0});
  std::array<uint32_t, 4> expected = {0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu,
                                      0x9B00DBD8u};
  ASSERT_EQ(results, expected);

  uint32_t u32_max = std::numeric_limits<uint32_t>::max();
  results = Philox4x32(u32_max, u32_max, {u32_max, u32_max, u32_max, u32_max});
  expected = {0x408F276Du, 0x41C83B0Eu, 0xA20BC7C6u, 0x6D5451FDu};
  ASSERT_EQ(results, expected);

  results = Philox4x32(0xA4093822u, 0x299F31D0u,
                       {0x243F6A88u, 0x85A308D3u, 0x13198A2Eu, 0x03707344u});
  expected = {0xD16CFE09u, 0x94FDCCEBu, 0x5001E420u, 0x24126EA1u};
  ASSERT_EQ(results, expected);
}

}  // namespace
}  // namespace tflite
