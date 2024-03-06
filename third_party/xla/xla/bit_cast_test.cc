/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/bit_cast.h"

#include <cstdint>

#include "Eigen/Core"  // from @eigen_archive
#include "xla/test.h"
#include "tsl/platform/bfloat16.h"

namespace xla {
namespace {

using ::Eigen::half;
using ::tsl::bfloat16;

TEST(BitCastTest, BackAndForth) {
  for (uint32_t n = 0; n < 0x10000; ++n) {
    uint16_t initial_rep = n;
    bfloat16 float_val = BitCast<bfloat16>(initial_rep);
    uint16_t final_rep = BitCast<uint16_t>(float_val);

    EXPECT_EQ(initial_rep, final_rep);
  }

  for (uint32_t n = 0; n < 0x10000; ++n) {
    uint16_t initial_rep = n;
    half float_val = BitCast<half>(initial_rep);
    uint16_t final_rep = BitCast<uint16_t>(float_val);

    EXPECT_EQ(initial_rep, final_rep);
  }
}

}  // namespace
}  // namespace xla
