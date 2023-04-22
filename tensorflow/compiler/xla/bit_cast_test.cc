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

#include "tensorflow/compiler/xla/bit_cast.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::Eigen::half;
using ::tensorflow::bfloat16;

TEST(BitCastTest, BackAndForth) {
  for (uint32 n = 0; n < 0x10000; ++n) {
    uint16 initial_rep = n;
    bfloat16 float_val = BitCast<bfloat16>(initial_rep);
    uint16 final_rep = BitCast<uint16>(float_val);

    EXPECT_EQ(initial_rep, final_rep);
  }

  for (uint32 n = 0; n < 0x10000; ++n) {
    uint16 initial_rep = n;
    half float_val = BitCast<half>(initial_rep);
    uint16 final_rep = BitCast<uint16>(float_val);

    EXPECT_EQ(initial_rep, final_rep);
  }
}

}  // namespace
}  // namespace xla
