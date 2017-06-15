/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using ConversionsTest = HloTestBase;

TEST_F(ConversionsTest, Int64ToInt32) {
  int64 src[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9
  };
  std::vector<char> res = ConvertInt64ToInt32((void*)src, sizeof(src));

  EXPECT_EQ(20 * sizeof(int32), res.size());

  int32* d = reinterpret_cast<int32*>(res.data());
  EXPECT_EQ(0, d[0]);
  EXPECT_EQ(1, d[1]);
  EXPECT_EQ(2, d[2]);
  EXPECT_EQ(0, d[10]);
  EXPECT_EQ(-1, d[11]);
  EXPECT_EQ(-9, d[19]);
}

TEST_F(ConversionsTest, Int32ToInt64) {
  int32 src[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9
  };
  std::vector<char> res = ConvertInt32ToInt64((void*)src, sizeof(src));

  EXPECT_EQ(20 * sizeof(int64), res.size());

  int64* d = reinterpret_cast<int64*>(res.data());
  EXPECT_EQ(0, d[0]);
  EXPECT_EQ(1, d[1]);
  EXPECT_EQ(2, d[2]);
  EXPECT_EQ(0, d[10]);
  EXPECT_EQ(-1, d[11]);
  EXPECT_EQ(-9, d[19]);
}


}
}
}
