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

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace {

TEST(IntegralTypes, Basic) {
  EXPECT_EQ(1, sizeof(int8));
  EXPECT_EQ(2, sizeof(int16));
  EXPECT_EQ(4, sizeof(int32));
  EXPECT_EQ(8, sizeof(int64_t));

  EXPECT_EQ(1, sizeof(uint8));
  EXPECT_EQ(2, sizeof(uint16));
  EXPECT_EQ(4, sizeof(uint32));
  EXPECT_EQ(8, sizeof(uint64));
}

TEST(IntegralTypes, MinAndMaxConstants) {
  EXPECT_EQ(static_cast<uint8>(kint8min), static_cast<uint8>(kint8max) + 1);
  EXPECT_EQ(static_cast<uint16>(kint16min), static_cast<uint16>(kint16max) + 1);
  EXPECT_EQ(static_cast<uint32>(kint32min), static_cast<uint32>(kint32max) + 1);
  EXPECT_EQ(static_cast<uint64>(kint64min), static_cast<uint64>(kint64max) + 1);

  EXPECT_EQ(0, static_cast<uint8>(kuint8max + 1));
  EXPECT_EQ(0, static_cast<uint16>(kuint16max + 1));
  EXPECT_EQ(0, static_cast<uint32>(kuint32max + 1));
  EXPECT_EQ(0, static_cast<uint64>(kuint64max + 1));
}

}  // namespace
}  // namespace tsl
