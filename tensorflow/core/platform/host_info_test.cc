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

#include "tensorflow/core/platform/host_info.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(IsSockAddrValid, ValidSockAddr) {
  EXPECT_TRUE(port::IsValidSockAddr(string("1.2.3.4"), string("1234")));
  EXPECT_TRUE(port::IsValidSockAddr(string("0.0.0.0"), string("1234")));
  EXPECT_TRUE(port::IsValidSockAddr(string("1.2.3.4"), string("0")));
  EXPECT_TRUE(port::IsValidSockAddr(string("0.0.0.0"), string("0")));
}

TEST(IsSockAddrValid, InvalidSockAddr) {
  EXPECT_FALSE(port::IsValidSockAddr(string("100.2.3.0."), string("2223")));
  EXPECT_FALSE(port::IsValidSockAddr(string("100.2.3"), string("2223")));
  EXPECT_FALSE(port::IsValidSockAddr(string("100.2.3.0"), string("-1")));
  EXPECT_FALSE(port::IsValidSockAddr(string("100.2.3.0."), string("-2223")));
}

}  // namespace
}  // namespace tensorflow
