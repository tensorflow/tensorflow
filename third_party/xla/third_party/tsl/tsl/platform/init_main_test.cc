/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/init_main.h"

#include <cstring>

#include "xla/tsl/platform/test.h"

namespace tsl {
namespace port {
namespace {

class InitMainTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
#if !defined(PLATFORM_GOOGLE)
    char* argv0 = const_cast<char*>("dummy");
    char** argv = &argv0;
    int argc = 1;
    tsl::port::InitMain(argv[0], &argc, &argv);
#endif
  }
};

TEST_F(InitMainTest, GetArgvs) {
  const auto& argvs = GetArgvs();
  EXPECT_FALSE(argvs.empty());
  EXPECT_EQ(argvs[0], GetArgv0());
}

TEST_F(InitMainTest, GetArgv0) {
  const char* argv0 = GetArgv0();
  ASSERT_NE(argv0, nullptr);
  EXPECT_NE(strlen(argv0), 0);
}

}  // namespace
}  // namespace port
}  // namespace tsl
