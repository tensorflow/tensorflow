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
// Testing proper operation of the stacktrace handler.

#include <csignal>

#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace {

TEST(StacktraceHandlerTest, GeneratesStacktrace) {
  // Just make sure we can detect one of the calls in testing stack.
  EXPECT_DEATH(raise(SIGABRT), "testing::internal::UnitTestImpl::RunAllTests");
}

}  // namespace
}  // namespace tsl
