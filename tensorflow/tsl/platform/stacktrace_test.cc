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
// Testing proper operation of the stacktrace handler.

#include "tensorflow/tsl/platform/stacktrace.h"

#include <string>

#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

#if defined(TF_HAS_STACKTRACE)

TEST(StacktraceTest, StacktraceWorks) {
  std::string stacktrace = CurrentStackTrace();
  LOG(INFO) << "CurrentStackTrace():\n" << stacktrace;
  std::string expected_frame = "testing::internal::UnitTestImpl::RunAllTests";
  EXPECT_NE(stacktrace.find(expected_frame), std::string::npos);
}

#endif  // defined(TF_HAS_STACKTRACE)

}  // namespace
}  // namespace tsl
