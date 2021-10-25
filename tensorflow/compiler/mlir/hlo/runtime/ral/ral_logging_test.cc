/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing ppermissions and
limitations under the License.
==============================================================================*/

#include "ral/ral_logging.h"

#ifndef PLATFORM_GOOGLE
#include <gtest/gtest.h>
#else
#include "testing/base/public/gunit.h"
#endif

namespace testing {

using ::mlir::disc_ral::ERROR;
using ::mlir::disc_ral::FATAL;
using ::mlir::disc_ral::INFO;
using ::mlir::disc_ral::WARNING;
using ::mlir::disc_ral::internal::LogMessage;

TEST(Logging, Log) {
  EXPECT_TRUE(DISC_VLOG_IS_ON(0));
  EXPECT_FALSE(DISC_VLOG_IS_ON(1));

  LogMessage logger(__FILE__, __LINE__, ERROR);
  logger << "Hello";
  EXPECT_EQ(logger.GetFilterStringForTesting(INFO), "Hello");
  EXPECT_EQ(logger.GetFilterStringForTesting(WARNING), "Hello");
  EXPECT_EQ(logger.GetFilterStringForTesting(ERROR), "Hello");
  EXPECT_EQ(logger.GetFilterStringForTesting(FATAL), "");
}

}  // namespace testing
