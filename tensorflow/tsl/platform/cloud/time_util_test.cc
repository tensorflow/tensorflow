/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/platform/cloud/time_util.h"

#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {

TEST(TimeUtil, ParseRfc3339Time) {
  int64_t mtime_nsec;
  TF_EXPECT_OK(ParseRfc3339Time("2016-04-29T23:15:24.896Z", &mtime_nsec));
  // Compare milliseconds instead of nanoseconds.
  EXPECT_NEAR(1461971724896, mtime_nsec / 1000 / 1000, 1);
}

TEST(TimeUtil, ParseRfc3339Time_ParseError) {
  int64_t mtime_nsec;
  EXPECT_EQ("Unrecognized RFC 3339 time format: 2016-04-29",
            ParseRfc3339Time("2016-04-29", &mtime_nsec).error_message());
}

}  // namespace tsl
