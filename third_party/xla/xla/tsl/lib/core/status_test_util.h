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

#ifndef XLA_TSL_LIB_CORE_STATUS_TEST_UTIL_H_
#define XLA_TSL_LIB_CORE_STATUS_TEST_UTIL_H_

#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

// Macros for testing the results of functions that return tensorflow::Status.
#define TF_EXPECT_OK(statement) EXPECT_THAT((statement), ::tsl::testing::IsOk())
#define TF_ASSERT_OK(statement) ASSERT_THAT((statement), ::tsl::testing::IsOk())

// There are no EXPECT_NOT_OK/ASSERT_NOT_OK macros since they would not
// provide much value (when they fail, they would just print the OK status
// which conveys no more information than EXPECT_FALSE(status.ok());
// If you want to check for particular errors, a better alternative is with
// status matchers:
// EXPECT_THAT(s, tensorflow::testing::StatusIs(status.code(), "message"));

#endif  // XLA_TSL_LIB_CORE_STATUS_TEST_UTIL_H_
