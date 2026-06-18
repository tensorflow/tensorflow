/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/error/check.h"

#include <gtest/gtest.h>
#include "xla/error/debug_me_context_util.h"
#include "xla/tsl/platform/debug_me_context.h"

namespace xla::error {
namespace {

using ::testing::ContainsRegex;

TEST(CheckTest, TrueCondition_DoesNotCrash) { XLA_CHECK(true); }

TEST(CheckTest, TrueConditionWithExpression_EvaluatesOnlyOnce) {
  int x = 0;
  XLA_CHECK(++x == 1);
  EXPECT_EQ(x, 1);
}

TEST(CheckTest, TrueConditionWithMessageExpression_DoesNotEvaluate) {
  int x = 0;
  XLA_CHECK(true) << ++x;
  EXPECT_EQ(x, 0);
}

TEST(CheckTest, FalseCondition_Crashes) {
  EXPECT_DEATH(
      XLA_CHECK(false),
      ContainsRegex(
          "check_test.cc:[0-9]+] E0012: Internal: Check failed: false"));
}

TEST(CheckTest, FalseConditionWithMessage_Crashes) {
  EXPECT_DEATH(XLA_CHECK(false) << "custom error message",
               ContainsRegex(
                   "check_test.cc:[0-9]+] E0012: Internal: Check failed: false "
                   "custom error message"));
}

TEST(CheckTest, FalseConditionWithExpression_Crashes) {
  int x = 0;
  EXPECT_DEATH(
      XLA_CHECK(x == 1),
      ContainsRegex(
          "check_test.cc:[0-9]+] E0012: Internal: Check failed: x == 1"));
}

TEST(CheckTest, FalseCondition_WithMessageAndDebugContext_Crashes) {
  tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
                                                 "MyTestPass");
  EXPECT_DEATH(XLA_CHECK(false) << "custom error message",
               ContainsRegex(
                   "check_test.cc:[0-9]+] E0012: Internal: Check failed: false "
                   "custom error "
                   "message\nDebugMeContext:\nHLO Passes: MyTestPass"));
}

TEST(CheckTest, QFatalTrueCondition_DoesNotCrash) { XLA_QCHECK(true); }

TEST(CheckTest, QFatalTrueConditionWithExpression_EvaluatesOnlyOnce) {
  int x = 0;
  XLA_QCHECK(++x == 1);
  EXPECT_EQ(x, 1);
}

TEST(CheckTest, QFatalTrueConditionWithMessageExpression_DoesNotEvaluate) {
  int x = 0;
  XLA_QCHECK(true) << ++x;
  EXPECT_EQ(x, 0);
}

TEST(CheckTest, QFatalCondition_Crashes) {
  EXPECT_DEATH(
      XLA_QCHECK(false),
      ContainsRegex(
          "check_test.cc:[0-9]+] E0012: Internal: Check failed: false"));
}

TEST(CheckTest, QFatalConditionWithMessage_Crashes) {
  EXPECT_DEATH(XLA_QCHECK(false) << "custom error message",
               ContainsRegex(
                   "check_test.cc:[0-9]+] E0012: Internal: Check failed: false "
                   "custom error message"));
}

TEST(CheckTest, QFatalConditionWithExpression_Crashes) {
  int x = 0;
  EXPECT_DEATH(
      XLA_QCHECK(x == 1),
      ContainsRegex(
          "check_test.cc:[0-9]+] E0012: Internal: Check failed: x == 1"));
}

TEST(CheckTest, QFatalCondition_WithMessageAndDebugContext_Crashes) {
  tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
                                                 "MyTestPass");
  EXPECT_DEATH(XLA_QCHECK(false) << "custom error message",
               ContainsRegex(
                   "check_test.cc:[0-9]+] E0012: Internal: Check failed: false "
                   "custom error "
                   "message\nDebugMeContext:\nHLO Passes: MyTestPass"));
}

TEST(CheckTest, DCheckTrueNoDebug_DoesNotCrash) { XLA_DCHECK(true); }

}  // namespace
}  // namespace xla::error
