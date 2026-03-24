/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/mlir/utils/error_util.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace mlir {
namespace {

class ErrorUtilTest : public ::testing::Test {
 protected:
  ErrorUtilTest()
      : id_(StringAttr::get(&context_, "test.py")),
        loc_(FileLineColLoc::get(&context_, id_, 0, 0)) {}

  MLIRContext context_;
  StringAttr id_;
  FileLineColLoc loc_;
};

using BaseScopedDiagnosticHandlerTest = ErrorUtilTest;

TEST_F(BaseScopedDiagnosticHandlerTest, OkWithoutDiagnosticGetsPassedThrough) {
  TF_EXPECT_OK(
      BaseScopedDiagnosticHandler(&context_).Combine(absl::OkStatus()));
}

TEST_F(BaseScopedDiagnosticHandlerTest,
       VerifyDiagnosticsAreCapturedAsUnknownStatus) {
  BaseScopedDiagnosticHandler handler(&context_);
  emitError(loc_) << "Diagnostic message";
  ASSERT_TRUE(absl::IsUnknown(handler.ConsumeStatus()));
}

TEST_F(BaseScopedDiagnosticHandlerTest, VerifyPassedInErrorsArePropagated) {
  const absl::Status err = absl::InternalError("Passed in error");
  ASSERT_TRUE(
      absl::IsInternal(BaseScopedDiagnosticHandler(&context_).Combine(err)));
}

TEST_F(BaseScopedDiagnosticHandlerTest,
       VerifyThatReportedDiagnosticsAreAppendedToPassedInError) {
  BaseScopedDiagnosticHandler ssdh(&context_);
  emitError(loc_) << "Diagnostic message reported";
  emitError(loc_) << "Second diagnostic message reported";
  const absl::Status s = ssdh.Combine(absl::InternalError("Passed in error"));
  ASSERT_TRUE(absl::IsInternal(s));
  EXPECT_TRUE(absl::StrContains(s.message(), "Passed in error"));
  EXPECT_TRUE(absl::StrContains(s.message(), "Diagnostic message reported"));
  EXPECT_TRUE(
      absl::StrContains(s.message(), "Second diagnostic message reported"));
}

}  // namespace
}  // namespace mlir
