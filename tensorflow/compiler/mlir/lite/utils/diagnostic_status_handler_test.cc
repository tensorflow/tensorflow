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

#include "tensorflow/compiler/mlir/lite/utils/diagnostic_status_handler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::TFL {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(DiagnosticStatusHandlerTest, OkWithoutDiagnostics) {
  MLIRContext context;
  DiagnosticStatusHandler handler(&context);
  EXPECT_TRUE(handler.ok());
  EXPECT_THAT(handler.ConsumeStatus(), IsOk());
}

TEST(DiagnosticStatusHandlerTest, CollectsErrors) {
  MLIRContext context;
  DiagnosticStatusHandler handler(&context);
  Location loc = FileLineColLoc::get(&context, "file.mlir", 3, 7);
  emitError(loc) << "first problem";
  emitError(loc) << "second problem";

  EXPECT_FALSE(handler.ok());
  EXPECT_THAT(handler.ConsumeStatus(),
              StatusIs(absl::StatusCode::kUnknown,
                       AllOf(HasSubstr("file.mlir:3:7: error: first problem"),
                             HasSubstr("second problem"))));
  // Consuming clears the collected errors.
  EXPECT_TRUE(handler.ok());
  EXPECT_THAT(handler.ConsumeStatus(), IsOk());
}

TEST(DiagnosticStatusHandlerTest, IgnoresWarningsAndRemarks) {
  MLIRContext context;
  DiagnosticStatusHandler handler(&context);
  Location loc = UnknownLoc::get(&context);
  emitWarning(loc) << "a warning";
  emitRemark(loc) << "a remark";
  EXPECT_TRUE(handler.ok());

  emitError(loc) << "an error";
  EXPECT_THAT(handler.ConsumeStatus(),
              StatusIs(absl::StatusCode::kUnknown,
                       AllOf(HasSubstr("an error"), Not(HasSubstr("warning")),
                             Not(HasSubstr("remark")))));
}

TEST(DiagnosticStatusHandlerTest, PropagatesWhenRequested) {
  MLIRContext context;
  int outer_errors = 0;
  ScopedDiagnosticHandler outer(&context, [&](Diagnostic& diag) {
    ++outer_errors;
    return success();
  });
  Location loc = UnknownLoc::get(&context);
  {
    DiagnosticStatusHandler handler(&context);
    emitError(loc) << "kept";
    EXPECT_THAT(handler.ConsumeStatus(), Not(IsOk()));
  }
  EXPECT_EQ(outer_errors, 0);
  {
    DiagnosticStatusHandler handler(&context, /*propagate=*/true);
    emitError(loc) << "propagated";
    EXPECT_THAT(handler.ConsumeStatus(), Not(IsOk()));
  }
  EXPECT_EQ(outer_errors, 1);
}

}  // namespace
}  // namespace mlir::TFL
