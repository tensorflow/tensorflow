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

#include "tensorflow/compiler/xla/mlir/utils/error_util.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status.h"

namespace mlir {
namespace {

TEST(ErrorUtilTest, BaseScopedDiagnosticHandler) {
  MLIRContext context;
  auto id = StringAttr::get(&context, "//tensorflow/python/test.py");
  auto loc = FileLineColLoc::get(&context, id, 0, 0);

  // Test OK without diagnostic gets passed through.
  {
    TF_EXPECT_OK(tsl::FromAbslStatus(
        BaseScopedDiagnosticHandler(&context).Combine(absl::OkStatus())));
  }

  // Verify diagnostics are captured as Unknown status.
  {
    BaseScopedDiagnosticHandler handler(&context);
    emitError(loc) << "Diagnostic message";
    ASSERT_TRUE(absl::IsUnknown(handler.ConsumeStatus()));
  }

  // Verify passed in errors are propagated.
  {
    absl::Status err = absl::InternalError("Passed in error");
    ASSERT_TRUE(
        absl::IsInternal(BaseScopedDiagnosticHandler(&context).Combine(err)));
  }

  // Verify diagnostic reported are append to passed in error.
  {
    auto function = [&]() {
      emitError(loc) << "Diagnostic message reported";
      emitError(loc) << "Second diagnostic message reported";
      return absl::InternalError("Passed in error");
    };

    BaseScopedDiagnosticHandler ssdh(&context);
    absl::Status s = ssdh.Combine(function());
    ASSERT_TRUE(absl::IsInternal(s));
    EXPECT_TRUE(absl::StrContains(s.message(), "Passed in error"));
    EXPECT_TRUE(absl::StrContains(s.message(), "Diagnostic message reported"));
    EXPECT_TRUE(
        absl::StrContains(s.message(), "Second diagnostic message reported"));
  }
}

}  // namespace
}  // namespace mlir
