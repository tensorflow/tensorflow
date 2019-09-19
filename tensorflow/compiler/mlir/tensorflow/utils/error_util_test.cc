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

#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

#include "llvm/ADT/Twine.h"
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace mlir {
namespace {

using testing::HasSubstr;

TEST(ErrorUtilTest, StatusScopedDiagnosticHandler) {
  MLIRContext context;
  auto id = Identifier::get("test.cc", &context);
  auto loc = FileLineColLoc::get(id, 0, 0, &context);

  // Test OK without diagnostic gets passed through.
  {
    TF_ASSERT_OK(StatusScopedDiagnosticHandler(&context).Combine(Status::OK()));
  }

  // Verify diagnostics are captured as Unknown status.
  {
    StatusScopedDiagnosticHandler handler(&context);
    emitError(loc) << "Diagnostic message";
    ASSERT_TRUE(tensorflow::errors::IsUnknown(handler.ConsumeStatus()));
  }

  // Verify passed in errors are propagated.
  {
    Status err = tensorflow::errors::Internal("Passed in error");
    ASSERT_TRUE(tensorflow::errors::IsInternal(
        StatusScopedDiagnosticHandler(&context).Combine(err)));
  }

  // Verify diagnostic reported are append to passed in error.
  {
    auto function = [&]() {
      emitError(loc) << "Diagnostic message reported";
      emitError(loc) << "Second diagnostic message reported";
      return tensorflow::errors::Internal("Passed in error");
    };
    Status s = StatusScopedDiagnosticHandler(&context).Combine(function());
    ASSERT_TRUE(tensorflow::errors::IsInternal(s));
    EXPECT_THAT(s.error_message(), HasSubstr("Passed in error"));
    EXPECT_THAT(s.error_message(), HasSubstr("Diagnostic message reported"));
    EXPECT_THAT(s.error_message(),
                HasSubstr("Second diagnostic message reported"));
  }
}

}  // namespace
}  // namespace mlir
