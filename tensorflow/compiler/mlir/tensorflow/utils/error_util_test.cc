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
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace mlir {
namespace {

using testing::HasSubstr;

TEST(ErrorUtilTest, StatusScopedDiagnosticHandler) {
  MLIRContext context;
  auto id = Identifier::get("//tensorflow/python/test.py", &context);
  auto loc = FileLineColLoc::get(&context, id, 0, 0);

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
    StatusScopedDiagnosticHandler ssdh(&context);
    Status s = ssdh.Combine(function());
    ASSERT_TRUE(tensorflow::errors::IsInternal(s));
    EXPECT_THAT(s.error_message(), HasSubstr("Passed in error"));
    EXPECT_THAT(s.error_message(), HasSubstr("Diagnostic message reported"));
    EXPECT_THAT(s.error_message(),
                HasSubstr("Second diagnostic message reported"));
  }
}

TEST(ErrorUtilTest, StatusScopedDiagnosticHandlerWithFilter) {
  // Filtering logic is based on tensorflow::IsInternalFrameForFilename()
  // Note we are surfacing the locations that are NOT internal frames
  // so locations that fail IsInternalFrameForFilename() evaluation pass the
  // filter.

  // These locations will fail the IsInternalFrameForFilename() check so will
  // pass the filter.
  MLIRContext context;
  auto id =
      Identifier::get("//tensorflow/python/keras/keras_file.py", &context);
  auto loc = FileLineColLoc::get(&context, id, 0, 0);
  auto id2 =
      Identifier::get("//tensorflow/python/something/my_test.py", &context);
  auto loc2 = FileLineColLoc::get(&context, id2, 0, 0);
  auto id3 = Identifier::get("python/tensorflow/show_file.py", &context);
  auto loc3 = FileLineColLoc::get(&context, id3, 0, 0);

  // These locations will be evalauted as internal frames, passing the
  // IsInternalFramesForFilenames() check so will be filtered out.
  auto id_filtered =
      Identifier::get("//tensorflow/python/dir/filtered_file_A.py", &context);
  auto loc_filtered = FileLineColLoc::get(&context, id_filtered, 0, 0);
  auto id_filtered2 =
      Identifier::get("dir/tensorflow/python/filtered_file_B.py", &context);
  auto loc_filtered2 = FileLineColLoc::get(&context, id_filtered2, 0, 0);

  // Build a small stack for each error; the MLIR diagnostic filtering will
  // surface a location that would otherwise be filtered if it is the only
  // location associated with an error; therefore we need a combinatination of
  // locations to test.
  auto callsite_loc = mlir::CallSiteLoc::get(loc, loc_filtered);
  auto callsite_loc2 = mlir::CallSiteLoc::get(loc2, loc_filtered2);
  auto callsite_loc3 = mlir::CallSiteLoc::get(loc_filtered2, loc3);

  // Test with filter on.
  StatusScopedDiagnosticHandler ssdh_filter(&context, false, true);
  emitError(callsite_loc) << "Error 1";
  emitError(callsite_loc2) << "Error 2";
  emitError(callsite_loc3) << "Error 3";
  Status s_filtered = ssdh_filter.ConsumeStatus();
  // Check for the files that should not be filtered.
  EXPECT_THAT(s_filtered.error_message(), HasSubstr("keras"));
  EXPECT_THAT(s_filtered.error_message(), HasSubstr("test.py"));
  EXPECT_THAT(s_filtered.error_message(), HasSubstr("show_file"));
  // Verify the filtered files are not present.
  EXPECT_THAT(s_filtered.error_message(), Not(HasSubstr("filtered_file")));
}

TEST(ErrorUtilTest, StatusScopedDiagnosticHandlerWithoutFilter) {
  // Filtering logic should be off so all files should 'pass'.
  MLIRContext context;
  // This file would pass the filter if it was on.
  auto id =
      Identifier::get("//tensorflow/python/keras/keras_file.py", &context);
  auto loc = FileLineColLoc::get(&context, id, 0, 0);

  // The '_filtered' locations would be evaluated as internal frames, so would
  // not pass the filter if it was on.
  auto id_filtered =
      Identifier::get("//tensorflow/python/dir/filtered_file_A.py", &context);
  auto loc_filtered = FileLineColLoc::get(&context, id_filtered, 0, 0);
  auto id_filtered2 =
      Identifier::get("dir/tensorflow/python/filtered_file_B.py", &context);
  auto loc_filtered2 = FileLineColLoc::get(&context, id_filtered2, 0, 0);
  auto id_filtered3 =
      Identifier::get("//tensorflow/python/something/my_op.py", &context);
  auto loc_filtered3 = FileLineColLoc::get(&context, id_filtered3, 0, 0);

  // Build a small stack for each error; the MLIR diagnostic filtering will
  // surface a location that would otherwise be filtered if it is the only
  // location associated with an error; therefore we need a combinatination of
  // locations to test.
  auto callsite_loc = mlir::CallSiteLoc::get(loc, loc_filtered);
  auto callsite_loc2 = mlir::CallSiteLoc::get(loc_filtered3, loc_filtered2);

  // Test with filter off.
  StatusScopedDiagnosticHandler ssdh_no_filter(&context, false, false);
  emitError(callsite_loc) << "Error 1";
  emitError(callsite_loc2) << "Error 2";
  Status s_no_filter = ssdh_no_filter.ConsumeStatus();
  // All files should be present, especially the 'filtered' ones.
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("keras"));
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("my_op"));
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("filtered_file_A"));
  EXPECT_THAT(s_no_filter.error_message(), HasSubstr("filtered_file_B"));
}

}  // namespace
}  // namespace mlir
