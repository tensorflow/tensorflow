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

#include "tensorflow/compiler/mlir/lite/python/conversion_failure_reporter.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/python/pass_debug_instrumentation.h"
#include "xla/tsl/platform/env.h"

namespace mlir::TFL {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST(ConversionFailureReporterTest, GetOrCreateDebugDirCustom) {
  std::string dir =
      ConversionFailureReporter::GetOrCreateDebugDir("/my/custom/dir");
  EXPECT_EQ(dir, "/my/custom/dir");
}

TEST(ConversionFailureReporterTest, GetOrCreateDebugDirDefaultDynamic) {
  std::string dir = ConversionFailureReporter::GetOrCreateDebugDir("");
  EXPECT_THAT(dir, HasSubstr("/tmp/litert_conv_"));
}

TEST(ConversionFailureReporterTest, ParseDiagnosticFull) {
  const std::string raw_error = R"(
third_party/models/test.py:42:1: error: custom error message
  %val = "tf.UnsupportedOp"(%0)
  ^
third_party/models/caller.py:10:5: note: called from
  return helper_function(x)
third_party/models/graph.mlir:15:3: note: see current operation: "tfl.custom"(%arg0) {
  attr1 = 42 : i32,
  attr2 = "value"
}
)";

  FailureReport report = ConversionFailureReporter::ParseDiagnostic(
      raw_error, "PassPipeline", "INVALID_ARGUMENT", "TestPass",
      "--test-pass-flag", "main_func");

  EXPECT_EQ(report.stage, "PassPipeline");
  EXPECT_EQ(report.status_code, "INVALID_ARGUMENT");
  EXPECT_EQ(report.failing_pass, "TestPass");
  EXPECT_EQ(report.failing_pass_arg, "--test-pass-flag");
  EXPECT_EQ(report.failing_function, "main_func");

  // Primary error
  EXPECT_EQ(report.primary_error.location, "third_party/models/test.py:42:1");
  EXPECT_EQ(report.primary_error.severity, "error");
  EXPECT_EQ(report.primary_error.message, "custom error message");
  EXPECT_EQ(report.primary_error.code_snippet,
            "%val = \"tf.UnsupportedOp\"(%0)");

  // Call stack
  ASSERT_THAT(report.call_stack, SizeIs(1));
  EXPECT_EQ(report.call_stack[0].location, "third_party/models/caller.py:10:5");
  EXPECT_EQ(report.call_stack[0].code_snippet, "return helper_function(x)");

  // Operation context with multi-line continuation
  EXPECT_EQ(report.operation_context.location,
            "third_party/models/graph.mlir:15:3");
  EXPECT_THAT(report.operation_context.operation,
              HasSubstr("\"tfl.custom\"(%arg0) {"));
  EXPECT_THAT(report.operation_context.operation,
              HasSubstr("attr1 = 42 : i32,"));
  EXPECT_THAT(report.operation_context.operation,
              HasSubstr("attr2 = \"value\""));
}

TEST(ConversionFailureReporterTest, WriteFailureJsonCreatesArtifacts) {
  std::string test_dir;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&test_dir));
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(test_dir).ok());

  MLIRContext context;
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>("module {}", &context);
  ASSERT_TRUE(module);

  ConversionFailureReporter::WriteFailureJson(
      test_dir, *module, "loc:1:1: error: failed pass", "TestStage",
      "INVALID_ARGUMENT", "FailingPass", "--pass-arg",
      /*write_module_artifacts=*/true, "main");

  std::string failure_json_path = absl::StrCat(test_dir, "/failure.json");
  std::string elided_mlir_path =
      absl::StrCat(test_dir, "/before_failure_elided.mlir");
  std::string bytecode_path = absl::StrCat(test_dir, "/module_bytecode.mlirbc");

  EXPECT_TRUE(tsl::Env::Default()->FileExists(failure_json_path).ok());
  EXPECT_TRUE(tsl::Env::Default()->FileExists(elided_mlir_path).ok());
  EXPECT_TRUE(tsl::Env::Default()->FileExists(bytecode_path).ok());

  std::string json_contents;
  ASSERT_TRUE(tsl::ReadFileToString(tsl::Env::Default(), failure_json_path,
                                    &json_contents)
                  .ok());
  EXPECT_THAT(json_contents, HasSubstr("\"stage\": \"TestStage\""));
  EXPECT_THAT(json_contents, HasSubstr("\"failing_pass\": \"FailingPass\""));
  EXPECT_THAT(json_contents, HasSubstr("\"failing_function\": \"main\""));
}

TEST(PipelineFailureCoordinatorTest, ReportSerializationFailure) {
  std::string test_dir;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&test_dir));
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(test_dir).ok());

  MLIRContext context;
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>("module {}", &context);
  ASSERT_TRUE(module);

  PipelineFailureCoordinator coordinator(test_dir, /*enable_debug=*/true);
  absl::Status status = coordinator.ReportSerializationFailure(
      *module, absl::InternalError("internal serialization error"),
      "diag details");

  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(status.message(),
              HasSubstr("Failed to serialize to FlatBuffer: diag details"));

  std::string failure_json_path = absl::StrCat(test_dir, "/failure.json");
  EXPECT_TRUE(tsl::Env::Default()->FileExists(failure_json_path).ok());
}

}  // namespace
}  // namespace mlir::TFL
