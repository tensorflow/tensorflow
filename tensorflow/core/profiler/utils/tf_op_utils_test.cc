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

#include "tensorflow/core/profiler/utils/tf_op_utils.h"

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(TfOpUtilsTest, TfOpTest) {
  const absl::string_view kName = "OpName:OpType";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTensorFlow);
  EXPECT_EQ(tf_op.name, "OpName");
  EXPECT_EQ(tf_op.type, "OpType");
  EXPECT_EQ(TfOpEventName(kName), "OpType");  // type only
}

TEST(TfOpUtilsTest, InternalTfOpTest) {
  const absl::string_view kName = "OpName:_InternalOpType";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTensorFlow);
  EXPECT_EQ(tf_op.name, "OpName");
  EXPECT_EQ(tf_op.type, "_InternalOpType");
  EXPECT_EQ(TfOpEventName(kName), "_InternalOpType");  // type only
}

TEST(TfOpUtilsTest, TfOpWithPathTest) {
  const absl::string_view kName = "path/to/name:OpType";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTensorFlow);
  EXPECT_EQ(tf_op.name, "path/to/name");
  EXPECT_EQ(tf_op.type, "OpType");
  EXPECT_EQ(TfOpEventName(kName), "OpType");  // type only
}

TEST(TfOpUtilsTest, ShortDatasetOpTest) {
  const absl::string_view kName = "Iterator::Batch";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTfData);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kDatasetOp);
  EXPECT_EQ(TfOpEventName(kName), kName);
}

TEST(TfOpUtilsTest, LongDatasetOpTest) {
  const absl::string_view kName = "Iterator::Batch::Map::TfRecord";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTfData);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kDatasetOp);
  EXPECT_EQ(TfOpEventName(kName), "Iterator::TfRecord");  // shorter name
}

TEST(TfOpUtilsTest, TraceMeTest) {
  const absl::string_view kName = "MyTraceMe";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kUnknown);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kUnknownOp);
  EXPECT_EQ(TfOpEventName(kName), kName);
}

TEST(TfOpUtilsTest, TraceMeWithColonTest) {
  // "12345" is not a valid op type.
  const absl::string_view kName = "RunStep/Server:54635";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kUnknown);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kUnknownOp);
  EXPECT_EQ(TfOpEventName(kName), kName);
}

TEST(TfOpUtilsTest, TraceMeWithDoubleColonTest) {
  const absl::string_view kName = "XLA::StartProgram";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kUnknown);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kUnknownOp);
  EXPECT_EQ(TfOpEventName(kName), kName);
}

TEST(TfOpUtilsTest, TraceMeWithTrailingWhitespaceTest) {
  const absl::string_view kName = "SessionRun ";
  const absl::string_view kNameTrimmed = "SessionRun";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kUnknown);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kUnknownOp);
  EXPECT_EQ(TfOpEventName(kName), kNameTrimmed);
}

TEST(TfOpUtilsTest, MemcpyHToDTest) {
  const absl::string_view kName = "MemcpyHToD";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kMemcpyHToD);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kMemcpyHToDOp);
  EXPECT_EQ(TfOpEventName(kName), kName);
}

TEST(TfOpUtilsTest, MemcpyDToHTest) {
  const absl::string_view kName = "MemcpyDToH";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kMemcpyDToH);
  EXPECT_EQ(tf_op.name, kName);
  EXPECT_EQ(tf_op.type, kMemcpyDToHOp);
  EXPECT_EQ(TfOpEventName(kName), kName);
}

TEST(TfOpUtilsTest, JaxOpTest) {
  const absl::string_view kName = "op_name:op_type";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kJax);
  EXPECT_EQ(tf_op.name, "op_name");
  EXPECT_EQ(tf_op.type, "op_type");
  EXPECT_EQ(TfOpEventName(kName), "op_type");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
