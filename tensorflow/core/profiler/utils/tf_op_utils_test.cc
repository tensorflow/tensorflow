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

TEST(TfOpUtilsTest, JaxOpNameTest) {
  const absl::string_view kOpName = "namescope/add";
  const absl::string_view kOpType = "add";
  EXPECT_TRUE(IsJaxOpNameAndType(kOpName, kOpType));
}

TEST(TfOpUtilsTest, JaxOpNameWithMetadataTest) {
  const absl::string_view kOpName =
      "pmap(<unnamed wrapped function>)/gather[ "
      "dimension_numbers=GatherDimensionNumbers(offset_dims=(2,), "
      "collapsed_slice_dims=(0, 1), start_index_map=(0, 1))\n                  "
      "                       slice_sizes=(1, 1, 81) ]:gather";
  const absl::string_view kOpType = "gather";
  EXPECT_TRUE(IsJaxOpNameAndType(kOpName, kOpType));
}

TEST(TfOpUtilsTest, OtherXlaOpTest) {
  const absl::string_view kName =
      "namescope.1/namespace__opname2d:namespace__opname2d";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kJax);
  EXPECT_EQ(tf_op.name, "namescope.1/namespace__opname2d");
  EXPECT_EQ(tf_op.type, "namespace__opname2d");
  EXPECT_EQ(TfOpEventName(kName), "namespace__opname2d");
}

TEST(TfOpUtilsTest, OtherXlaOpNameTest) {
  const absl::string_view kOpName = "namescope.1/namespace__opname2d";
  const absl::string_view kOpType = "namespace__opname2d";
  EXPECT_TRUE(IsJaxOpNameAndType(kOpName, kOpType));
}

TEST(TfOpUtilsTest, OpWithoutTypeTest) {
  const absl::string_view kName = "namescope/OpName_1:";  // with trailing ':'
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTensorFlow);
  EXPECT_EQ(tf_op.name, "namescope/OpName_1");
  EXPECT_EQ(tf_op.type, "OpName");
  EXPECT_EQ(TfOpEventName(kName),
            "OpName");  // without trailing ':', name scopes and suffix
}

TEST(TfOpUtilsTest, OpTypeWithUnderscoreTest) {
  const absl::string_view kName = "namescope/OpName_a:";  // with trailing ':'
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTensorFlow);
  EXPECT_EQ(tf_op.name, "namescope/OpName_a");
  EXPECT_EQ(tf_op.type, "OpName_a");
  EXPECT_EQ(TfOpEventName(kName),
            "OpName_a");  // without trailing ':', name scopes
}

TEST(TfOpUtilsTest, NameScopeTest) {
  const absl::string_view kName = "scope-1/scope2/OpName:OpType";
  TfOp tf_op = ParseTfOpFullname(kName);
  EXPECT_EQ(tf_op.category, Category::kTensorFlow);
  EXPECT_EQ(tf_op.name, "scope-1/scope2/OpName");
  EXPECT_EQ(tf_op.type, "OpType");
  std::vector<absl::string_view> name_scopes = ParseTfNameScopes(tf_op);
  EXPECT_EQ(name_scopes.size(), 2);
  EXPECT_EQ(name_scopes[0], "scope-1");
  EXPECT_EQ(name_scopes[1], "scope2");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
