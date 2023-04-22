/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_graph_utils.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class DebugGraphUtilsTest : public ::testing::Test {
 protected:
  Status ParseDebugOpName(const string& debug_op_name,
                          string* debug_op_name_proper,
                          std::unordered_map<string, string>* attributes) {
    return DebugNodeInserter::ParseDebugOpName(
        debug_op_name, debug_op_name_proper, attributes);
  }
};

TEST_F(DebugGraphUtilsTest, TestParseNoAttributeDebugOpName) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;
  TF_ASSERT_OK(
      ParseDebugOpName("DebugIdentity", &debug_op_name_proper, &attributes));
  ASSERT_EQ("DebugIdentity", debug_op_name_proper);
  ASSERT_EQ(0, attributes.size());
}

TEST_F(DebugGraphUtilsTest, TestMalformedDebugOpName) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;

  Status s = ParseDebugOpName("(mute_if_healthy=true)", &debug_op_name_proper,
                              &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName("DebugNumericSummary(", &debug_op_name_proper,
                       &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName("DebugNumericSummary)", &debug_op_name_proper,
                       &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());
}

TEST_F(DebugGraphUtilsTest, TestDebugOpNameWithMalformedAttributes) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;

  Status s = ParseDebugOpName("DebugNumericSummary(=)", &debug_op_name_proper,
                              &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName("DebugNumericSummary(mute_if_healthy=)",
                       &debug_op_name_proper, &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName("DebugNumericSummary(=true)", &debug_op_name_proper,
                       &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName("DebugNumericSummary(mute_if_healthy:true)",
                       &debug_op_name_proper, &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName("DebugNumericSummary(mute_if_healthy=true;threshold=)",
                       &debug_op_name_proper, &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());

  s = ParseDebugOpName(
      "DebugNumericSummary(mute_if_healthy=true;threshold:300.0)",
      &debug_op_name_proper, &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());
}

TEST_F(DebugGraphUtilsTest, TestValidDebugOpNameWithSingleAttribute) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;

  TF_ASSERT_OK(ParseDebugOpName("DebugNumericSummary()", &debug_op_name_proper,
                                &attributes));
  ASSERT_EQ("DebugNumericSummary", debug_op_name_proper);
  ASSERT_EQ(0, attributes.size());

  attributes.clear();
  TF_ASSERT_OK(ParseDebugOpName("DebugNumericSummary(mute_if_healthy=true)",
                                &debug_op_name_proper, &attributes));
  ASSERT_EQ("DebugNumericSummary", debug_op_name_proper);
  ASSERT_EQ(1, attributes.size());
  ASSERT_EQ("true", attributes["mute_if_healthy"]);
}

TEST_F(DebugGraphUtilsTest, TestValidDebugOpNameWithMoreThanOneAttributes) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;
  TF_ASSERT_OK(ParseDebugOpName(
      "DebugNumericSummary(mute_if_healthy=true; threshold=300.0)",
      &debug_op_name_proper, &attributes));
  ASSERT_EQ("DebugNumericSummary", debug_op_name_proper);
  ASSERT_EQ(2, attributes.size());
  ASSERT_EQ("true", attributes["mute_if_healthy"]);
  ASSERT_EQ("300.0", attributes["threshold"]);

  attributes.clear();
  TF_ASSERT_OK(ParseDebugOpName(
      "DebugNumericSummary(mute_if_healthy=true;threshold=300.0;first_n=100)",
      &debug_op_name_proper, &attributes));
  ASSERT_EQ("DebugNumericSummary", debug_op_name_proper);
  ASSERT_EQ(3, attributes.size());
  ASSERT_EQ("true", attributes["mute_if_healthy"]);
  ASSERT_EQ("300.0", attributes["threshold"]);
  ASSERT_EQ("100", attributes["first_n"]);
}

TEST_F(DebugGraphUtilsTest, TestValidDebugOpNameWithMoreDuplicateAttributes) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;
  Status s = ParseDebugOpName(
      "DebugNumericSummary(mute_if_healthy=true; lower_bound=3; "
      "mute_if_healthy=false;)",
      &debug_op_name_proper, &attributes);
  ASSERT_EQ(errors::Code::INVALID_ARGUMENT, s.code());
}

TEST_F(DebugGraphUtilsTest, TestValidDebugOpNameWithWhitespaceInAttributes) {
  string debug_op_name_proper;
  std::unordered_map<string, string> attributes;

  TF_ASSERT_OK(ParseDebugOpName(
      "DebugNumericSummary(  mute_if_healthy=true; threshold=300.0  )",
      &debug_op_name_proper, &attributes));
  ASSERT_EQ("DebugNumericSummary", debug_op_name_proper);
  ASSERT_EQ(2, attributes.size());
  ASSERT_EQ("true", attributes["mute_if_healthy"]);
  ASSERT_EQ("300.0", attributes["threshold"]);

  attributes.clear();
  TF_ASSERT_OK(ParseDebugOpName(
      "DebugNumericSummary(;;mute_if_healthy=true; threshold=300.0;;)",
      &debug_op_name_proper, &attributes));
  ASSERT_EQ("DebugNumericSummary", debug_op_name_proper);
  ASSERT_EQ(2, attributes.size());
  ASSERT_EQ("true", attributes["mute_if_healthy"]);
  ASSERT_EQ("300.0", attributes["threshold"]);
}

}  // namespace tensorflow
