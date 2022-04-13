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

#include "tensorflow/core/framework/op_gen_lib.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

constexpr char kTestOpList[] = R"(op {
  name: "testop"
  input_arg {
    name: "arg_a"
  }
  input_arg {
    name: "arg_b"
  }
  output_arg {
    name: "arg_c"
  }
  attr {
    name: "attr_a"
  }
  deprecation {
    version: 123
    explanation: "foo"
  }
})";

constexpr char kTestApiDef[] = R"(op {
  graph_op_name: "testop"
  visibility: VISIBLE
  endpoint {
    name: "testop1"
  }
  in_arg {
    name: "arg_a"
  }
  in_arg {
    name: "arg_b"
  }
  out_arg {
    name: "arg_c"
  }
  attr {
    name: "attr_a"
  }
  summary: "Mock op for testing."
  description: <<END
Description for the
testop.
END
  arg_order: "arg_a"
  arg_order: "arg_b"
}
)";

TEST(OpGenLibTest, MultilinePBTxt) {
  // Non-multiline pbtxt
  const string pbtxt = R"(foo: "abc"
foo: ""
foo: "\n\n"
foo: "abc\nEND"
  foo: "ghi\njkl\n"
bar: "quotes:\""
)";

  // Field "foo" converted to multiline but not "bar".
  const string ml_foo = R"(foo: <<END
abc
END
foo: <<END

END
foo: <<END



END
foo: <<END0
abc
END
END0
  foo: <<END
ghi
jkl

END
bar: "quotes:\""
)";

  // Both fields "foo" and "bar" converted to multiline.
  const string ml_foo_bar = R"(foo: <<END
abc
END
foo: <<END

END
foo: <<END



END
foo: <<END0
abc
END
END0
  foo: <<END
ghi
jkl

END
bar: <<END
quotes:"
END
)";

  // ToMultiline
  EXPECT_EQ(ml_foo, PBTxtToMultiline(pbtxt, {"foo"}));
  EXPECT_EQ(pbtxt, PBTxtToMultiline(pbtxt, {"baz"}));
  EXPECT_EQ(ml_foo_bar, PBTxtToMultiline(pbtxt, {"foo", "bar"}));

  // FromMultiline
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(pbtxt));
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(ml_foo));
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(ml_foo_bar));
}

TEST(OpGenLibTest, PBTxtToMultilineErrorCases) {
  // Everything correct.
  EXPECT_EQ("f: <<END\n7\nEND\n", PBTxtToMultiline("f: \"7\"\n", {"f"}));

  // In general, if there is a problem parsing in PBTxtToMultiline, it leaves
  // the line alone.

  // No colon
  EXPECT_EQ("f \"7\"\n", PBTxtToMultiline("f \"7\"\n", {"f"}));
  // Only converts strings.
  EXPECT_EQ("f: 7\n", PBTxtToMultiline("f: 7\n", {"f"}));
  // No quote after colon.
  EXPECT_EQ("f: 7\"\n", PBTxtToMultiline("f: 7\"\n", {"f"}));
  // Only one quote
  EXPECT_EQ("f: \"7\n", PBTxtToMultiline("f: \"7\n", {"f"}));
  // Illegal escaping
  EXPECT_EQ("f: \"7\\\"\n", PBTxtToMultiline("f: \"7\\\"\n", {"f"}));
}

TEST(OpGenLibTest, PBTxtToMultilineComments) {
  const string pbtxt = R"(f: "bar"  # Comment 1
    f: "\n"  # Comment 2
)";
  const string ml = R"(f: <<END
bar
END  # Comment 1
    f: <<END


END  # Comment 2
)";

  EXPECT_EQ(ml, PBTxtToMultiline(pbtxt, {"f"}));
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(ml));
}

TEST(OpGenLibTest, ApiDefAccessInvalidName) {
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  ASSERT_EQ(nullptr, api_map.GetApiDef("testop5"));
}

TEST(OpGenLibTest, ApiDefInitializedFromOpDef) {
  tensorflow::ApiDef expected_api_def;
  protobuf::TextFormat::ParseFromString(
R"(graph_op_name: "testop"
visibility: VISIBLE
endpoint {
  name: "testop"
}
in_arg {
  name: "arg_a"
  rename_to: "arg_a"
}
in_arg {
  name: "arg_b"
  rename_to: "arg_b"
}
out_arg {
  name: "arg_c"
  rename_to: "arg_c"
}
attr {
  name: "attr_a"
  rename_to: "attr_a"
}
arg_order: "arg_a"
arg_order: "arg_b"
)",
      &expected_api_def);
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  const auto* api_def = api_map.GetApiDef("testop");
  ASSERT_EQ(api_def->DebugString(), expected_api_def.DebugString());
}

TEST(OpGenLibTest, ApiDefLoadSingleApiDef) {
  tensorflow::ApiDefs expected_api_defs;
  protobuf::TextFormat::ParseFromString(R"(op {
  graph_op_name: "testop"
  visibility: VISIBLE
  endpoint {
    name: "testop1"
  }
  in_arg {
    name: "arg_a"
    rename_to: "arg_a"
  }
  in_arg {
    name: "arg_b"
    rename_to: "arg_b"
  }
  out_arg {
    name: "arg_c"
    rename_to: "arg_c"
  }
  attr {
    name: "attr_a"
    rename_to: "attr_a"
  }
  summary: "Mock op for testing."
  description: "Description for the\ntestop."
  arg_order: "arg_a"
  arg_order: "arg_b"
}
)",
      &expected_api_defs);
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));
  const auto* api_def = api_map.GetApiDef("testop");
  EXPECT_EQ(1, api_def->endpoint_size());
  EXPECT_EQ("testop1", api_def->endpoint(0).name());

  ApiDefs api_defs;
  *api_defs.add_op() = *api_def;
  EXPECT_EQ(api_defs.DebugString(), expected_api_defs.DebugString());
}

TEST(OpGenLibTest, ApiDefOverrideVisibility) {
  const string api_def1 = R"(
op {
  graph_op_name: "testop"
  endpoint {
    name: "testop2"
  }
}
)";
  const string api_def2 = R"(
op {
  graph_op_name: "testop"
  visibility: HIDDEN
  endpoint {
    name: "testop2"
  }
}
)";
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));
  auto* api_def = api_map.GetApiDef("testop");
  EXPECT_EQ(ApiDef::VISIBLE, api_def->visibility());

  // Loading ApiDef with default visibility should
  // keep current visibility.
  TF_CHECK_OK(api_map.LoadApiDef(api_def1));
  EXPECT_EQ(ApiDef::VISIBLE, api_def->visibility());

  // Loading ApiDef with non-default visibility,
  // should update visibility.
  TF_CHECK_OK(api_map.LoadApiDef(api_def2));
  EXPECT_EQ(ApiDef::HIDDEN, api_def->visibility());
}

TEST(OpGenLibTest, ApiDefOverrideEndpoints) {
  const string api_def1 = R"(
op {
  graph_op_name: "testop"
  endpoint {
    name: "testop2"
  }
}
)";
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));
  auto* api_def = api_map.GetApiDef("testop");
  ASSERT_EQ(1, api_def->endpoint_size());
  EXPECT_EQ("testop1", api_def->endpoint(0).name());

  TF_CHECK_OK(api_map.LoadApiDef(api_def1));
  ASSERT_EQ(1, api_def->endpoint_size());
  EXPECT_EQ("testop2", api_def->endpoint(0).name());
}

TEST(OpGenLibTest, ApiDefOverrideArgs) {
  const string api_def1 = R"(
op {
  graph_op_name: "testop"
  in_arg {
    name: "arg_a"
    rename_to: "arg_aa"
  }
  out_arg {
    name: "arg_c"
    rename_to: "arg_cc"
  }
  arg_order: "arg_b"
  arg_order: "arg_a"
}
)";
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));
  TF_CHECK_OK(api_map.LoadApiDef(api_def1));
  const auto* api_def = api_map.GetApiDef("testop");
  ASSERT_EQ(2, api_def->in_arg_size());
  EXPECT_EQ("arg_aa", api_def->in_arg(0).rename_to());
  // 2nd in_arg is not renamed
  EXPECT_EQ("arg_b", api_def->in_arg(1).rename_to());

  ASSERT_EQ(1, api_def->out_arg_size());
  EXPECT_EQ("arg_cc", api_def->out_arg(0).rename_to());

  ASSERT_EQ(2, api_def->arg_order_size());
  EXPECT_EQ("arg_b", api_def->arg_order(0));
  EXPECT_EQ("arg_a", api_def->arg_order(1));
}

TEST(OpGenLibTest, ApiDefOverrideDescriptions) {
  const string api_def1 = R"(
op {
  graph_op_name: "testop"
  summary: "New summary"
  description: <<END
New description
END
  description_prefix: "A"
  description_suffix: "Z"
}
)";

  const string api_def2 = R"(
op {
  graph_op_name: "testop"
  description_prefix: "B"
  description_suffix: "Y"
}
)";

  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));
  TF_CHECK_OK(api_map.LoadApiDef(api_def1));
  const auto* api_def = api_map.GetApiDef("testop");
  EXPECT_EQ("New summary", api_def->summary());
  EXPECT_EQ("A\nNew description\nZ", api_def->description());
  EXPECT_EQ("", api_def->description_prefix());
  EXPECT_EQ("", api_def->description_suffix());

  TF_CHECK_OK(api_map.LoadApiDef(api_def2));
  EXPECT_EQ("B\nA\nNew description\nZ\nY", api_def->description());
  EXPECT_EQ("", api_def->description_prefix());
  EXPECT_EQ("", api_def->description_suffix());
}

TEST(OpGenLibTest, ApiDefInvalidOpInOverride) {
  const string api_def1 = R"(
op {
  graph_op_name: "different_testop"
  endpoint {
    name: "testop2"
  }
}
)";
  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT

  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));
  TF_CHECK_OK(api_map.LoadApiDef(api_def1));
  ASSERT_EQ(nullptr, api_map.GetApiDef("different_testop"));
}

TEST(OpGenLibTest, ApiDefInvalidArgOrder) {
  const string api_def1 = R"(
op {
  graph_op_name: "testop"
  arg_order: "arg_a"
  arg_order: "unexpected_arg"
}
)";

  const string api_def2 = R"(
op {
  graph_op_name: "testop"
  arg_order: "arg_a"
}
)";

  const string api_def3 = R"(
op {
  graph_op_name: "testop"
  arg_order: "arg_a"
  arg_order: "arg_a"
}
)";

  OpList op_list;
  protobuf::TextFormat::ParseFromString(kTestOpList, &op_list);  // NOLINT
  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(kTestApiDef));

  // Loading with incorrect arg name in arg_order should fail.
  auto status = api_map.LoadApiDef(api_def1);
  ASSERT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());

  // Loading with incorrect number of args in arg_order should fail.
  status = api_map.LoadApiDef(api_def2);
  ASSERT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());

  // Loading with the same argument twice in arg_order should fail.
  status = api_map.LoadApiDef(api_def3);
  ASSERT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());
}

TEST(OpGenLibTest, ApiDefInvalidSyntax) {
  const string api_def = R"pb(
    op { bad_op_name: "testop" }
  )pb";

  OpList op_list;
  ApiDefMap api_map(op_list);
  // Loading with invalid syntax (e.g. unrecognized field name) should fail.
  auto status = api_map.LoadApiDef(api_def);
  ASSERT_EQ(tensorflow::error::INVALID_ARGUMENT, status.code());
}

TEST(OpGenLibTest, ApiDefUpdateDocs) {
  const string op_list1 = R"(op {
  name: "testop"
  input_arg {
    name: "arg_a"
    description: "`arg_a`, `arg_c`, `attr_a`, `testop`"
  }
  output_arg {
    name: "arg_c"
    description: "`arg_a`, `arg_c`, `attr_a`, `testop`"
  }
  attr {
    name: "attr_a"
    description: "`arg_a`, `arg_c`, `attr_a`, `testop`"
  }
  description: "`arg_a`, `arg_c`, `attr_a`, `testop`"
}
)";

  const string api_def1 = R"(
op {
  graph_op_name: "testop"
  endpoint {
    name: "testop2"
  }
  in_arg {
    name: "arg_a"
    rename_to: "arg_aa"
  }
  out_arg {
    name: "arg_c"
    rename_to: "arg_cc"
    description: "New description: `arg_a`, `arg_c`, `attr_a`, `testop`"
  }
  attr {
    name: "attr_a"
    rename_to: "attr_aa"
  }
}
)";
  OpList op_list;
  protobuf::TextFormat::ParseFromString(op_list1, &op_list);  // NOLINT
  ApiDefMap api_map(op_list);
  TF_CHECK_OK(api_map.LoadApiDef(api_def1));
  api_map.UpdateDocs();

  const string expected_description =
      "`arg_aa`, `arg_cc`, `attr_aa`, `testop2`";
  EXPECT_EQ(expected_description, api_map.GetApiDef("testop")->description());
  EXPECT_EQ(expected_description,
            api_map.GetApiDef("testop")->in_arg(0).description());
  EXPECT_EQ("New description: " + expected_description,
            api_map.GetApiDef("testop")->out_arg(0).description());
  EXPECT_EQ(expected_description,
            api_map.GetApiDef("testop")->attr(0).description());
}
}  // namespace
}  // namespace tensorflow
