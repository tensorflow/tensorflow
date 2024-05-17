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

#include "tensorflow/cc/framework/cc_op_gen.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kBaseOpDef[] = R"(
op {
  name: "Foo"
  input_arg {
    name: "images"
    description: "Images to process."
  }
  input_arg {
    name: "dim"
    description: "Description for dim."
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    description: "Description for output."
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    description: "Type for images"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
      }
    }
    default_value {
      i: 1
    }
  }
  summary: "Summary for op Foo."
  description: "Description for op Foo."
}
)";

void ExpectHasSubstr(StringPiece s, StringPiece expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

void ExpectDoesNotHaveSubstr(StringPiece s, StringPiece expected) {
  EXPECT_FALSE(absl::StrContains(s, expected))
      << "'" << s << "' contains '" << expected << "'";
}

void ExpectSubstrOrder(const string& s, const string& before,
                       const string& after) {
  int before_pos = s.find(before);
  int after_pos = s.find(after);
  ASSERT_NE(std::string::npos, before_pos);
  ASSERT_NE(std::string::npos, after_pos);
  EXPECT_LT(before_pos, after_pos)
      << before << " is not before " << after << " in " << s;
}

// Runs WriteCCOps and stores output in (internal_)cc_file_path and
// (internal_)h_file_path.
void GenerateCcOpFiles(Env* env, const OpList& ops,
                       const ApiDefMap& api_def_map, string* h_file_text,
                       string* internal_h_file_text) {
  const string& tmpdir = testing::TmpDir();

  const auto h_file_path = io::JoinPath(tmpdir, "test.h");
  const auto cc_file_path = io::JoinPath(tmpdir, "test.cc");
  const auto internal_h_file_path = io::JoinPath(tmpdir, "test_internal.h");
  const auto internal_cc_file_path = io::JoinPath(tmpdir, "test_internal.cc");

  cc_op::WriteCCOps(ops, api_def_map, h_file_path, cc_file_path);

  TF_ASSERT_OK(ReadFileToString(env, h_file_path, h_file_text));
  TF_ASSERT_OK(
      ReadFileToString(env, internal_h_file_path, internal_h_file_text));
}

TEST(CcOpGenTest, TestVisibilityChangedToHidden) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  visibility: HIDDEN
}
)";
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string h_file_text, internal_h_file_text;
  // Without ApiDef
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(h_file_text, "class Foo");
  ExpectDoesNotHaveSubstr(internal_h_file_text, "class Foo");

  // With ApiDef
  TF_ASSERT_OK(api_def_map.LoadApiDef(api_def));
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(internal_h_file_text, "class Foo");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo");
}

TEST(CcOpGenTest, TestArgNameChanges) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  arg_order: "dim"
  arg_order: "images"
}
)";
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT

  ApiDefMap api_def_map(op_defs);
  string cc_file_text, h_file_text;
  string internal_cc_file_text, internal_h_file_text;
  // Without ApiDef
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectSubstrOrder(h_file_text, "Input images", "Input dim");

  // With ApiDef
  TF_ASSERT_OK(api_def_map.LoadApiDef(api_def));
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectSubstrOrder(h_file_text, "Input dim", "Input images");
}

TEST(CcOpGenTest, TestEndpoints) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  endpoint {
    name: "Foo1"
  }
  endpoint {
    name: "Foo2"
  }
}
)";
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT

  ApiDefMap api_def_map(op_defs);
  string cc_file_text, h_file_text;
  string internal_cc_file_text, internal_h_file_text;
  // Without ApiDef
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(h_file_text, "class Foo {");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo1");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo2");

  // With ApiDef
  TF_ASSERT_OK(api_def_map.LoadApiDef(api_def));
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(h_file_text, "class Foo1");
  ExpectHasSubstr(h_file_text, "typedef Foo1 Foo2");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo {");
}
}  // namespace
}  // namespace tensorflow
