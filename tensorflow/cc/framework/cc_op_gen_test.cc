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

#include <memory>
#include <string>

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

constexpr char kHiddenApiDef[] = R"(
op {
  graph_op_name: "Foo"
  visibility: HIDDEN
}
)";

constexpr char kReorderedArgsApiDef[] = R"(
op {
  graph_op_name: "Foo"
  arg_order: "dim"
  arg_order: "images"
}
)";

constexpr char kEndpointsApiDef[] = R"(
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

struct GeneratedCcOpTexts {
  std::string header;
  std::string source;
  std::string internal_header;
  std::string internal_source;
};

void ExpectHasSubstr(absl::string_view text, absl::string_view expected) {
  EXPECT_TRUE(absl::StrContains(text, expected))
      << "'" << text << "' does not contain '" << expected << "'";
}

void ExpectDoesNotHaveSubstr(absl::string_view text, absl::string_view expected) {
  EXPECT_FALSE(absl::StrContains(text, expected))
      << "'" << text << "' contains '" << expected << "'";
}

void ExpectSubstrOrder(absl::string_view text, absl::string_view before,
                       absl::string_view after) {
  const size_t before_pos = text.find(before);
  const size_t after_pos = text.find(after);

  ASSERT_NE(std::string::npos, before_pos)
      << "Missing substring: " << before;
  ASSERT_NE(std::string::npos, after_pos)
      << "Missing substring: " << after;

  EXPECT_LT(before_pos, after_pos)
      << "'" << before << "' is not before '" << after << "' in:\n"
      << text;
}

class CcOpGenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs_));
    api_def_map_ = std::make_unique<ApiDefMap>(op_defs_);
  }

  GeneratedCcOpTexts GenerateCcOpFiles() const {
    const std::string tmpdir = testing::TmpDir();

    const std::string h_file_path = io::JoinPath(tmpdir, "test.h");
    const std::string cc_file_path = io::JoinPath(tmpdir, "test.cc");
    const std::string internal_h_file_path =
        io::JoinPath(tmpdir, "test_internal.h");
    const std::string internal_cc_file_path =
        io::JoinPath(tmpdir, "test_internal.cc");

    cc_op::WriteCCOps(op_defs_, *api_def_map_, h_file_path, cc_file_path);

    GeneratedCcOpTexts result;
    TF_ASSERT_OK(ReadFileToString(env_, h_file_path, &result.header));
    TF_ASSERT_OK(ReadFileToString(env_, cc_file_path, &result.source));
    TF_ASSERT_OK(
        ReadFileToString(env_, internal_h_file_path, &result.internal_header));
    TF_ASSERT_OK(ReadFileToString(env_, internal_cc_file_path,
                                  &result.internal_source));
    return result;
  }

  Env* const env_ = Env::Default();
  OpList op_defs_;
  std::unique_ptr<ApiDefMap> api_def_map_;
};

TEST_F(CcOpGenTest, TestVisibilityChangedToHidden) {
  const auto without_api_def = GenerateCcOpFiles();

  ExpectHasSubstr(without_api_def.header, "class Foo");
  ExpectDoesNotHaveSubstr(without_api_def.internal_header, "class Foo");

  ASSERT_TRUE(api_def_map_->LoadApiDef(kHiddenApiDef));
  const auto with_api_def = GenerateCcOpFiles();

  ExpectHasSubstr(with_api_def.internal_header, "class Foo");
  ExpectDoesNotHaveSubstr(with_api_def.header, "class Foo");
}

TEST_F(CcOpGenTest, TestArgNameChanges) {
  const auto without_api_def = GenerateCcOpFiles();
  ExpectSubstrOrder(without_api_def.header, "Input images", "Input dim");

  ASSERT_TRUE(api_def_map_->LoadApiDef(kReorderedArgsApiDef));
  const auto with_api_def = GenerateCcOpFiles();
  ExpectSubstrOrder(with_api_def.header, "Input dim", "Input images");
}

TEST_F(CcOpGenTest, TestEndpoints) {
  const auto without_api_def = GenerateCcOpFiles();

  ExpectHasSubstr(without_api_def.header, "class Foo {");
  ExpectDoesNotHaveSubstr(without_api_def.header, "class Foo1");
  ExpectDoesNotHaveSubstr(without_api_def.header, "class Foo2");

  ASSERT_TRUE(api_def_map_->LoadApiDef(kEndpointsApiDef));
  const auto with_api_def = GenerateCcOpFiles();

  ExpectHasSubstr(with_api_def.header, "class Foo1");
  ExpectHasSubstr(with_api_def.header, "typedef Foo1 Foo2");
  ExpectDoesNotHaveSubstr(with_api_def.header, "class Foo {");
}

}  // namespace
}  // namespace tensorflow