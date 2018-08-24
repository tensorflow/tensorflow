/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/js/ops/ts_op_gen.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void ExpectContainsStr(StringPiece s, StringPiece expected) {
  EXPECT_TRUE(str_util::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

void ExpectDoesNotContainStr(StringPiece s, StringPiece expected) {
  EXPECT_FALSE(str_util::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

constexpr char kBaseOpDef[] = R"(
op {
  name: "Foo"
  input_arg {
    name: "images"
    type_attr: "T"
    number_attr: "N"
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
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  summary: "Summary for op Foo."
  description: "Description for op Foo."
}
)";

// Generate TypeScript code
void GenerateTsOpFileText(const string& op_def_str, const string& api_def_str,
                          string* ts_file_text) {
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(
      op_def_str.empty() ? kBaseOpDef : op_def_str, &op_defs);
  ApiDefMap api_def_map(op_defs);

  if (!api_def_str.empty()) {
    TF_ASSERT_OK(api_def_map.LoadApiDef(api_def_str));
  }

  const string& tmpdir = testing::TmpDir();
  const auto ts_file_path = io::JoinPath(tmpdir, "test.ts");

  WriteTSOps(op_defs, api_def_map, ts_file_path);
  TF_ASSERT_OK(ReadFileToString(env, ts_file_path, ts_file_text));
}

TEST(TsOpGenTest, TestImports) {
  string ts_file_text;
  GenerateTsOpFileText("", "", &ts_file_text);

  const string expected = R"(
import * as tfc from '@tensorflow/tfjs-core';
import {createTensorsTypeOpAttr, nodeBackend} from './op_utils';
)";
  ExpectContainsStr(ts_file_text, expected);
}

TEST(TsOpGenTest, InputSingleAndList) {
  const string api_def = R"(
op {
  name: "Foo"
  input_arg {
    name: "images"
    type_attr: "T"
    number_attr: "N"
  }
}
)";

  string ts_file_text;
  GenerateTsOpFileText("", api_def, &ts_file_text);

  const string expected = R"(
export function Foo(images: tfc.Tensor[], dim: tfc.Tensor): tfc.Tensor {
)";
  ExpectContainsStr(ts_file_text, expected);
}

TEST(TsOpGenTest, TestVisibility) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  visibility: HIDDEN
}
)";

  string ts_file_text;
  GenerateTsOpFileText("", api_def, &ts_file_text);

  const string expected = R"(
export function Foo(images: tfc.Tensor[], dim: tfc.Tensor): tfc.Tensor {
)";
  ExpectDoesNotContainStr(ts_file_text, expected);
}

TEST(TsOpGenTest, SkipDeprecated) {
  const string op_def = R"(
op {
  name: "DeprecatedFoo"
  input_arg {
    name: "input"
    type_attr: "T"
    description: "Description for input."
  }
  output_arg {
    name: "output"
    description: "Description for output."
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    description: "Type for input"
    allowed_values {
      list {
        type: DT_FLOAT 
      }
    }
  }
  deprecation {
    explanation: "Deprecated."
  }
}
)";

  string ts_file_text;
  GenerateTsOpFileText(op_def, "", &ts_file_text);

  ExpectDoesNotContainStr(ts_file_text, "DeprecatedFoo");
}

TEST(TsOpGenTest, MultiOutput) {
  const string op_def = R"(
op {
  name: "MultiOutputFoo"
  input_arg {
    name: "input"
    description: "Description for input."
    type_attr: "T"
  }
  output_arg {
    name: "output1"
    description: "Description for output 1."
    type: DT_FLOAT
  }
  output_arg {
    name: "output2"
    description: "Description for output 2."
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    description: "Type for input"
    allowed_values {
      list {
        type: DT_FLOAT 
      }
    }
  }
  summary: "Summary for op MultiOutputFoo."
  description: "Description for op MultiOutputFoo."
}
)";

  string ts_file_text;
  GenerateTsOpFileText(op_def, "", &ts_file_text);

  const string expected = R"(
export function MultiOutputFoo(input: tfc.Tensor): tfc.Tensor[] {
)";
  ExpectContainsStr(ts_file_text, expected);
}

TEST(TsOpGenTest, OpAttrs) {
  string ts_file_text;
  GenerateTsOpFileText("", "", &ts_file_text);

  const string expectedFooAttrs = R"(
  const opAttrs = [
    createTensorsTypeOpAttr('T', images),
    {name: 'N', type: nodeBackend().binding.TF_ATTR_INT, value: images.length}
  ];
)";

  ExpectContainsStr(ts_file_text, expectedFooAttrs);
}

}  // namespace
}  // namespace tensorflow
