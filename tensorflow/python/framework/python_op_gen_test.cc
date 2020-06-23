/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/framework/python_op_gen.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace {

constexpr char kBaseOpDef[] = R"(
op {
  name: "Foo"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T2"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
      }
    }
  }
  attr {
    name: "T2"
    type: "type"
    allowed_values {
      list {
        type: DT_STRING
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  summary: "Summary for op Foo."
  description: "Description for op Foo."
}
op {
  name: "Bar"
  input_arg {
    name: "x"
    type: DT_STRING
  }
  input_arg {
    name: "y"
    type: DT_QINT8
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
  summary: "Summary for op Bar."
  description: "Description for op Bar."
}
op {
  name: "FooBar"
  input_arg {
    name: "x"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
  attr {
    name: "t"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_INT8
      }
    }
  }
  attr {
    name: "var1"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "var2"
    type: "int"
    default_value {
      i: 0
    }
  }
  summary: "Summary for op FooBar."
  description: "Description for op FooBar."
}
op {
  name: "Baz"
  input_arg {
    name: "inputs"
    number_attr: "N"
    type_list_attr: "T"
  }
  output_arg {
    name: "output1"
    type: DT_BOOL
  }
  output_arg {
    name: "output2"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "bool"
  }
  attr {
    name: "N"
    type: "int"
  }
  summary: "Summary for op Baz."
  description: "Description for op Baz."
}
)";

std::unordered_set<string> type_annotate_ops {
  "Foo",
  "Bar",
  "FooBar",
  "Baz"
};


void ExpectHasSubstr(const string& s, const string& expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'Generated ops does not contain '" << expected << "'";
}

void ExpectDoesNotHaveSubstr(const string& s, const string& expected) {
  EXPECT_FALSE(absl::StrContains(s, expected))
      << "'Generated ops contains '" << expected << "'";
}

void ExpectSubstrOrder(const string& s, const string& before,
                       const string& after) {
  int before_pos = s.find(before);
  int after_pos = s.find(after);
  ASSERT_NE(std::string::npos, before_pos);
  ASSERT_NE(std::string::npos, after_pos);
  EXPECT_LT(before_pos, after_pos)
      << before << "' is not before '" << after;
}

TEST(PythonOpGen, Basic) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);

  string code = GetPythonOps(ops, api_def_map, {}, "", {});

  EXPECT_TRUE(absl::StrContains(code, "def case"));
  // TODO(mdan): Add tests to verify type annotations are correctly added.
}

TEST(PythonOpGen, TypeAnnotateSingleTypeTensor) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typed_bar = "def bar(x: _ops.Tensor[_dtypes.String], y: _ops.Tensor[_dtypes.QInt8], name=None) -> _ops.Tensor[_dtypes.Bool]:";
  ExpectHasSubstr(code, typed_bar);

  const string untyped_bar = "def bar(x, y, name=None):";
  ExpectDoesNotHaveSubstr(code, untyped_bar);
}

TEST(PythonOpGen, TypeAnnotateMultiTypeTensor) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typed_foo = "def foo(x: _ops.Tensor[TV_Foo_T], y: _ops.Tensor[TV_Foo_T2], name=None) -> _ops.Tensor[TV_Foo_T]:";
  ExpectHasSubstr(code, typed_foo);
}

TEST(PythonOpGen, GenerateCorrectTypeVars) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typevars_foo = R"(
TV_Foo_T = TypeVar("TV_Foo_T", _dtypes.Int8, _dtypes.UInt8)
TV_Foo_T2 = TypeVar("TV_Foo_T2", _dtypes.Float32, _dtypes.Float64, _dtypes.String)
)";

  ExpectHasSubstr(code, typevars_foo);
}

TEST(PythonOpGen, TypeAnnotateFallback) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typed_foo_fallback = "def foo_eager_fallback(x: _ops.Tensor[TV_Foo_T], y: _ops.Tensor[TV_Foo_T2], name, ctx) -> _ops.Tensor[TV_Foo_T]:";
  ExpectHasSubstr(code, typed_foo_fallback);
}

TEST(PythonOpGen, GenerateTypeVarAboveOp) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typevar_foo = "TV_Foo_";
  const string def_foo = "def foo";
  ExpectSubstrOrder(code, typevar_foo, def_foo);
}


TEST(PythonOpGen, TypeAnnotateDefaultParams) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string params = "def foo_bar(x: _ops.Tensor[_dtypes.Float32], t: TV_FooBar_t, var1: bool = False, var2: int = 0, name=None)";
  const string params_fallback = "def foo_bar_eager_fallback(x: _ops.Tensor[_dtypes.Float32], t: TV_FooBar_t, var1: bool, var2: int, name, ctx)";
  ExpectHasSubstr(code, params);
  ExpectHasSubstr(code, params_fallback);
}

TEST(PythonOpGen, NoTypingSequenceTensors) {
  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string baz_def_line = "def baz(inputs, name=None):";

  ExpectHasSubstr(code, baz_def_line);
}

// TODO(mdan): Include more tests with synhtetic ops and api defs.

}  // namespace
}  // namespace tensorflow
