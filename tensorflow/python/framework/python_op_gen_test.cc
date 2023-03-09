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

#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/python/framework/op_reg_offset.pb.h"

namespace tensorflow {
namespace {

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
  EXPECT_LT(before_pos, after_pos) << before << "' is not before '" << after;
}

TEST(PythonOpGen, TypeAnnotateAllOps) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);

  string code =
      GetPythonOps(ops, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string all_types =
      ", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, "
      "_atypes.Complex64, "
      "_atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Half, "
      "_atypes.Int16, "
      "_atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, "
      "_atypes.QInt32, "
      "_atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, "
      "_atypes.String, "
      "_atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8, "
      "_atypes.Variant)";

  const string fake_param_typevar =
      "TV_FakeParam_dtype = TypeVar(\"TV_FakeParam_dtype\"" + all_types;
  const string fake_param =
      "def fake_param_eager_fallback(dtype: TV_FakeParam_dtype, shape, name, "
      "ctx) -> _atypes.TensorFuzzingAnnotation[TV_FakeParam_dtype]:";
  const string fake_param_fallback =
      "def fake_param_eager_fallback(dtype: TV_FakeParam_dtype, shape, name, "
      "ctx) -> _atypes.TensorFuzzingAnnotation[TV_FakeParam_dtype]:";

  ExpectHasSubstr(code, fake_param_typevar);
  ExpectHasSubstr(code, fake_param);
  ExpectHasSubstr(code, fake_param_fallback);

  const string to_bool_typevar =
      "TV_ToBool_T = TypeVar(\"TV_ToBool_T\"" + all_types;
  const string to_bool_ =
      "def to_bool(input: _atypes.TensorFuzzingAnnotation[TV_ToBool_T], "
      "name=None) -> "
      "_atypes.TensorFuzzingAnnotation[_atypes.Bool]:";
  const string to_bool_fallback =
      "def to_bool_eager_fallback(input: "
      "_atypes.TensorFuzzingAnnotation[TV_ToBool_T], name, ctx) "
      "-> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:";

  ExpectHasSubstr(code, to_bool_typevar);
  ExpectHasSubstr(code, to_bool_);
  ExpectHasSubstr(code, to_bool_fallback);
}

TEST(PythonOpGen, TypeAnnotateSingleTypeTensor) {
  constexpr char kBaseOpDef[] = R"(
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
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string typed_bar =
      "def bar(x: _atypes.TensorFuzzingAnnotation[_atypes.String], y: "
      "_atypes.TensorFuzzingAnnotation[_atypes.QInt8], "
      "name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:";
  ExpectHasSubstr(code, typed_bar);

  const string untyped_bar = "def bar(x, y, name=None):";
  ExpectDoesNotHaveSubstr(code, untyped_bar);
}

TEST(PythonOpGen, TypeAnnotateMultiTypeTensor) {
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
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string typed_foo =
      "def foo(x: _atypes.TensorFuzzingAnnotation[TV_Foo_T], y: "
      "_atypes.TensorFuzzingAnnotation[TV_Foo_T2], name=None) "
      "-> _atypes.TensorFuzzingAnnotation[TV_Foo_T]:";
  ExpectHasSubstr(code, typed_foo);
}

TEST(PythonOpGen, GenerateCorrectTypeVars) {
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
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string typevars_foo = R"(
TV_Foo_T = TypeVar("TV_Foo_T", _atypes.Int8, _atypes.UInt8)
TV_Foo_T2 = TypeVar("TV_Foo_T2", _atypes.Float32, _atypes.Float64, _atypes.String)
)";

  ExpectHasSubstr(code, typevars_foo);
}

TEST(PythonOpGen, TypeAnnotateFallback) {
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
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string typed_foo_fallback =
      "def foo_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_Foo_T], y: "
      "_atypes.TensorFuzzingAnnotation[TV_Foo_T2], name, ctx) -> "
      "_atypes.TensorFuzzingAnnotation[TV_Foo_T]:";
  ExpectHasSubstr(code, typed_foo_fallback);
}

TEST(PythonOpGen, GenerateTypeVarAboveOp) {
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
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string typevar_foo = "TV_Foo_";
  const string def_foo = "def foo";
  ExpectSubstrOrder(code, typevar_foo, def_foo);
}

TEST(PythonOpGen, TypeAnnotateDefaultParams) {
  constexpr char kBaseOpDef[] = R"(
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
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string params =
      "def foo_bar(x: _atypes.TensorFuzzingAnnotation[_atypes.Float32], t: "
      "TV_FooBar_t, "
      "var1:bool=False, var2:int=0, name=None)";
  const string params_fallback =
      "def foo_bar_eager_fallback(x: "
      "_atypes.TensorFuzzingAnnotation[_atypes.Float32], t: "
      "TV_FooBar_t, var1: bool, var2: int, name, ctx)";
  ExpectHasSubstr(code, params);
  ExpectHasSubstr(code, params_fallback);
}

TEST(PythonOpGen, NoTypingSequenceTensors) {
  constexpr char kBaseOpDef[] = R"(
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

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code =
      GetPythonOps(op_defs, api_def_map, OpRegOffsets(), /* hidden_ops= */ {},
                   /* source_file_list= */ {});

  const string baz_def_line = "def baz(inputs, name=None):";

  ExpectHasSubstr(code, baz_def_line);
}

TEST(PythonOpGen, InsertCommentsForSourceFileLocation) {
  std::vector<string> source_file_list{"some_ops.cc", "another_ops.cc"};
  OpList op_defs;
  ApiDefMap api_def_map(op_defs);
  string code = GetPythonOps(op_defs, api_def_map, OpRegOffsets(),
                             /* hidden_ops= */ {}, source_file_list);

  ExpectHasSubstr(code,
                  "Original C++ source file: some_ops.cc, another_ops.cc");
}

TEST(PythonOpGen, GenerateMetadataWhenOpRegOffsetsIsPresent) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Baz"
  }
  )";

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  OpRegOffsets offsets;
  auto* offset = offsets.add_offsets();
  offset->set_name("Baz");
  offset->set_filepath("some_ops.cc");
  offset->set_start(0);
  offset->set_end(0);

  string code = GetPythonOps(op_defs, api_def_map, offsets, {}, {});

  ExpectHasSubstr(code, "# kythe.proto.metadata.GeneratedCodeInfo:");
}

TEST(PythonOpGen, NotGenerateMetadataWhenOpRegOffsetsIsEmpty) {
  OpList op_defs;
  ApiDefMap api_def_map(op_defs);
  string code = GetPythonOps(op_defs, api_def_map, OpRegOffsets(), {}, {});

  ExpectDoesNotHaveSubstr(code, "# kythe.proto.metadata.GeneratedCodeInfo:");
}

}  // namespace
}  // namespace tensorflow
