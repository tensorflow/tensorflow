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

#include "tensorflow/compiler/xla/shape.h"

#include <numeric>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(ShapeTest, ProgramShapeToFromProto) {
  ProgramShape program_shape;
  *program_shape.add_parameters() = ShapeUtil::MakeShape(F32, {1, 2, 3});
  *program_shape.add_parameters() = ShapeUtil::MakeTokenShape();
  *program_shape.add_parameters() = ShapeUtil::MakeShape(S64, {});
  *program_shape.add_parameters() = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()}),
       ShapeUtil::MakeShape(F32, {42, 42})});

  *program_shape.mutable_result() = ShapeUtil::MakeShape(F32, {7});

  program_shape.add_parameter_names("foo");
  program_shape.add_parameter_names("bar");
  program_shape.add_parameter_names("baz");
  program_shape.add_parameter_names("qux qux");

  // Create a copy of the program shape by round-tripping through a proto.
  ProgramShape program_shape_copy(program_shape.ToProto());
  ASSERT_EQ(program_shape.parameters_size(),
            program_shape_copy.parameters_size());
  for (int i = 0; i < program_shape.parameters_size(); ++i) {
    EXPECT_TRUE(ShapeUtil::Equal(program_shape.parameters(i),
                                 program_shape_copy.parameters(i)));
  }

  EXPECT_TRUE(
      ShapeUtil::Equal(program_shape.result(), program_shape_copy.result()));

  ASSERT_EQ(program_shape.parameter_names_size(),
            program_shape_copy.parameter_names_size());
  for (int i = 0; i < program_shape.parameter_names_size(); ++i) {
    EXPECT_EQ(program_shape.parameter_names(i),
              program_shape_copy.parameter_names(i));
  }
}

TEST(ShapeTest, ProgramShapeToString) {
  Shape opaque = ShapeUtil::MakeOpaqueShape();
  Shape token = ShapeUtil::MakeTokenShape();
  Shape scalar = ShapeUtil::MakeShape(F32, {});
  Shape matrix = ShapeUtil::MakeShape(U32, {1, 2});
  Shape matrix2 = ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {0, 1});
  Shape tuple = ShapeUtil::MakeTupleShape({opaque, scalar, matrix, matrix2});
  Shape nested_tuple = ShapeUtil::MakeTupleShape({tuple, matrix, token});

  ProgramShape prog = ShapeUtil::MakeProgramShape(
      {opaque, scalar, matrix, matrix2, tuple, nested_tuple}, nested_tuple);
  EXPECT_EQ(
      "((unknown): opaque[], "
      "(unknown): f32[], "
      "(unknown): u32[1,2], "
      "(unknown): s32[3,4], "
      "(unknown): (opaque[], f32[], u32[1,2], s32[3,4]), "
      "(unknown): ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      ShapeUtil::HumanString(prog));

  prog.add_parameter_names("arg0");
  prog.add_parameter_names("scalar");
  prog.add_parameter_names("matrix");
  prog.add_parameter_names("matrix2");
  prog.add_parameter_names("tuple");
  prog.add_parameter_names("nested_tuple");
  EXPECT_EQ(
      "(arg0: opaque[], "
      "scalar: f32[], "
      "matrix: u32[1,2], "
      "matrix2: s32[3,4], "
      "tuple: (opaque[], f32[], u32[1,2], s32[3,4]), "
      "nested_tuple: ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], "
      "token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      ShapeUtil::HumanString(prog));
}

}  // namespace
}  // namespace xla
