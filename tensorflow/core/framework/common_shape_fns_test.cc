/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/common_shape_fns.h"

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

TEST(CommonShapeFnsTest, NoOutputShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("Assert")
                  .Input("condition: bool")
                  .Input("data: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "Assert")
                  .Input("condition", 0, DT_BOOL)
                  .Input({{"data", 0, DT_FLOAT}})
                  .Finalize(&def));

  InferenceContext c(&def, op_def, {"[]", "[10]"}, {});
  TF_EXPECT_OK(NoOutputs(&c));
  EXPECT_EQ(0, c.num_outputs());
}

TEST(CommonShapeFnsTest, ScalarShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("L2Loss")
                  .Input("t: float")
                  .Output("t: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(
      NodeDefBuilder("test", "L2Loss").Input("t", 0, DT_FLOAT).Finalize(&def));

  {
    InferenceContext c(&def, op_def, {"[]"}, {});
    TF_EXPECT_OK(ScalarShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ(0, c.Rank(output));
  }

  {
    InferenceContext c(&def, op_def, {"[1,23,4,4,2]"}, {});
    TF_EXPECT_OK(ScalarShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ(0, c.Rank(output));
  }
}

TEST(CommonShapeFnsTest, MatMulShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("MatMul")
                  .Input("a: float")
                  .Input("b: float")
                  .Output("c: float")
                  .Attr("transpose_a:bool=false")
                  .Attr("transpose_b:bool=false")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                  .Input("a", 0, DT_FLOAT)
                  .Input("b", 0, DT_FLOAT)
                  .Attr("transpose_a", false)
                  .Attr("transpose_b", false)
                  .Finalize(&def));

  {
    InferenceContext c(&def, op_def, {"[2,3]", "[3,4]"}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // Unknown inner dimension for one
    InferenceContext c(&def, op_def, {"[2,?]", "[3,4]"}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // Invalid rank.
    InferenceContext c(&def, op_def, {"[2]", "[3,4]"}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ("Invalid argument: Shape must be rank 2 but is rank 1",
              s.ToString());
  }

  {
    // Unknown outer dimension
    InferenceContext c(&def, op_def, {"[2,3]", "[3,?]"}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_FALSE(c.ValueKnown(c.Dim(output, 1)));
  }

  {
    // Inner shapes not compatible
    InferenceContext c(&def, op_def, {"[2,5]", "[3,4]"}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ("Invalid argument: Dimensions must be equal, but are 5 and 3",
              s.ToString());
  }

  {
    // Inner shapes not compatible
    InferenceContext c(&def, op_def, {"[2,5,3]", "[3,5,4]"}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ("Invalid argument: Shape must be rank 2 but is rank 3",
              s.ToString());
  }

  {
    // transpose_a
    TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("transpose_a", true)
                    .Attr("transpose_b", false)
                    .Attr("type", DT_FLOAT)
                    .Finalize(&def));

    InferenceContext c(&def, op_def, {"[3,2]", "[3,4]"}, {});
    auto s = MatMulShape(&c);
    const Shape* output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // transpose_b
    TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("transpose_a", false)
                    .Attr("transpose_b", true)
                    .Attr("type", DT_FLOAT)
                    .Finalize(&def));

    InferenceContext c(&def, op_def, {"[2,3]", "[4,3]"}, {});
    auto s = MatMulShape(&c);
    const Shape* output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }
}

TEST(CommonShapeFnsTest, BiasAddShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("BiasAdd")
                  .Input("a: float")
                  .Input("b: float")
                  .Output("c: float")
                  .Finalize(&op_reg_data));

  OpDef op_def = op_reg_data.op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                  .Input("a", 0, DT_FLOAT)
                  .Input("b", 0, DT_FLOAT)
                  .Finalize(&def));

  {
    InferenceContext c(&def, op_def, {"[2,10]", "[10]"}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(10, c.Value(c.Dim(output, 1)));
  }

  {
    // Unknown ranks.
    InferenceContext c(&def, op_def, {"?", "?"}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    const Shape* output = c.output(0);
    EXPECT_FALSE(c.RankKnown(output));
  }

  {
    // Rank > 2
    InferenceContext c(&def, op_def, {"[4,3,4,2,15]", "[15]"}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ("[4,3,4,2,15]", c.DebugString(output));
  }

  {
    // NCHW format
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(&def, op_def, {"[2,3,4,5]", "[3]"}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ("[2,3,4,5]", c.DebugString(output));
  }

  {
    // NCHW format with high input rank
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(&def, op_def, {"[8,6,4,2,3,4,5]", "[3]"}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    const Shape* output = c.output(0);
    EXPECT_EQ("[8,6,4,2,3,4,5]", c.DebugString(output));
  }

  {
    // Input rank not high enough
    InferenceContext c(&def, op_def, {"[3]", "[3]"}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }

  {
    // NCHW rank not high enough
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    // NCHW format
    InferenceContext c(&def, op_def, {"[2,3,4]", "[3]"}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }
}

}  // namespace shape_inference
}  // namespace tensorflow
