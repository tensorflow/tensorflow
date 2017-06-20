/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ParsingOpsTest, DecodeRaw_ShapeFn) {
  ShapeInferenceTestOp op("DecodeRaw");

  // Output is input + an unknown dim.
  INFER_OK(op, "?", "?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,d0_2,?]");
}

TEST(ParsingOpsTest, DecodeCSV_ShapeFn) {
  ShapeInferenceTestOp op("DecodeCSV");
  auto set_n_outputs = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    std::vector<DataType> out_types;
    for (int i = 0; i < n; ++i) {
      src_list.emplace_back("b", 0, DT_FLOAT);
      out_types.push_back(DT_FLOAT);
    }
    TF_ASSERT_OK(NodeDefBuilder("test", "DecodeCSV")
                     .Input("a", 0, DT_STRING)
                     .Input(src_list)
                     .Attr("OUT_TYPE", out_types)
                     .Finalize(&op.node_def));
  };

  // Output is always n copies of input 0.
  set_n_outputs(2);
  INFER_OK(op, "?;?;?", "in0;in0");
  INFER_OK(op, "[1,2,?,4];?;?", "in0;in0");
  INFER_OK(op, "[1,2,?,4];[?];[?]", "in0;in0");

  // Check errors in the record_defaults inputs.
  INFER_ERROR("must be rank 1", op, "?;?;[]");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("Shape of a default must be", op, "?;?;[2]");
  INFER_ERROR("Shape of a default must be", op, "?;[2];?");
}

static std::vector<TensorShapeProto> MakeDenseShapes(int size,
                                                     bool add_extra_shape,
                                                     int unknown_outer_dims) {
  std::vector<TensorShapeProto> shapes(size);
  for (int i = 0; i < size; ++i) {
    // Make shapes be the sequence [?,1]; [?,1,2], [?,1,2,3]...
    // where the number of prefixed ? depends on unknown_outer_dims.
    if (i == 0) {
      for (int d = 0; d < unknown_outer_dims; ++d) {
        shapes[i].add_dim()->set_size(-1);
      }
    } else {
      shapes[i] = shapes[i - 1];
    }
    shapes[i].add_dim()->set_size(i + 1);
  }
  if (add_extra_shape) {
    shapes.resize(shapes.size() + 1);
  }
  return shapes;
}

TEST(ParsingOpsTest, ParseExample_ShapeFn) {
  ShapeInferenceTestOp op("ParseExample");
  auto set_outputs = [&op](int num_sparse, int num_dense,
                           bool add_extra_shape = false,
                           int unknown_outer_dims = 0) {
    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    NodeDefBuilder::NodeOut string_in{"a", 0, DT_STRING};

    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseExample")
            .Input("serialized", 0, DT_STRING)
            .Input("names", 0, DT_STRING)
            .Input(NodeOutList(num_sparse, string_in))
            .Input(NodeOutList(num_dense, string_in))
            .Input(NodeOutList(num_dense, string_in))
            .Attr("sparse_types", DataTypeList(num_sparse, DT_FLOAT))
            .Attr("dense_types", DataTypeList(num_dense, DT_FLOAT))
            .Attr("dense_shapes", MakeDenseShapes(num_dense, add_extra_shape,
                                                  unknown_outer_dims))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'names'.
  set_outputs(0 /* num_sparse */, 0 /* num_dense */);
  INFER_OK(op, "?;?", "");
  INFER_OK(op, "[10];[20]", "");
  INFER_ERROR("must be rank 1", op, "[1,2];?");
  INFER_ERROR("must be rank 1", op, "?;[2,3]");

  // Verify the sparse and dense outputs.
  set_outputs(2 /* num_sparse */, 3 /* num_dense */);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"  // sparse outputs
            "[?,1];[?,1,2];[?,1,2,3]"));    // dense outputs
  INFER_OK(op, "[10];?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"         // sparse outputs
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3]"));  // dense outputs

  // Confirm an error from ParseSingleExampleAttrs.Init().
  set_outputs(2, 3, true /* add_extra_shape */);
  INFER_ERROR("len(dense_keys) != len(dense_shapes)", op,
              "?;?;?;?;?;?;?;?;?;?");

  // Allow variable strides
  set_outputs(2, 3, false /* add_extra_shape */, 1 /* unknown_outer_dims */);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"      // sparse outputs
            "[?,?,1];[?,?,1,2];[?,?,1,2,3]"));  // dense outputs
  INFER_OK(op, "[10];?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"               // sparse outputs
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3]"));  // dense outputs

  set_outputs(2, 3, true /* add_extra_shape */, 1 /* unknown_outer_dims */);
  INFER_ERROR("len(dense_keys) != len(dense_shapes)", op,
              "?;?;?;?;?;?;?;?;?;?");

  // Variable inner dimensions are not supported
  set_outputs(2, 3, false /* add_extra_shape */, 2 /* unknown_outer_dims */);
  INFER_ERROR("shapes[0] has unknown rank or unknown inner dimensions", op,
              "?;?;?;?;?;?;?;?;?;?");
}

TEST(ParsingOpsTest, ParseSingleSequenceExample_ShapeFn) {
  ShapeInferenceTestOp op("ParseSingleSequenceExample");
  auto set_outputs = [&op](int num_context_sparse, int num_context_dense,
                           int num_feature_list_sparse,
                           int num_feature_list_dense,
                           bool add_extra_shape = false) {
    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    NodeDefBuilder::NodeOut string_in{"a", 0, DT_STRING};

    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseSingleSequenceExample")
            .Input("serialized", 0, DT_STRING)
            .Input("feature_list_dense_missing_assumed_empty", 0, DT_STRING)
            .Input(NodeOutList(num_context_sparse, string_in))
            .Input(NodeOutList(num_context_dense, string_in))
            .Input(NodeOutList(num_feature_list_sparse, string_in))
            .Input(NodeOutList(num_feature_list_dense, string_in))
            .Input(NodeOutList(num_context_dense, string_in))
            .Input("debug_name", 0, DT_STRING)
            .Attr("context_sparse_types",
                  DataTypeList(num_context_sparse, DT_FLOAT))
            .Attr("context_dense_types",
                  DataTypeList(num_context_dense, DT_FLOAT))
            .Attr("context_dense_shapes",
                  MakeDenseShapes(num_context_dense, add_extra_shape, 0))
            .Attr("feature_list_sparse_types",
                  DataTypeList(num_feature_list_sparse, DT_FLOAT))
            .Attr("feature_list_dense_types",
                  DataTypeList(num_feature_list_dense, DT_FLOAT))
            .Attr("feature_list_dense_shapes",
                  MakeDenseShapes(num_feature_list_dense, add_extra_shape, 0))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'feature_list_dense_missing_assumed_empty'.
  set_outputs(0, 0, 0, 0);
  INFER_OK(op, "?;?;?", "");
  INFER_OK(op, "[];[20];?", "");
  INFER_ERROR("must be rank 0", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[2,3];?");

  // context inputs with no feature_list inputs.
  set_outputs(2 /* num_context_sparse */, 3 /* num_context_dense */, 0, 0);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?;?",
           ("[?,1];[?,1];[?];[?];[1];[1];"  // context sparse outputs
            "[1];[1,2];[1,2,3]"));          // context dense outputs

  // feature_list inputs with no context inputs.
  set_outputs(0, 0, 2 /* num_feature_list_sparse */,
              3 /* num_feature_list_dense */);
  INFER_OK(op, "?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"  // feature_list sparse outputs
            "[?,1];[?,1,2];[?,1,2,3]"));    // feature_list dense outputs

  // Combine previous two test cases.
  set_outputs(2, 3, 2, 3);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?;?;?;?;?;?;?",
           ("[?,1];[?,1];[?];[?];[1];[1];"  // context sparse outputs
            "[1];[1,2];[1,2,3];"            // context dense outputs
            "[?,2];[?,2];[?];[?];[2];[2];"  // feature_list sparse outputs
            "[?,1];[?,1,2];[?,1,2,3]"));    // feature_list dense outputs

  // Confirm an error from ParseSingleSequenceExampleAttrs.Init().
  set_outputs(1, 1, 1, 1, true /* add_extra_shape */);
  INFER_ERROR("len(context_dense_keys) != len(context_dense_shapes)", op,
              "?;?;?;?;?;?;?;?");
}

}  // end namespace tensorflow
