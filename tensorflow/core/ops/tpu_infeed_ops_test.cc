/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(InfeedDequeueTupleTest, MatchingShapesAndDtypes) {
  ShapeInferenceTestOp op("InfeedDequeueTuple");
  TF_ASSERT_OK(
      NodeDefBuilder("test", "InfeedDequeueTuple")
          .Attr("dtypes", std::vector<DataType>{DT_INT32, DT_FLOAT})
          .Attr("shapes",
                std::vector<PartialTensorShape>{PartialTensorShape({1}),
                                                PartialTensorShape({2, -1})})
          .Finalize(&op.node_def));
  INFER_OK(op, "", "[1];[2,?]");
}

TEST(InfeedDequeueTupleTest, MoreShapesThanDtypes) {
  ShapeInferenceTestOp op("InfeedDequeueTuple");
  TF_ASSERT_OK(NodeDefBuilder("test", "InfeedDequeueTuple")
                   .Attr("dtypes", std::vector<DataType>{DT_INT32})
                   .Attr("shapes",
                         std::vector<PartialTensorShape>{
                             PartialTensorShape({1}), PartialTensorShape({1})})
                   .Finalize(&op.node_def));
  INFER_ERROR("The length of shapes (2) must match the length of dtypes (1)",
              op, "");
}

TEST(InfeedDequeueTupleTest, FewerShapesThanDtypes) {
  ShapeInferenceTestOp op("InfeedDequeueTuple");
  TF_ASSERT_OK(
      NodeDefBuilder("test", "InfeedDequeueTuple")
          .Attr("dtypes", std::vector<DataType>{DT_INT32, DT_INT32})
          .Attr("shapes",
                std::vector<PartialTensorShape>{PartialTensorShape({1})})
          .Finalize(&op.node_def));
  INFER_ERROR("The length of shapes (1) must match the length of dtypes (2)",
              op, "");
}

}  // namespace
}  // namespace tensorflow
