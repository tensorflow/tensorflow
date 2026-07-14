/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(AllToAll, UnknownRank) {
  ShapeInferenceTestOp op("AllToAll");
  op.input_tensors.resize(2);

  INFER_OK(op, "?;?", "?");
}

TEST(AllToAll, KnownRankUnknownDims) {
  ShapeInferenceTestOp op("AllToAll");
  op.input_tensors.resize(2);
  AddNodeAttr("concat_dimension", 0, &op.node_def);
  AddNodeAttr("split_count", 1, &op.node_def);
  AddNodeAttr("split_dimension", 1, &op.node_def);

  // split_dimension is unknown.
  INFER_OK(op, "[?,1];[?,?]", "?");
  // concat_dimension is unknown.
  INFER_OK(op, "[1,?];[?,?]", "?");
  // Both split_dimension and concat_dimension are unknown.
  INFER_OK(op, "[?,?];[?,?]", "?");
}

}  // end namespace tensorflow
