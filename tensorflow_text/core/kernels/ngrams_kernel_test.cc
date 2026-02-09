// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

TEST(NgramsStringJoin, UnknownRank) {
  ShapeInferenceTestOp op("TFText>NgramsStringJoin");
  op.input_tensors.resize(1);
  AddNodeAttr("RAGGED_RANK", 0, &op.node_def);
  AddNodeAttr("width", 1, &op.node_def);

  INFER_OK(op, "?", "?");
}

TEST(NgramsStringJoin, KnownRankUnknownDims) {
  ShapeInferenceTestOp op("TFText>NgramsStringJoin");
  op.input_tensors.resize(1);
  AddNodeAttr("RAGGED_RANK", 0, &op.node_def);
  AddNodeAttr("width", 1, &op.node_def);

  INFER_OK(op, "[1,?]", "[1,?]");
}

TEST(NgramsStringJoin, LastDimWidth) {
  ShapeInferenceTestOp op("TFText>NgramsStringJoin");
  op.input_tensors.resize(1);
  AddNodeAttr("RAGGED_RANK", 0, &op.node_def);
  AddNodeAttr("width", 3, &op.node_def);

  INFER_OK(op, "[?,5]", "[?,3]");
}

TEST(NgramsStringJoin, LastDimWidthClampZero) {
  ShapeInferenceTestOp op("TFText>NgramsStringJoin");
  op.input_tensors.resize(1);
  AddNodeAttr("RAGGED_RANK", 0, &op.node_def);
  AddNodeAttr("width", 3, &op.node_def);

  INFER_OK(op, "[?,1]", "[?,0]");
}

}  // end namespace tensorflow
