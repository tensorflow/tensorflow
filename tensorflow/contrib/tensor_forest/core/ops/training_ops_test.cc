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

// TODO(cwhipkey): iwyu
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(TrainingOpsTest, UpdateFertileSlots_ShapeFn) {
  ShapeInferenceTestOp op("UpdateFertileSlots");
  INFER_OK(op, "?;?;?;?;?;?;?", "[2,?];[?];[?]");
}

TEST(TrainingOpsTest, ScatterAddNdim_ShapeFn) {
  ShapeInferenceTestOp op("ScatterAddNdim");
  INFER_OK(op, "?;?;?", "");
}

TEST(TrainingOpsTest, GrowTree_ShapeFn) {
  ShapeInferenceTestOp op("GrowTree");
  INFER_OK(op, "?;?;?;?;?;?", "[?];[?,2];[?];[1]");
}

TEST(TrainingOpsTest, FinishedNodes_ShapeFn) {
  ShapeInferenceTestOp op("FinishedNodes");
  INFER_OK(op, "?;?;?;?;?;?;?;?", "[?];[?]");
}

TEST(TrainingOpsTest, BestSplits_ShapeFn) {
  ShapeInferenceTestOp op("BestSplits");
  INFER_OK(op, "?;?;?;?;?;?", "[?]");
  INFER_OK(op, "[?];?;?;?;?;?", "[d0_0]");
  INFER_OK(op, "[1];?;?;?;?;?", "[d0_0]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2];?;?;?;?;?");
}

TEST(TrainingOpsTest, SampleInputs_ShapeFn) {
  ShapeInferenceTestOp op("SampleInputs");

  // input[6].dim(1) determines dims in the output.
  INFER_OK(op, "?;?;?;?;?;?;?;?", "[?];[?,?];[?,?]");
  INFER_OK(op, "?;?;?;?;?;?;[?,?];?", "[?];[?,d6_1];[?,d6_1]");
  INFER_OK(op, "?;?;?;?;?;?;[1,2];?", "[?];[?,d6_1];[?,d6_1]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op,
              "?;?;?;?;?;?;[1,2,3];?");
}

}  // namespace tensorflow
