// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
// =============================================================================
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(StatsOpsTest, CreateFertileStatsVariable_ShapeFn) {
  ShapeInferenceTestOp op("CreateFertileStatsVariable");
  INFER_OK(op, "[1];[1]", "");
}

TEST(StatsOpsTest, FertileStatsSerialize_ShapeFn) {
  ShapeInferenceTestOp op("FertileStatsSerialize");
  INFER_OK(op, "[1]", "[]");
}

TEST(StatsOpsTest, FertileStatsDeserialize_ShapeFn) {
  ShapeInferenceTestOp op("FertileStatsDeserialize");
  INFER_OK(op, "[1];[1]", "");
}

TEST(StatsOpsTest, GrowTreeV4_ShapeFn) {
  ShapeInferenceTestOp op("GrowTreeV4");
  INFER_OK(op, "[1];[1];?", "");
}

TEST(StatsOpsTest, ProcessInputV4_ShapeFn) {
  ShapeInferenceTestOp op("ProcessInputV4");
  INFER_OK(op, "[1];[1];?;?;?;?;?;?;?", "[?]");
}

TEST(StatsOpsTest, FinalizeTree_ShapeFn) {
  ShapeInferenceTestOp op("FinalizeTree");
  INFER_OK(op, "[1];[1]", "");
}

}  // namespace tensorflow
