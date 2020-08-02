/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST_F(OpsTestBase, DenseCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("DenseCountSparseOutput");
  INFER_OK(op, "[?];?", "[?,1];[?];[1]");
  INFER_OK(op, "[?,?];?", "[?,2];[?];[2]");
}

TEST_F(OpsTestBase, SparseCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("SparseCountSparseOutput");
  INFER_OK(op, "[?,1];?;?;?", "[?,d0_1];[?];[d0_1]");
  INFER_OK(op, "[?,2];?;?;?", "[?,d0_1];[?];[d0_1]");
}

TEST_F(OpsTestBase, RaggedCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("RaggedCountSparseOutput");
  INFER_OK(op, "?;[?];?", "[?,2];[?];[2]");
}
}  // namespace
}  // namespace tensorflow
