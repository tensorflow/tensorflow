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

#include <cstdint>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

void BuildNcclBroadcastRecv(DataType shape_type, ShapeInferenceTestOp* op) {
  TF_ASSERT_OK(NodeDefBuilder("test", "_NcclBroadcastRecv")
                   .Input("shape", 0, shape_type)
                   .Attr("T", DT_FLOAT)
                   .Attr("num_devices", 2)
                   .Attr("shared_name", "s")
                   .Finalize(&op->node_def));
}

}  // namespace

TEST(NcclOpsTest, NcclBroadcastRecv_ShapeFn_Int32) {
  ShapeInferenceTestOp op("_NcclBroadcastRecv");
  BuildNcclBroadcastRecv(DT_INT32, &op);

  INFER_OK(op, "[3]", "[?,?,?]");
  INFER_ERROR("Shape must be rank 1", op, "[1,2]");

  Tensor shape_t = test::AsTensor<int32_t>({2, 3});
  op.input_tensors.resize(1);
  op.input_tensors[0] = &shape_t;
  INFER_OK(op, "[2]", "[2,3]");
}

TEST(NcclOpsTest, NcclBroadcastRecv_ShapeFn_Int64) {
  ShapeInferenceTestOp op("_NcclBroadcastRecv");
  BuildNcclBroadcastRecv(DT_INT64, &op);

  INFER_OK(op, "[3]", "[?,?,?]");

  // A dimension that does not fit in int32 must be preserved.
  Tensor shape_t = test::AsTensor<int64_t>({int64_t{1} << 32, 2});
  op.input_tensors.resize(1);
  op.input_tensors[0] = &shape_t;
  INFER_OK(op, "[2]", "[4294967296,2]");
}

}  // namespace tensorflow
