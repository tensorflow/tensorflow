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

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class GruOpsTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    TF_Status* status = TF_NewStatus();
    auto* lib = TF_LoadLibrary(
        "tensorflow/contrib/rnn/python/ops/_gru_ops.so", status);
    TF_Code code = TF_GetCode(status);
    string status_msg(TF_Message(status));
    TF_DeleteStatus(status);
    ASSERT_EQ(TF_OK, code) << status_msg;
    TF_DeleteLibraryHandle(lib);
  }
};

TEST_F(GruOpsTest, GRUBlockCell_ShapeFn) {
  ShapeInferenceTestOp op("GRUBlockCell");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;[?];?;?;?;?");

  // Output
  INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?,?];[?,?];[?,?]");
  INFER_OK(op, "[?,?];[?,?];?;?;?;?",
           "[d0_0,d1_1];[d0_0,d1_1];[d0_0,d1_1];[d0_0,d1_1]");
}

TEST_F(GruOpsTest, GRUBlockCellGrad_ShapeFn) {
  ShapeInferenceTestOp op("GRUBlockCellGrad");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?;?;?;?;?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;[?];?;?;?;?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;?;[?];?;?;?;?;?;?;?");

  // Output
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?", "[?,?];[?,?];[?,?];[?,?]");
  INFER_OK(op, "[?,?];[?,?];[?,?];?;?;?;?;?;?;?",
           "in0;[d0_0,d1_1];[d0_0,d1_1];[d0_0,d2_1]");
}

}  // namespace tensorflow
