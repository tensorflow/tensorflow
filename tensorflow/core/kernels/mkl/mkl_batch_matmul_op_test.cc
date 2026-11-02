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

#if defined(INTEL_MKL)

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class MklBatchMatMulV2OpTest : public OpsTestBase {
 protected:
  void MakeOp(bool adj_x = false, bool adj_y = false,
              const char* op_name = "_MklBatchMatMulV2") {
    TF_ASSERT_OK(NodeDefBuilder("batch_matmul", op_name)
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DT_FLOAT)
                     .Attr("adj_x", adj_x)
                     .Attr("adj_y", adj_y)
                     .Attr("_kernel", "MklNameChangeOp")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// A valid [1, 2] x [2, 1] product.
TEST_F(MklBatchMatMulV2OpTest, Simple) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 2}), {1.0f, 2.0f});
  AddInputFromArray<float>(TensorShape({2, 1}), {3.0f, 4.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_FLOAT, TensorShape({1, 1}));
  test::FillValues<float>(&expected, {11.0f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

TEST_F(MklBatchMatMulV2OpTest, OutputShapeOverflow) {
  MakeOp();
  // Both inputs are empty, but the output shape [2^32, 2^32] has more
  // elements than int64 can represent.
  AddInputFromArray<float>(TensorShape({int64_t{1} << 32, 0}), {});
  AddInputFromArray<float>(TensorShape({0, int64_t{1} << 32}), {});
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Encountered overflow when multiplying"));
}

TEST_F(MklBatchMatMulV2OpTest, OutputShapeLhsRowsOverflow) {
  MakeOp(/*adj_x=*/true);
  // Batch shape is [2^32] and adj_x swaps lhs_rows to 2^32, so appending
  // lhs_rows overflows int64 before rhs_cols is reached.
  AddInputFromArray<float>(
      TensorShape({int64_t{1} << 32, 0, int64_t{1} << 32}), {});
  AddInputFromArray<float>(TensorShape({int64_t{1} << 32, 0, 1}), {});
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Encountered overflow when multiplying"));
}

TEST_F(MklBatchMatMulV2OpTest, V1OutputShapeOverflow) {
  MakeOp(/*adj_x=*/false, /*adj_y=*/false, /*op_name=*/"_MklBatchMatMul");
  AddInputFromArray<float>(TensorShape({int64_t{1} << 32, 0}), {});
  AddInputFromArray<float>(TensorShape({0, int64_t{1} << 32}), {});
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Encountered overflow when multiplying"));
}

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
