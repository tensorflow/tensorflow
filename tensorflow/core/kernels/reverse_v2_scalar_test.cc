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

// Test for GitHub issue #110038
// ReverseV2 should validate axis range for scalar tensors

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class ReverseV2OpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, DataType index_type = DT_INT32) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ReverseV2")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Attr("T", data_type)
                     .Attr("Tidx", index_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// Test case 1: Scalar tensor with invalid axis [1,2,3] should raise error
TEST_F(ReverseV2OpTest, ScalarTensorWithInvalidAxes) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create scalar tensor
  AddInputFromArray<float>(TensorShape({}), {4.0f});
  
  // Invalid axes for scalar (rank 0) - any axis is invalid
  AddInputFromArray<int32>(TensorShape({3}), {1, 2, 3});
  
  // This should fail with InvalidArgument error
  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "out of valid range"))
      << "Expected 'out of valid range' error, but got: " << s.message();
}

// Test case 2: Scalar tensor with axis [0] should raise error
TEST_F(ReverseV2OpTest, ScalarTensorWithAxisZero) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create scalar tensor
  AddInputFromArray<float>(TensorShape({}), {4.0f});
  
  // Axis [0] is invalid for scalar (rank 0)
  AddInputFromArray<int32>(TensorShape({1}), {0});
  
  // This should fail with InvalidArgument error
  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "out of valid range"))
      << "Expected 'out of valid range' error, but got: " << s.message();
}

// Test case 3: Scalar tensor with negative axis [-1] should raise error
TEST_F(ReverseV2OpTest, ScalarTensorWithNegativeAxis) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create scalar tensor
  AddInputFromArray<float>(TensorShape({}), {4.0f});
  
  // Axis [-1] should canonicalize to -1 + 0 = -1, which is invalid
  AddInputFromArray<int32>(TensorShape({1}), {-1});
  
  // This should fail with InvalidArgument error
  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "out of valid range"))
      << "Expected 'out of valid range' error, but got: " << s.message();
}

// Test case 4: Scalar tensor with empty axis should succeed
TEST_F(ReverseV2OpTest, ScalarTensorWithEmptyAxis) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create scalar tensor
  AddInputFromArray<float>(TensorShape({}), {4.0f});
  
  // Empty axis list is valid - no reversal, just return input
  AddInputFromArray<int32>(TensorShape({0}), {});
  
  // This should succeed
  TF_ASSERT_OK(RunOpKernel());
  
  // Output should equal input
  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->scalar<float>()(), 4.0f);
}

// Test case 5: Non-scalar tensor with valid axis should succeed (control test)
TEST_F(ReverseV2OpTest, VectorTensorWithValidAxis) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create vector tensor
  AddInputFromArray<float>(TensorShape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
  
  // Axis [0] is valid for vector (rank 1)
  AddInputFromArray<int32>(TensorShape({1}), {0});
  
  // This should succeed
  TF_ASSERT_OK(RunOpKernel());
  
  // Output should be reversed
  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {4.0f, 3.0f, 2.0f, 1.0f});
  test::ExpectTensorEqual<float>(expected, *output);
}

// Test case 6: Vector tensor with invalid axis [1] should raise error
TEST_F(ReverseV2OpTest, VectorTensorWithInvalidAxis) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create vector tensor (rank 1)
  AddInputFromArray<float>(TensorShape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
  
  // Axis [1] is invalid for vector (rank 1) - valid range is [0, 0]
  AddInputFromArray<int32>(TensorShape({1}), {1});
  
  // This should fail with InvalidArgument error
  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "out of valid range"))
      << "Expected 'out of valid range' error, but got: " << s.message();
}

// Test case 7: Empty tensor with empty axis should succeed
TEST_F(ReverseV2OpTest, EmptyTensorWithEmptyAxis) {
  MakeOp(DT_FLOAT, DT_INT32);
  
  // Create empty tensor
  AddInputFromArray<float>(TensorShape({0}), {});
  
  // Empty axis list
  AddInputFromArray<int32>(TensorShape({0}), {});
  
  // This should succeed
  TF_ASSERT_OK(RunOpKernel());
  
  // Output should be empty
  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->NumElements(), 0);
}

}  // namespace
}  // namespace tensorflow
