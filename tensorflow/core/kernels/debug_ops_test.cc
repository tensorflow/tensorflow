/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class DebugIdentityOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type) {
    TF_CHECK_OK(NodeDefBuilder("op", "DebugIdentity")
                    .Input(FakeInput(input_type))
                    .Attr("tensor_name", "FakeTensor:0")
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(DebugIdentityOpTest, Int32Success_6) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  // Verify the identity output
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(DebugIdentityOpTest, Int32Success_2_3) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(DebugIdentityOpTest, StringSuccess) {
  TF_ASSERT_OK(Init(DT_STRING));
  AddInputFromArray<string>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<string>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(DebugIdentityOpTest, RefInputError) { TF_ASSERT_OK(Init(DT_INT32_REF)); }

// Tests for DebugNanCountOp
class DebugNanCountOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type) {
    TF_CHECK_OK(NodeDefBuilder("op", "DebugNanCount")
                    .Input(FakeInput(input_type))
                    .Attr("tensor_name", "FakeTensor:0")
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(DebugNanCountOpTest, Float_has_NaNs) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(TensorShape({6}),
                           {1.1, std::numeric_limits<float>::quiet_NaN(), 3.3,
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(), 6.6});
  TF_ASSERT_OK(RunOpKernel());

  // Verify the NaN-count debug signal
  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {3});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

TEST_F(DebugNanCountOpTest, Float_no_NaNs) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(
      TensorShape({6}),
      {1.1, 2.2, 3.3, std::numeric_limits<float>::infinity(), 5.5, 6.6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {0});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

TEST_F(DebugNanCountOpTest, Double_has_NaNs) {
  TF_ASSERT_OK(Init(DT_DOUBLE));
  AddInputFromArray<double>(TensorShape({6}),
                            {1.1, std::numeric_limits<double>::quiet_NaN(), 3.3,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(), 6.6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {3});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

TEST_F(DebugNanCountOpTest, Double_no_NaNs) {
  TF_ASSERT_OK(Init(DT_DOUBLE));
  AddInputFromArray<double>(
      TensorShape({6}),
      {1.1, 2.2, 3.3, std::numeric_limits<double>::infinity(), 5.5, 6.6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {0});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow
