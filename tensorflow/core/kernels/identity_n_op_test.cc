/* Copyright 2015-2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class IdentityNOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input0_type, DataType input1_type) {
    TF_CHECK_OK(NodeDefBuilder("op", "IdentityN")
                    .Input(FakeInput({input0_type, input1_type}))
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(IdentityNOpTest, Int32DoubleSuccess_6) {
  TF_ASSERT_OK(Init(DT_INT32, DT_DOUBLE));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<double>(TensorShape({6}),
                            {7.3, 8.3, 9.3, 10.3, 11.3, 12.3});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected0(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected0, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected0, *GetOutput(0));
  Tensor expected1(allocator(), DT_DOUBLE, TensorShape({6}));
  test::FillValues<double>(&expected1, {7.3, 8.3, 9.3, 10.3, 11.3, 12.3});
  test::ExpectTensorEqual<double>(expected1, *GetOutput(1));
}

TEST_F(IdentityNOpTest, Int32Success_2_3) {
  TF_ASSERT_OK(Init(DT_INT32, DT_INT32));
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({2, 3}), {7, 8, 9, 10, 11, 12});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  test::FillValues<int32>(&expected, {7, 8, 9, 10, 11, 12});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(1));
}

TEST_F(IdentityNOpTest, StringInt32Success) {
  TF_ASSERT_OK(Init(DT_STRING, DT_INT32));
  AddInputFromArray<string>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  AddInputFromArray<int32>(TensorShape({8}), {1, 3, 5, 7, 9, 11, 13, 15});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected0(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<string>(&expected0, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<string>(expected0, *GetOutput(0));
  Tensor expected1(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected1, {1, 3, 5, 7, 9, 11, 13, 15});
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(1));
}

}  // namespace
}  // namespace tensorflow
