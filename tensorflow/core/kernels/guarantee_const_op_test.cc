/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GuaranteeConstOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type) {
    TF_CHECK_OK(NodeDefBuilder("op", "GuaranteeConst")
                    .Input(FakeInput(input_type))
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(GuaranteeConstOpTest, Int32Success_6) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(GuaranteeConstOpTest, Int32Success_2_3) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(GuaranteeConstOpTest, StringSuccess) {
  TF_ASSERT_OK(Init(DT_STRING));
  AddInputFromArray<tstring>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<tstring>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(GuaranteeConstOpTest, ResourceInputError) {
  TF_ASSERT_OK(Init(DT_RESOURCE));
  AddResourceInput("", "resource", new Var(DT_INT32));
  const auto status = RunOpKernel();
  ASSERT_EQ(error::INVALID_ARGUMENT, status.code());
}

}  // namespace
}  // namespace tensorflow
