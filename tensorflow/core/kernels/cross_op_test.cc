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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class CrossOpTest : public OpsTestBase {
 protected:
  CrossOpTest() {
    TF_EXPECT_OK(NodeDefBuilder("cross_op", "Cross")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(CrossOpTest, Zero) {
  AddInputFromArray<float>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<float>(TensorShape({3}), {0, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {0, 0, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CrossOpTest, RightHandRule) {
  AddInputFromArray<float>(TensorShape({2, 3}), {1, 0, 0, /**/ 0, 1, 0});
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 1, 0, /**/ 1, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {{0, 0, 1, /**/ 0, 0, -1}});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CrossOpTest, ArbitraryNonintegral) {
  const float u1 = -0.669, u2 = -0.509, u3 = 0.125;
  const float v1 = -0.477, v2 = 0.592, v3 = -0.110;
  const float s1 = u2 * v3 - u3 * v2;
  const float s2 = u3 * v1 - u1 * v3;
  const float s3 = u1 * v2 - u2 * v1;

  AddInputFromArray<float>(TensorShape({3}), {u1, u2, u3});
  AddInputFromArray<float>(TensorShape({3}), {v1, v2, v3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {s1, s2, s3});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

class CrossOpIntTest : public OpsTestBase {
 protected:
  CrossOpIntTest() {
    TF_EXPECT_OK(NodeDefBuilder("cross_int_op", "Cross")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(CrossOpIntTest, RightHandRule) {
  AddInputFromArray<int>(TensorShape({2, 3}), {2, 0, 0, /**/ 0, 2, 0});
  AddInputFromArray<int>(TensorShape({2, 3}), {0, 2, 0, /**/ 2, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int>(&expected, {{0, 0, 4, /**/ 0, 0, -4}});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

}  // namespace tensorflow
