/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
class FusedBatchNormOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormOpTest, Training) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_mean, {10, 10});
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_variance, {14.00, 14.00});
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 0.01);
}

TEST_F(FusedBatchNormOpTest, Inference) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {10, 10});
  AddInputFromArray<float>(TensorShape({2}), {11.67f, 11.67f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

class FusedBatchNormGradOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormGradOpTest, Simple) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_grad_op", "FusedBatchNormGrad")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {2, 2, 9, 9, -4, -4, 5, 5, 8, 8, 7, 7});
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {1, 1, 7, 7, 4, 4, -3, -3, -11, -11, 13, 13});
  AddInputFromArray<float>(TensorShape({2}), {4, 4});
  AddInputFromArray<float>(TensorShape({2}), {1.833f, 1.833f});
  AddInputFromArray<float>(TensorShape({2}), {57.472f, 57.472f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_x(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected_x, {-1.34, -1.34, 2.47, 2.47, -4.44, -4.44,
                                        0.17, 0.17, 1.60, 1.60, 1.53, 1.53});
  test::ExpectTensorNear<float>(expected_x, *GetOutput(0), 0.01);

  Tensor expected_scale(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_scale, {-1.6488, -1.6488});
  test::ExpectTensorNear<float>(expected_scale, *GetOutput(1), 0.01);

  Tensor expected_offset(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_offset, {27, 27});
  test::ExpectTensorNear<float>(expected_offset, *GetOutput(2), 0.01);
}
}  // namespace tensorflow
