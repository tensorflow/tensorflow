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

#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
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

class BatchNormOpTest : public OpsTestBase {};

TEST_F(BatchNormOpTest, Simple) {
  TF_EXPECT_OK(
      NodeDefBuilder("batch_norm_op", "BatchNormWithGlobalNormalization")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("scale_after_normalization", false)
          .Attr("variance_epsilon", 0.001)
          .Finalize(node_def()));
  TF_EXPECT_OK(InitOpWithGraphVersion(8));
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6});
  AddInputFromArray<float>(TensorShape({2}), {10, 20});
  AddInputFromArray<float>(TensorShape({2}), {0.25, 0.5});
  AddInputFromArray<float>(TensorShape({2}), {0.1, 0.6});
  AddInputFromArray<float>(TensorShape({2}), {0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(
      &expected, {-17.86, -22.00, -15.87, -20.59, -13.87, -19.18, -21.86,
                  -33.31, -23.85, -34.72, -25.85, -36.13});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

TEST_F(BatchNormOpTest, Fp16) {
  TF_EXPECT_OK(
      NodeDefBuilder("batch_norm_op", "BatchNormWithGlobalNormalization")
          .Input(FakeInput(DT_HALF))
          .Input(FakeInput(DT_HALF))
          .Input(FakeInput(DT_HALF))
          .Input(FakeInput(DT_HALF))
          .Input(FakeInput(DT_HALF))
          .Attr("scale_after_normalization", false)
          .Attr("variance_epsilon", 0.001)
          .Finalize(node_def()));
  TF_EXPECT_OK(InitOpWithGraphVersion(8));
  AddInputFromList<Eigen::half>(TensorShape({1, 1, 6, 2}),
                                {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6});
  AddInputFromList<Eigen::half>(TensorShape({2}), {10, 20});
  AddInputFromList<Eigen::half>(TensorShape({2}), {0.25, 0.5});
  AddInputFromList<Eigen::half>(TensorShape({2}), {0.1, 0.6});
  AddInputFromList<Eigen::half>(TensorShape({2}), {0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_HALF, TensorShape({1, 1, 6, 2}));
  test::FillValues<Eigen::half>(
      &expected, {-17.86, -22.00, -15.87, -20.59, -13.87, -19.18, -21.86,
                  -33.31, -23.85, -34.72, -25.85, -36.13});
  test::ExpectTensorNear<Eigen::half>(expected, *GetOutput(0), 0.1);
}

}  // namespace tensorflow
