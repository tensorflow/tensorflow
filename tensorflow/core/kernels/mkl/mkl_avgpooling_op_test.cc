/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Regression tests for MklAvgPoolingGradOp input validation.
// These tests verify that malformed inputs produce clean InvalidArgument
// errors instead of crashing with CHECK failures (see #118354).

class MklAvgPoolingGradOpTest : public OpsTestBase {};

// ---------- 2D pooling gradient tests (_MklNativeAvgPoolGrad) ----------

// Test: scalar orig_input_shape should return InvalidArgument, not crash.
TEST_F(MklAvgPoolingGradOpTest, ScalarOrigInputShape_2D) {
  TF_ASSERT_OK(NodeDefBuilder("avg_pool_grad", "_MklNativeAvgPoolGrad")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("ksize", {1, 2, 2, 1})
                   .Attr("strides", {1, 1, 1, 1})
                   .Attr("padding", "VALID")
                   .Attr("data_format", "NHWC")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Scalar orig_input_shape (0D) — should fail because it's not 1D.
  AddInputFromArray<int32>(TensorShape({}), {0});
  // Scalar grad tensor (0D).
  AddInputFromArray<float>(TensorShape({}), {0.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s)) << s;
  EXPECT_NE(s.message().find("orig_input_shape must be a 1D tensor"),
            std::string::npos)
      << s;
}

// Test: empty orig_input_shape (1D with 0 elements) produces rank-0
// output_shape, which should fail the rank check.
TEST_F(MklAvgPoolingGradOpTest, EmptyOrigInputShape_2D) {
  TF_ASSERT_OK(NodeDefBuilder("avg_pool_grad", "_MklNativeAvgPoolGrad")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("ksize", {1, 2, 2, 1})
                   .Attr("strides", {1, 1, 1, 1})
                   .Attr("padding", "VALID")
                   .Attr("data_format", "NHWC")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // 1D tensor with 0 elements — will produce a 0D output_shape.
  AddInputFromArray<int32>(TensorShape({0}), {});
  // Scalar grad tensor.
  AddInputFromArray<float>(TensorShape({}), {0.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s)) << s;
}

// Test: valid orig_input_shape but wrong-rank grad tensor.
TEST_F(MklAvgPoolingGradOpTest, WrongRankGrad_2D) {
  TF_ASSERT_OK(NodeDefBuilder("avg_pool_grad", "_MklNativeAvgPoolGrad")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("ksize", {1, 2, 2, 1})
                   .Attr("strides", {1, 1, 1, 1})
                   .Attr("padding", "VALID")
                   .Attr("data_format", "NHWC")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Valid 4-element orig_input_shape representing a 4D tensor.
  AddInputFromArray<int32>(TensorShape({4}), {1, 4, 4, 1});
  // 1D grad tensor (wrong rank — should be 4D).
  AddInputFromArray<float>(TensorShape({16}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s)) << s;
  EXPECT_NE(s.message().find("Expected grad tensor to be 4D"),
            std::string::npos)
      << s;
}

// ---------- 3D pooling gradient tests (_MklNativeAvgPool3DGrad) ----------

// Test: scalar orig_input_shape should return InvalidArgument, not crash.
TEST_F(MklAvgPoolingGradOpTest, ScalarOrigInputShape_3D) {
  TF_ASSERT_OK(NodeDefBuilder("avg_pool_3d_grad", "_MklNativeAvgPool3DGrad")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("ksize", {1, 2, 2, 2, 1})
                   .Attr("strides", {1, 1, 1, 1, 1})
                   .Attr("padding", "VALID")
                   .Attr("data_format", "NDHWC")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Scalar orig_input_shape (0D).
  AddInputFromArray<int32>(TensorShape({}), {0});
  // Scalar grad tensor (0D).
  AddInputFromArray<float>(TensorShape({}), {0.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s)) << s;
  EXPECT_NE(s.message().find("orig_input_shape must be a 1D tensor"),
            std::string::npos)
      << s;
}

// Test: valid orig_input_shape but wrong-rank grad tensor for 3D pooling.
TEST_F(MklAvgPoolingGradOpTest, WrongRankGrad_3D) {
  TF_ASSERT_OK(NodeDefBuilder("avg_pool_3d_grad", "_MklNativeAvgPool3DGrad")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("ksize", {1, 2, 2, 2, 1})
                   .Attr("strides", {1, 1, 1, 1, 1})
                   .Attr("padding", "VALID")
                   .Attr("data_format", "NDHWC")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Valid 5-element orig_input_shape representing a 5D tensor.
  AddInputFromArray<int32>(TensorShape({5}), {1, 4, 4, 4, 1});
  // Scalar grad tensor (wrong rank — should be 5D).
  AddInputFromArray<float>(TensorShape({}), {0.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s)) << s;
  EXPECT_NE(s.message().find("Expected grad tensor to be 5D"),
            std::string::npos)
      << s;
}

}  // namespace tensorflow

#endif  // INTEL_MKL
