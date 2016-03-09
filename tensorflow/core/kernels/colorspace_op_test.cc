/* Copyright 2015 Google Inc. All Rights Reserved.

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

class RGBToHSVOpTest : public OpsTestBase {
 protected:
  RGBToHSVOpTest() {
    TF_EXPECT_OK(NodeDefBuilder("rgb_to_hsv_op", "RGBToHSV")
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(RGBToHSVOpTest, CheckBlack) {
  // Black pixel should map to hsv = [0,0,0]
  AddInputFromArray<float>(TensorShape({3}), {0, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {0.0, 0.0, 0.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RGBToHSVOpTest, CheckGray) {
  // Gray pixel should have hue = saturation = 0.0, value = r/255
  AddInputFromArray<float>(TensorShape({3}), {.5, .5, .5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {0.0, 0.0, .5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RGBToHSVOpTest, CheckWhite) {
  // Gray pixel should have hue = saturation = 0.0, value = 1.0
  AddInputFromArray<float>(TensorShape({3}), {1, 1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {0.0, 0.0, 1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RGBToHSVOpTest, CheckRedMax) {
  // Test case where red channel dominates
  AddInputFromArray<float>(TensorShape({3}), {.8, .4, .2});
  TF_ASSERT_OK(RunOpKernel());

  float expected_h = 1. / 6. * .2 / .6;
  float expected_s = .6 / .8;
  float expected_v = .8 / 1.;

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {expected_h, expected_s, expected_v});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

TEST_F(RGBToHSVOpTest, CheckGreenMax) {
  // Test case where green channel dominates
  AddInputFromArray<float>(TensorShape({3}), {.2, .8, .4});
  TF_ASSERT_OK(RunOpKernel());

  float expected_h = 1. / 6. * (2.0 + (.2 / .6));
  float expected_s = .6 / .8;
  float expected_v = .8 / 1.;

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {expected_h, expected_s, expected_v});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

TEST_F(RGBToHSVOpTest, CheckBlueMax) {
  // Test case where blue channel dominates
  AddInputFromArray<float>(TensorShape({3}), {.4, .2, .8});
  TF_ASSERT_OK(RunOpKernel());

  float expected_h = 1. / 6. * (4.0 + (.2 / .6));
  float expected_s = .6 / .8;
  float expected_v = .8 / 1.;

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {expected_h, expected_s, expected_v});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

TEST_F(RGBToHSVOpTest, CheckNegativeDifference) {
  AddInputFromArray<float>(TensorShape({3}), {0, .1, .2});
  TF_ASSERT_OK(RunOpKernel());

  float expected_h = 1. / 6. * (4.0 + (-.1 / .2));
  float expected_s = .2 / .2;
  float expected_v = .2 / 1.;

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {expected_h, expected_s, expected_v});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

class HSVToRGBOpTest : public OpsTestBase {
 protected:
  HSVToRGBOpTest() {
    TF_EXPECT_OK(NodeDefBuilder("hsv_to_rgb_op", "HSVToRGB")
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(HSVToRGBOpTest, CheckBlack) {
  // Black pixel should map to rgb = [0,0,0]
  AddInputFromArray<float>(TensorShape({3}), {0.0, 0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {0, 0, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(HSVToRGBOpTest, CheckGray) {
  // Gray pixel should have hue = saturation = 0.0, value = r/255
  AddInputFromArray<float>(TensorShape({3}), {0.0, 0.0, .5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {.5, .5, .5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(HSVToRGBOpTest, CheckWhite) {
  // Gray pixel should have hue = saturation = 0.0, value = 1.0
  AddInputFromArray<float>(TensorShape({3}), {0.0, 0.0, 1.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {1, 1, 1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(HSVToRGBOpTest, CheckRedMax) {
  // Test case where red channel dominates
  float expected_h = 1. / 6. * .2 / .6;
  float expected_s = .6 / .8;
  float expected_v = .8 / 1.;

  AddInputFromArray<float>(TensorShape({3}),
                           {expected_h, expected_s, expected_v});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {.8, .4, .2});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

TEST_F(HSVToRGBOpTest, CheckGreenMax) {
  // Test case where green channel dominates
  float expected_h = 1. / 6. * (2.0 + (.2 / .6));
  float expected_s = .6 / .8;
  float expected_v = .8 / 1.;

  AddInputFromArray<float>(TensorShape({3}),
                           {expected_h, expected_s, expected_v});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {.2, .8, .4});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}

TEST_F(HSVToRGBOpTest, CheckBlueMax) {
  // Test case where blue channel dominates
  float expected_h = 1. / 6. * (4.0 + (.2 / .6));
  float expected_s = .6 / .8;
  float expected_v = .8 / 1.0;

  AddInputFromArray<float>(TensorShape({3}),
                           {expected_h, expected_s, expected_v});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {.4, .2, .8});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-6);
}
}  // namespace tensorflow
