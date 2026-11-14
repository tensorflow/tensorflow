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

#include <cstdint>

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

class Conv3DBackpropInputV2OpTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<const char*> {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(
        NodeDefBuilder("conv3d_backprop_input", "Conv3DBackpropInputV2")
            .Input(FakeInput(DT_INT32))
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_FLOAT))
            .Attr("T", DT_FLOAT)
            .Attr("strides", {1, 1, 1, 1, 1})
            .Attr("padding", "VALID")
            .Attr("data_format", "NDHWC")
            .Attr("dilations", {1, 1, 1, 1, 1})
            .Attr("_kernel", GetParam())
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

INSTANTIATE_TEST_SUITE_P(Kernels, Conv3DBackpropInputV2OpTest,
                         ::testing::Values("custom", "eigen_tensor"));

// A valid single-element input.
TEST_P(Conv3DBackpropInputV2OpTest, SingleElement) {
  MakeOp();
  AddInputFromArray<int32_t>(TensorShape({5}), {1, 1, 1, 1, 1});
  AddInputFromArray<float>(TensorShape({1, 1, 1, 1, 1}), {2.0f});
  AddInputFromArray<float>(TensorShape({1, 1, 1, 1, 1}), {3.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_FLOAT, TensorShape({1, 1, 1, 1, 1}));
  test::FillValues<float>(&expected, {6.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(Conv3DBackpropInputV2OpTest, AllTensorsEmpty) {
  MakeOp();
  // Zero batch and zero output channels: input, filter and out_backprop all
  // have zero elements, and the kernel returns an empty output tensor.
  AddInputFromArray<int32_t>(TensorShape({5}), {0, 1, 1, 1, 1});
  AddInputFromArray<float>(TensorShape({1, 1, 1, 1, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 1, 1, 1, 0}), {});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), TensorShape({0, 1, 1, 1, 1}));
}

TEST_P(Conv3DBackpropInputV2OpTest, EmptyOutBackpropZeroesInputGrad) {
  MakeOp();
  AddInputFromArray<int32_t>(TensorShape({5}), {1, 2, 2, 2, 1});
  AddInputFromArray<float>(TensorShape({1, 1, 1, 1, 0}), {});
  AddInputFromArray<float>(TensorShape({1, 2, 2, 2, 0}), {});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_FLOAT, TensorShape({1, 2, 2, 2, 1}));
  test::FillValues<float>(
      &expected, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow
