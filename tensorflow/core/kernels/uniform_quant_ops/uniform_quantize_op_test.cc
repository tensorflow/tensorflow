/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

class UniformQuantizeOpsTest : public OpsTestBase {
 protected:
};

TEST_F(UniformQuantizeOpsTest, QuantizeInvalidQuantizationAxis) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", -2)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  // quantization_axis < -1.
  EXPECT_TRUE(absl::IsInvalidArgument(InitOp()));

  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", 2)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<int32>(TensorShape({}), {0});

  // quantization_axis >= input tensor rank.
  EXPECT_TRUE(absl::IsInvalidArgument(RunOpKernel()));
}

TEST_F(UniformQuantizeOpsTest, PerTensorQuantize) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", -1)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-27.0, -20.0, 0.0, 1.0, 5.0, 10.0});
  AddInputFromArray<float>(TensorShape({}), {0.25});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({2, 3}));
  // Input element -27.0 is quantized to -127 (not -128) because
  // output_quantization_min_val is -127.
  test::FillValues<qint8>(&expected, {-127, -100, -20, -16, 0, 20});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizeOpsTest, PerChannelQuantize) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", 0)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-27.0, -20.0, 0.0, 1.0, 5.0, 10.0});
  AddInputFromArray<float>(TensorShape({2}), {0.25, 0.5});
  AddInputFromArray<int32>(TensorShape({2}), {-20, -10});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({2, 3}));
  // Input element -27.0 is quantized to -127 (not -128) because
  // output_quantization_min_val is -127.
  test::FillValues<qint8>(&expected, {-127, -100, -20, -8, 0, 10});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

}  // namespace tensorflow
