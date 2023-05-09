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

class UniformRequantizeOpTest : public OpsTestBase {
 protected:
};

TEST_F(UniformRequantizeOpTest, RequantizeInvalidQuantizationAxis) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_axis", -2)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  // input_quantization_axis < -1.
  EXPECT_TRUE(absl::IsInvalidArgument(InitOp()));

  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_axis", 0)
          .Attr("output_quantization_axis", 1)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  // input_quantization_axis and output_quantization_axis both >= 0 but
  // different.
  EXPECT_TRUE(absl::IsInvalidArgument(InitOp()));

  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_axis", 2)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<int32>(TensorShape({}), {0});

  // input_quantization_axis >= input tensor rank.
  EXPECT_TRUE(absl::IsInvalidArgument(RunOpKernel()));
}

TEST_F(UniformRequantizeOpTest, PerTensorToPerTensorReQuantize) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-28, -21, -1, 0, 4, 9});
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {-1});
  AddInputFromArray<float>(TensorShape({}), {0.125});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({2, 3}));
  // Input element -26 is requantized to -127 (not -128) because
  // output_quantization_min_val is -127.
  test::FillValues<qint8>(&expected, {-127, -100, -20, -16, 0, 20});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

TEST_F(UniformRequantizeOpTest, PerChannelToPerTensorReQuantize) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("input_quantization_axis", 0)
          .Attr("output_quantization_axis", -1)
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-28, -21, -1, -1, 3, 8});
  AddInputFromArray<float>(TensorShape({2}), {0.5, 0.6});
  AddInputFromArray<int32>(TensorShape({2}), {-1, -2});
  AddInputFromArray<float>(TensorShape({}), {0.125});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({2, 3}));
  // Input element -26 is requantized to -127 (not -128) because
  // output_quantization_min_val is -127.
  test::FillValues<qint8>(&expected, {-127, -100, -20, -15, 4, 28});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

TEST_F(UniformRequantizeOpTest, PerTensorToPerChannelReQuantize) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("input_quantization_axis", -1)
          .Attr("output_quantization_axis", 0)
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-28, -21, -1, -1, 3, 8});
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {-1});
  AddInputFromArray<float>(TensorShape({2}), {0.125, 0.3});
  AddInputFromArray<int32>(TensorShape({2}), {-20, -10});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({2, 3}));
  // Input element -26 is requantized to -127 (not -128) because
  // output_quantization_min_val is -127.
  test::FillValues<qint8>(&expected, {-127, -100, -20, -10, -3, 5});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

TEST_F(UniformRequantizeOpTest, PerChannelToPerChannelReQuantize) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformRequantize")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT32)
          .Attr("Tout", DT_QINT8)
          .Attr("input_quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("input_quantization_max_val", static_cast<int32_t>(2147483647))
          .Attr("input_quantization_axis", 0)
          .Attr("output_quantization_axis", 0)
          .Attr("output_quantization_min_val", -127)
          .Attr("output_quantization_max_val", 127)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-28, -21, -1, -1, 3, 8});
  AddInputFromArray<float>(TensorShape({2}), {0.5, 0.6});
  AddInputFromArray<int32>(TensorShape({2}), {-1, -2});
  AddInputFromArray<float>(TensorShape({2}), {0.125, 0.3});
  AddInputFromArray<int32>(TensorShape({2}), {-20, -10});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({2, 3}));
  // Input element -26 is requantized to -127 (not -128) because
  // output_quantization_min_val is -127.
  test::FillValues<qint8>(&expected, {-127, -100, -20, -8, 0, 10});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

}  // namespace tensorflow
