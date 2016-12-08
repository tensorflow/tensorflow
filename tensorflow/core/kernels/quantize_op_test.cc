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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class QuantizedOpTest : public OpsTestBase {
 protected:
};

TEST_F(QuantizedOpTest, QuantizeV2) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}),
                           {1.0, 1.25, 1.75, 127.0, 255.0, 500.0});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({6}));
  test::FillValues<quint8>(&expected, {1, 1, 2, 127, 255, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

TEST_F(QuantizedOpTest, QuantizeV2Ports) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}),
                           {1.0, 1.25, 1.75, 127.0, 255.0, 500.0});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({6}));
  test::FillValues<quint8>(&expected, {1, 1, 2, 127, 255, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_NEAR(255.0f, output_max, 1e-5f);
}

TEST_F(QuantizedOpTest, QuantizeV2EqualRange) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({6}));
  test::FillValues<quint8>(&expected, {0, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_LT(0.0f, output_max);
}

TEST_F(QuantizedOpTest, Dequantize) {
  TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "Dequantize")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<quint8>(TensorShape({6}), {1, 2, 4, 8, 16, 255});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected, {1.0, 2.0, 4.0, 8.0, 16.0, 255.0});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.5);
}

}  // end namespace tensorflow
