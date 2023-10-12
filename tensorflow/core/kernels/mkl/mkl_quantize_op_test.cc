/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class MklQuantizeV2OpTest : public OpsTestBase {};

TEST_F(MklQuantizeV2OpTest, small_uint8) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "_MklQuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({8}),
                           {0.0, 1.0, 1.25, 1.75, 127.0, 255.0, 500.0, 2.0});
  // min_range = 0
  AddInputFromArray<float>(TensorShape({}), {0});
  // max_range = 255
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({8}));
  Tensor expected_min(allocator(), DT_FLOAT, TensorShape({}));
  Tensor expected_max(allocator(), DT_FLOAT, TensorShape({}));
  // Input element 0.0 should map to 0.
  // Input element 500.0 is quantized to 255 because max_range = 255.
  test::FillValues<quint8>(&expected, {0, 1, 1, 2, 127, 255, 255, 2});
  test::FillValues<float>(&expected_min, {0.0});
  test::FillValues<float>(&expected_max, {255.0});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  test::ExpectTensorEqual<float>(expected_min, *GetOutput(1));
  test::ExpectTensorEqual<float>(expected_max, *GetOutput(2));
}
TEST_F(MklQuantizeV2OpTest, small_int8) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "_MklQuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({8}), {0.0, -1.0, 1.25, -1.75, -24.5,
                                              -255.0, -80.315, 256.0});
  AddInputFromArray<float>(TensorShape({}), {-50.0});
  AddInputFromArray<float>(TensorShape({}), {127.0});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({8}));
  Tensor expected_min(allocator(), DT_FLOAT, TensorShape({}));
  Tensor expected_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<qint8>(&expected, {0, -1, 1, -2, -25, -128, -81, 127});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
  test::FillValues<float>(&expected_min, {-127.0});
  test::FillValues<float>(&expected_max, {127.0});
  test::ExpectTensorEqual<float>(expected_min, *GetOutput(1));
  test::ExpectTensorEqual<float>(expected_max, *GetOutput(2));
}

TEST_F(MklQuantizeV2OpTest, small_minfirst) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "_MklQuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({8}),
                           {1.0, 1.25, 1.75, 2, 3.15, 127.0, 255.0, 500.0});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({8}));
  test::FillValues<quint8>(&expected, {1, 1, 2, 2, 3, 127, 255, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_NEAR(255.0f, output_max, 1e-5f);
}

TEST_F(MklQuantizeV2OpTest, small_minfirst_uint) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "_MklQuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({8}),
                           {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  AddInputFromArray<float>(TensorShape({}), {0.1});
  AddInputFromArray<float>(TensorShape({}), {0.8});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({8}));
  test::FillValues<quint8>(&expected, {32, 64, 96, 128, 159, 191, 223, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_NEAR(0.8f, output_max, 1e-5f);
}

TEST_F(MklQuantizeV2OpTest, small_minfirst_int) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "_MklQuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({8}),
                           {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8});
  AddInputFromArray<float>(TensorShape({}), {-0.8});
  AddInputFromArray<float>(TensorShape({}), {-0.1});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({8}));
  test::FillValues<quint8>(&expected, {223, 191, 159, 128, 96, 64, 32, 0});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  EXPECT_NEAR(-0.8f, output_min, 1e-5f);
  EXPECT_NEAR(0.0f, output_max, 1e-5f);
}

}  // end namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
