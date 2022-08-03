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

class UniformQuantizedDotTest : public OpsTestBase {
 protected:
};

TEST_F(UniformQuantizedDotTest, HybridPerTensorQuantized) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedDotHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", -128)
                   .Attr("rhs_quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2}), {-32.2, -12.1, 10.7, 11.6});
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // rhs output scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  // output should be similar to
  // dot(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(&expected, {16.0, -72.6, -161.2, 25.0, 69.6, 114.2});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/0.1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedDotTest, HybridPerChannelQuantized) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedDotHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", -128)
                   .Attr("rhs_quantization_max_val", 127)
                   .Attr("rhs_quantization_axis", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2}), {-32.2, -12.1, 10.7, 11.6});
  // rhs (per-channel quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // rhs output scales and zero_points.
  AddInputFromArray<float>(TensorShape({3}), {2.0, 4.0, 2.0});
  AddInputFromArray<int32>(TensorShape({3}), {2, 4, 2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  // output should be similar to
  // dot(lhs, [per-channel dequantized rhs])
  test::FillValues<float>(&expected, {16.0, 209.2, -161.2, 25.0, -39.2, 114.2});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/0.1, /*rtol=*/0.01);
}

}  // namespace tensorflow
