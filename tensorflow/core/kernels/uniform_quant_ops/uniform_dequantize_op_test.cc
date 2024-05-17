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

class UniformDequantizeOpTest : public OpsTestBase {
 protected:
};

TEST_F(UniformDequantizeOpTest, PerTensorDequantize) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformDequantize")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("quantization_axis", -1)
                   .Attr("quantization_min_val", -128)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint8>(TensorShape({2, 3}), {-128, -100, -20, -16, 0, 20});
  AddInputFromArray<float>(TensorShape({}), {0.25});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {-27.0, -20.0, 0.0, 1.0, 5.0, 10.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(UniformDequantizeOpTest, PerChannelDequantize) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformDequantize")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("quantization_axis", 1)
                   .Attr("quantization_min_val", -128)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint8>(TensorShape({2, 2, 3}),
                           {-128, -100, -20, -8, 0, 5, 10, 15, 20, 40, 50, 55});
  AddInputFromArray<float>(TensorShape({2}), {0.25, 0.5});
  AddInputFromArray<int32>(TensorShape({2}), {-20, -10});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 3}));
  test::FillValues<float>(&expected, {-27.0, -20.0, 0.0, 1.0, 5.0, 7.5, 7.5,
                                      8.75, 10.0, 25.0, 30.0, 32.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace tensorflow
