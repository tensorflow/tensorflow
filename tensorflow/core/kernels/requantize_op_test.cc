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

class RequantizeTest : public OpsTestBase {
 protected:
  void ConfigureRequantize() {
    TF_ASSERT_OK(NodeDefBuilder("requantize", "Requantize")
                     .Input(FakeInput(DT_QINT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("Tinput", DataTypeToEnum<qint32>::v())
                     .Attr("out_type", DataTypeToEnum<quint8>::v())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// Runs a manually generated array through the operator, and makes sure that the
// results match the expected hand-calculated values.
TEST_F(RequantizeTest, HandCraftedRequantize) {
  ConfigureRequantize();
  const int value_count = 3;

  // Requantize to -1 to 1.
  AddInputFromArray<qint32>(TensorShape({value_count}),
                            {-(1 << 23), 0, (1 << 23)});
  AddInputFromArray<float>(TensorShape({1}), {-256.0f});
  AddInputFromArray<float>(TensorShape({1}), {256.0f});
  AddInputFromArray<float>(TensorShape({1}), {-1.0f});
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({value_count}));
  test::FillValues<quint8>(&expected, {0, 128, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  test::ExpectTensorEqual<float>(test::AsScalar<float>(-1.0f), *GetOutput(1));
  test::ExpectTensorEqual<float>(test::AsScalar<float>(1.0f), *GetOutput(2));
}

TEST_F(RequantizeTest, InvalidOutputMin) {
  ConfigureRequantize();
  const int value_count = 3;

  AddInputFromArray<qint32>(TensorShape({value_count}),
                            {-(1 << 23), 0, (1 << 23)});
  AddInputFromArray<float>(TensorShape({1}), {-256.0f});
  AddInputFromArray<float>(TensorShape({1}), {256.0f});
  AddInputFromArray<float>(TensorShape({1}), {0.01f});
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  EXPECT_EQ("requested_output_min must be <= 0, but got 0.01",
            RunOpKernel().error_message());
}

TEST_F(RequantizeTest, InvalidOutputMax) {
  ConfigureRequantize();
  const int value_count = 3;

  AddInputFromArray<qint32>(TensorShape({value_count}),
                            {-(1 << 23), 0, (1 << 23)});
  AddInputFromArray<float>(TensorShape({1}), {-256.0f});
  AddInputFromArray<float>(TensorShape({1}), {256.0f});
  AddInputFromArray<float>(TensorShape({1}), {-10.0f});
  AddInputFromArray<float>(TensorShape({1}), {-11.0f});
  EXPECT_EQ(
      "requested_output_max must be >= requested_output_min, but got -11 and "
      "-10",
      RunOpKernel().error_message());
}

}  // end namespace tensorflow
