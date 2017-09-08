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

class QuantizeDownAndShrinkRangeTest : public OpsTestBase {
 protected:
};

// Runs a manually generated array through the operator, and makes sure that the
// results match the expected hand-calculated values.
TEST_F(QuantizeDownAndShrinkRangeTest, HandCrafted) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_down_and_shrink_range_op",
                              "QuantizeDownAndShrinkRange")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Tinput", DataTypeToEnum<qint32>::v())
                   .Attr("out_type", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // For this test we have an input that has the theoretical range of -256.0f to
  // +256.0f, but the actual values present only span -1.0f to 1.0f. We expect
  // the operator to take advantage of this, and rescale the output to fill up
  // the available range in the lower bit depth, and update to the true min and
  // max ranges.
  const int value_count = 3;
  AddInputFromArray<qint32>(TensorShape({value_count}),
                            {-(1 << 23), 0, (1 << 23)});
  AddInputFromArray<float>(TensorShape({1}), {-256.0f});
  AddInputFromArray<float>(TensorShape({1}), {256.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({value_count}));
  test::FillValues<quint8>(&expected, {0, 127, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  Tensor expected_min(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_min, {-1.0f});
  test::ExpectTensorEqual<float>(expected_min, *GetOutput(1));
  Tensor expected_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_max, {1.0f});
  test::ExpectTensorEqual<float>(expected_max, *GetOutput(2));
}

}  // end namespace tensorflow
