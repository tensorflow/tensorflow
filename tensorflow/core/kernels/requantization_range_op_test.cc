/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Declared here so we don't need to include it in a public header.
void CalculateUsedRange(const Tensor& input, qint32* actual_min_quantized,
                        qint32* actual_max_quantized);

class RequantizationRangeTest : public OpsTestBase {
 protected:
};

// Runs a manually generated array through the operator, and makes sure that the
// results match the expected hand-calculated values.
TEST_F(RequantizationRangeTest, HandCrafted) {
  TF_ASSERT_OK(NodeDefBuilder("requantization_range", "RequantizationRange")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Tinput", DataTypeToEnum<qint32>::v())
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
  Tensor expected_min(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_min, {-1.0f});
  test::ExpectTensorEqual<float>(expected_min, *GetOutput(0));
  Tensor expected_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_max, {1.0f});
  test::ExpectTensorEqual<float>(expected_max, *GetOutput(1));
}

static void BM_RequantizationRange(::testing::benchmark::State& state) {
  const int size = state.range(0);

  Tensor quantized_tensor(DT_QINT32, TensorShape({1, size}));
  test::FillFn<qint32>(&quantized_tensor, [](int n) { return qint32(n); });

  qint32 actual_min;
  qint32 actual_max;
  for (auto s : state) {
    CalculateUsedRange(quantized_tensor, &actual_min, &actual_max);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * 4);
}

BENCHMARK(BM_RequantizationRange)
    ->UseRealTime()
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Arg(100000000);

}  // end namespace tensorflow
