/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <functional>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

absl::Status MakeZeroElementShape(int rank, TensorShape* shape) {
  std::vector<int64_t> dims(rank, 1);
  if (rank > 0) {
    dims[0] = 0;
  }
  return TensorShape::BuildTensorShape(dims, shape);
}

class ExpandDimsOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("expand", "ExpandDims")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ExpandDimsOpTest, ExpandingMaxRankSucceeds) {
  MakeOp();
  TensorShape input_shape;
  TF_ASSERT_OK(
      MakeZeroElementShape(TensorShape::MaxDimensions() - 1, &input_shape));
  AddInput<float>(input_shape, [](int) { return 0.0f; });
  AddInputFromList<int32_t>(TensorShape({}), {0});

  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->dims(), TensorShape::MaxDimensions());
  EXPECT_EQ(GetOutput(0)->dim_size(0), 1);
}

TEST_F(ExpandDimsOpTest, ExpandingBeyondMaxRankFails) {
  MakeOp();
  TensorShape input_shape;
  TF_ASSERT_OK(MakeZeroElementShape(TensorShape::MaxDimensions(), &input_shape));
  AddInput<float>(input_shape, [](int) { return 0.0f; });
  AddInputFromList<int32_t>(TensorShape({}), {0});

  EXPECT_THAT(RunOpKernel(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("maximum supported rank")));
}

static void BM_ExpandDims(::testing::benchmark::State& state) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({1, 1, 1, 1}));
  input.flat<int32_t>()(0) = 10;

  Tensor axis(DT_INT32, TensorShape({}));
  axis.flat<int32_t>()(0) = 2;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "ExpandDims")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, axis))
                  .Attr("T", DT_INT32)
                  .Attr("Tdim", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
}

BENCHMARK(BM_ExpandDims)->UseRealTime();

}  // namespace
}  // namespace tensorflow
