/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedRangeOpTest : public ::tensorflow::OpsTestBase {
 protected:
  // Indices of output tensors.
  static constexpr int kSplitsOutput = 0;
  static constexpr int kValuesOutput = 1;

  // Builds the tensorflow test graph for the RaggedRange op.
  template <typename T>
  void BuildRaggedRangeGraph() {
    const auto& dtype = DataTypeToEnum<T>::v();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedRange")
                     .Input(FakeInput(dtype))  // starts
                     .Input(FakeInput(dtype))  // limits
                     .Input(FakeInput(dtype))  // deltas
                     .Attr("T", dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RaggedRangeOpTest, IntValues) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<int>(TensorShape({4}), {8, 7, 8, 1});   // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 2, 4, 6], [5, 6], [], [5, 4, 3, 2]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 4, 6, 6, 10}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({0, 2, 4, 6, 5, 6, 5, 4, 3, 2}));
}

TEST_F(RaggedRangeOpTest, FloatValues) {
  BuildRaggedRangeGraph<float>();
  AddInputFromArray<float>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<float>(TensorShape({4}), {8, 7, 8, 1});   // limits
  AddInputFromArray<float>(TensorShape({4}), {2, 1, 1, -1});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 2, 4, 6], [5, 6], [], [5, 4, 3, 2]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 4, 6, 6, 10}));
  test::ExpectTensorNear<float>(
      *GetOutput(kValuesOutput),
      test::AsTensor<float>({0, 2, 4, 6, 5, 6, 5, 4, 3, 2}), 0.1);
}

TEST_F(RaggedRangeOpTest, RangeSizeOverflow) {
  BuildRaggedRangeGraph<float>();
  AddInputFromArray<float>(TensorShape({2}), {1.1, 0.1});    // starts
  AddInputFromArray<float>(TensorShape({2}), {10.0, 1e10});  // limits
  AddInputFromArray<float>(TensorShape({2}), {1, 1e-10});    // deltas

  EXPECT_EQ(absl::StrCat("Requires ((limit - start) / delta) <= ",
                         std::numeric_limits<int64_t>::max()),
            RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, RangeSizeOverflow2) {
  BuildRaggedRangeGraph<int64>();
  AddInputFromArray<int64>(TensorShape({}), {static_cast<int64_t>(5e18)});
  AddInputFromArray<int64>(TensorShape({}), {static_cast<int64_t>(-5e18)});
  AddInputFromArray<int64>(TensorShape({}), {-1});

  EXPECT_EQ(absl::StrCat("Requires ((limit - start) / delta) <= ",
                         std::numeric_limits<int64_t>::max()),
            RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, BroadcastDeltas) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({3}), {0, 5, 8});  // starts
  AddInputFromArray<int>(TensorShape({3}), {8, 7, 8});  // limits
  AddInputFromArray<int>(TensorShape({}), {1});         // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 1, 2, 3, 4, 5, 6, 7], [5, 6], []]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 8, 10, 10}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({0, 1, 2, 3, 4, 5, 6, 7, 5, 6}));
}

TEST_F(RaggedRangeOpTest, BroadcastLimitsAndDeltas) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({}), {0});         // starts
  AddInputFromArray<int>(TensorShape({3}), {3, 0, 2});  // limits
  AddInputFromArray<int>(TensorShape({}), {1});         // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 1, 2], [], [0, 1]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 3, 3, 5}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({0, 1, 2, 0, 1}));
}

TEST_F(RaggedRangeOpTest, BroadcastStartsAndLimits) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({}), {0});         // starts
  AddInputFromArray<int>(TensorShape({}), {12});        // limits
  AddInputFromArray<int>(TensorShape({3}), {3, 4, 5});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 3, 6, 9], [0, 4, 8], [0, 5, 10]]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 4, 7, 10}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({0, 3, 6, 9, 0, 4, 8, 0, 5, 10}));
}

TEST_F(RaggedRangeOpTest, AllScalarInputs) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({}), {0});  // starts
  AddInputFromArray<int>(TensorShape({}), {5});  // limits
  AddInputFromArray<int>(TensorShape({}), {1});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 1, 2, 3, 4]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 5}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({0, 1, 2, 3, 4}));
}

TEST_F(RaggedRangeOpTest, InvalidArgsStarts) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4, 1}), {0, 5, 8, 5});  // starts
  AddInputFromArray<int>(TensorShape({4}), {8, 7, 8, 1});     // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});    // deltas
  EXPECT_EQ("starts must be a scalar or vector", RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsLimits) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});     // starts
  AddInputFromArray<int>(TensorShape({4, 1}), {8, 7, 8, 1});  // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});    // deltas
  EXPECT_EQ("limits must be a scalar or vector", RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsDeltas) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});      // starts
  AddInputFromArray<int>(TensorShape({4}), {8, 7, 8, 1});      // limits
  AddInputFromArray<int>(TensorShape({4, 1}), {2, 1, 1, -1});  // deltas
  EXPECT_EQ("deltas must be a scalar or vector", RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsShapeMismatch) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<int>(TensorShape({3}), {7, 8, 1});      // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});  // deltas
  EXPECT_EQ("starts, limits, and deltas must have the same shape",
            RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsZeroDelta) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<int>(TensorShape({4}), {7, 8, 8, 1});   // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 0, -1});  // deltas
  EXPECT_EQ("Requires delta != 0", RunOpKernel().message());
}

TEST_F(RaggedRangeOpTest, EmptyRangePositiveDelta) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({2}), {0, 5});  // starts
  AddInputFromArray<int>(TensorShape({2}), {5, 0});  // limits
  AddInputFromArray<int>(TensorShape({}), {2});      // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 2, 4], []]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 3, 3}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({0, 2, 4}));
}

TEST_F(RaggedRangeOpTest, EmptyRangeNegativeDelta) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({2}), {0, 5});  // starts
  AddInputFromArray<int>(TensorShape({2}), {5, 0});  // limits
  AddInputFromArray<int>(TensorShape({}), {-2});     // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[], [5, 3, 1]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 0, 3}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({5, 3, 1}));
}

TEST_F(RaggedRangeOpTest, ShapeFn) {
  // RaggedRange(starts, limits, deltas) -> [splits, values]
  ShapeInferenceTestOp op("RaggedRange");
  INFER_OK(op, "?;?;?", "[?];[?]");
  INFER_OK(op, "[3];[3];[3]", "[4];[?]");
  INFER_OK(op, "[3];[3];[]", "[4];[?]");  // broadcast deltas
  INFER_OK(op, "[3];[];[3]", "[4];[?]");  // broadcast limits
  INFER_OK(op, "[];[3];[3]", "[4];[?]");  // broadcast starts
  INFER_OK(op, "[];[];[]", "[2];[?]");    // degenerate case: all scalar inputs
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[5,5];[5];[5]");
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[5];[5,5];[5]");
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[5];[5];[5,5]");
  INFER_ERROR("Dimensions must be equal, but are 4 and 3", op, "[3];[4];[3]");
}

}  // namespace
}  // namespace tensorflow
