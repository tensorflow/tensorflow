// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_text/core/kernels/text_kernels_test_util.h"

namespace tensorflow {
namespace text {

using tensorflow::FakeInput;
using tensorflow::NodeDefBuilder;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::text_kernels_test_util::VectorEq;

class UnicodeScriptTokenizeWithOffsetsKernelTest
    : public tensorflow::OpsTestBase {
 public:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "UnicodeScriptTokenizeWithOffsets")
                 .Input(FakeInput())
                 .Input(FakeInput())
                 .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(UnicodeScriptTokenizeWithOffsetsKernelTest, Test) {
  MakeOp();
  AddInputFromArray<int32>(TensorShape({6}), {111, 112, 32, 116, 117, 118});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<int32> expected_values({111, 112, 116, 117, 118});
  std::vector<int64> expected_values_inner_splits({0, 2, 3, 5});
  std::vector<int64> expected_offset_starts({0, 3, 0});
  std::vector<int64> expected_offset_limits({2, 4, 2});
  std::vector<int64> output_outer_splits({0, 2, 3});
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_values_inner_splits));
  EXPECT_THAT(*GetOutput(2), VectorEq(expected_offset_starts));
  EXPECT_THAT(*GetOutput(3), VectorEq(expected_offset_limits));
  EXPECT_THAT(*GetOutput(4), VectorEq(output_outer_splits));
}

}  // namespace text
}  // namespace tensorflow
