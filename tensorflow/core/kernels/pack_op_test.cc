/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

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

class PackOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("pack", "Pack")
                     .Input(FakeInput(1, DT_FLOAT))
                     .Attr("N", 1)
                     .Attr("T", DT_FLOAT)
                     .Attr("axis", 0)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(PackOpTest, PackingMaxRankSucceeds) {
  MakeOp();
  TensorShape input_shape;
  TF_ASSERT_OK(
      MakeZeroElementShape(TensorShape::MaxDimensions() - 1, &input_shape));
  AddInput<float>(input_shape, [](int) { return 0.0f; });

  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->dims(), TensorShape::MaxDimensions());
  EXPECT_EQ(GetOutput(0)->dim_size(0), 1);
}

TEST_F(PackOpTest, PackingBeyondMaxRankFails) {
  MakeOp();
  TensorShape input_shape;
  TF_ASSERT_OK(MakeZeroElementShape(TensorShape::MaxDimensions(), &input_shape));
  AddInput<float>(input_shape, [](int) { return 0.0f; });

  EXPECT_THAT(RunOpKernel(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("maximum supported rank")));
}

}  // namespace
}  // namespace tensorflow
