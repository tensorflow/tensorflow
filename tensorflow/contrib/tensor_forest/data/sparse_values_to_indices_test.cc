// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
// =============================================================================
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

class SparseValuesToIndicesOpTest : public OpsTestBase {
 protected:
  void MakeSparseValuesToIndicesOp(int32 offset_bits) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "SparseValuesToIndices")
                     .Attr("offset_bits", offset_bits)
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseValuesToIndicesOpTest, Basic) {
  const int offset_bits = 3;
  const int num_values = 3;
  MakeSparseValuesToIndicesOp(offset_bits);

  AddInputFromArray<int64>(TensorShape({num_values, 2}), {0, 0, 0, 1, 1, 0});
  AddInputFromArray<string>(TensorShape({num_values}),
                            {"abcdef", "ghijkl", "mnopqr"});
  AddInputFromArray<int64>(TensorShape({1}), {0});

  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_indices_tensor = GetOutput(0);
  ASSERT_EQ(2, out_indices_tensor->dims());
  EXPECT_EQ(num_values, out_indices_tensor->dim_size(0));
  EXPECT_EQ(2, out_indices_tensor->dim_size(1));

  Tensor* out_values_tensor = GetOutput(1);
  ASSERT_EQ(1, out_values_tensor->dims());
  EXPECT_EQ(num_values, out_values_tensor->dim_size(0));

  // Check the results using Eigen::Tensor objects.
  auto out_indices = out_indices_tensor->matrix<int64>();
  // example numbers
  EXPECT_EQ(0, out_indices(0, 0));
  EXPECT_EQ(0, out_indices(1, 0));
  EXPECT_EQ(1, out_indices(2, 0));

  // make sure hashed indices are confined to 26 - offset_bits bits.
  const int64 mask = 0xFFFFFFFFFF800000;

  EXPECT_GT(out_indices(0, 1), 0);
  EXPECT_EQ(0, mask & out_indices(0, 1));

  EXPECT_GT(out_indices(1, 1), 0);
  EXPECT_EQ(0, mask & out_indices(1, 1));

  EXPECT_GT(out_indices(2, 1), 0);
  EXPECT_EQ(0, mask & out_indices(2, 1));

  // Ones are just indicators that the feature is "there".
  auto out_values = out_values_tensor->flat<float>();
  EXPECT_EQ(1.0, out_values(0));
  EXPECT_EQ(1.0, out_values(1));
  EXPECT_EQ(1.0, out_values(2));
}

}  // namespace
}  // namespace tensorflow
