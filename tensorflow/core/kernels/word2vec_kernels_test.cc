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

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

class NegTrainOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("neg_train", "NegTrain")
                     .Input(FakeInput(DT_FLOAT_REF))
                     .Input(FakeInput(DT_FLOAT_REF))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("num_negative_samples", 1)
                     .Attr("vocab_count", {1})
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOpWithGraphVersion(18));
  }
};

TEST_F(NegTrainOpTest, OutOfBoundsExample) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 1}), {1.0});  // w_in
  AddInputFromArray<float>(TensorShape({1, 1}), {1.0});  // w_out
  AddInputFromArray<int32>(TensorShape({1}), {5});       // example out of bounds
  AddInputFromArray<int32>(TensorShape({1}), {0});       // label
  AddInputFromArray<float>(TensorShape({}), {0.1});      // learning_rate

  absl::Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s));
  EXPECT_TRUE(absl::StrContains(s.message(), "examples value 5 out of range"));
}

TEST_F(NegTrainOpTest, OutOfBoundsLabel) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 1}), {1.0});  // w_in
  AddInputFromArray<float>(TensorShape({1, 1}), {1.0});  // w_out
  AddInputFromArray<int32>(TensorShape({1}), {0});       // example
  AddInputFromArray<int32>(TensorShape({1}), {5});       // label out of bounds
  AddInputFromArray<float>(TensorShape({}), {0.1});      // learning_rate

  absl::Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s));
  EXPECT_TRUE(absl::StrContains(s.message(), "labels value 5 out of range"));
}

TEST_F(NegTrainOpTest, NegativeExampleIndex) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 1}), {1.0});  // w_in
  AddInputFromArray<float>(TensorShape({1, 1}), {1.0});  // w_out
  AddInputFromArray<int32>(TensorShape({1}), {-1});      // negative example
  AddInputFromArray<int32>(TensorShape({1}), {0});       // label
  AddInputFromArray<float>(TensorShape({}), {0.1});      // learning_rate

  absl::Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(s));
  EXPECT_TRUE(absl::StrContains(s.message(), "examples value -1 out of range"));
}

}  // namespace
}  // namespace tensorflow
