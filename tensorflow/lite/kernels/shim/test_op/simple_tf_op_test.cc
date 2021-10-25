/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/test_op/simple_tf_op.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/tstring.h"

namespace tflite {
namespace shim {
namespace {

using ::tensorflow::DT_INT64;
using ::tensorflow::DT_STRING;
using ::tensorflow::FakeInput;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::ExpectTensorEqual;

class SimpleOpTfTest : public ::tensorflow::OpsTestBase {};

TEST_F(SimpleOpTfTest, Output1Size_5_N_2) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("simple_op", "SimpleOperation")
                   .Attr("output1_size", 5)
                   .Attr("N", 2)
                   .Input(FakeInput(DT_STRING))
                   .Input(FakeInput(2, DT_INT64))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {"abc"});
  AddInputFromArray<int64_t>(TensorShape({}), {123});
  AddInputFromArray<int64_t>(TensorShape({2}), {456, 789});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<int>(*GetOutput(0),
                         AsTensor<int>({0, 1, 2, 3, 4}, /*shape=*/{5}));
  ExpectTensorEqual<float>(
      *GetOutput(1), AsTensor<float>({0, 0.5, 1., 1.5, 2.}, /*shape=*/{5}));
  ExpectTensorEqual<tstring>(*GetOutput(2),
                             AsTensor<tstring>({"0", "1", "2"}, /*shape=*/{3}));
  ExpectTensorEqual<int64_t>(*GetOutput(3),
                             AsTensor<int64_t>({124}, /*shape=*/{}));
  ExpectTensorEqual<int64_t>(*GetOutput(4),
                             AsTensor<int64_t>({457, 790}, /*shape=*/{2}));
}

TEST_F(SimpleOpTfTest, Output1Size_3_N_0) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("simple_op", "SimpleOperation")
                   .Attr("output1_size", 3)
                   .Attr("N", 0)
                   .Input(FakeInput(DT_STRING))
                   .Input(FakeInput(0, DT_INT64))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {"abc"});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<int>(*GetOutput(0),
                         AsTensor<int>({0, 1, 2, 3, 4}, /*shape=*/{5}));
  ExpectTensorEqual<float>(*GetOutput(1),
                           AsTensor<float>({0, 0.5, 1.}, /*shape=*/{3}));
  ExpectTensorEqual<tstring>(*GetOutput(2),
                             AsTensor<tstring>({"0", "1", "2"}, /*shape=*/{3}));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
