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

using ::tensorflow::DT_STRING;
using ::tensorflow::FakeInput;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::ExpectTensorEqual;

class SimpleOpTfTest : public ::tensorflow::OpsTestBase {};

TEST_F(SimpleOpTfTest, OutputSize5) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("simple_op", "SimpleOperation")
                   .Attr("output2_size", 5)
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {"abc"});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<int>(*GetOutput(0),
                         AsTensor<int>({0, 1, 2, 3, 4}, /*shape=*/{5}));
  ExpectTensorEqual<float>(
      *GetOutput(1), AsTensor<float>({0, 0.5, 1., 1.5, 2.}, /*shape=*/{5}));
  ExpectTensorEqual<int>(*GetOutput(2),
                         AsTensor<int>({0, 1, 2}, /*shape=*/{3}));
}

TEST_F(SimpleOpTfTest, OutputSize3) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("simple_op", "SimpleOperation")
                   .Attr("output2_size", 3)
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {"abc"});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<int>(*GetOutput(0),
                         AsTensor<int>({0, 1, 2, 3, 4}, /*shape=*/{5}));
  ExpectTensorEqual<float>(*GetOutput(1),
                           AsTensor<float>({0, 0.5, 1.}, /*shape=*/{3}));
  ExpectTensorEqual<int>(*GetOutput(2),
                         AsTensor<int>({0, 1, 2}, /*shape=*/{3}));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
