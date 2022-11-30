/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"

namespace tflite {
namespace shim {
namespace {

using ::tensorflow::DT_FLOAT;
using ::tensorflow::DT_INT32;
using ::tensorflow::DT_INT64;
using ::tensorflow::FakeInput;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::TensorShape;
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::ExpectTensorEqual;

class TmplOpTfTest : public ::tensorflow::OpsTestBase {};

TEST_F(TmplOpTfTest, float_int32) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("tmpl_op", "TemplatizedOperation")
                   .Attr("AType", DT_FLOAT)
                   .Attr("BType", DT_INT32)
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({}), {10.5});
  AddInputFromArray<int32_t>(TensorShape({}), {20});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<float>(*GetOutput(0),
                           AsTensor<float>({30.5}, /*shape=*/{}));
}

TEST_F(TmplOpTfTest, int32_int64) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("tmpl_op", "TemplatizedOperation")
                   .Attr("AType", DT_INT32)
                   .Attr("BType", DT_INT64)
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT64))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<int32_t>(TensorShape({}), {10});
  AddInputFromArray<int64_t>(TensorShape({}), {20});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.
  ExpectTensorEqual<float>(*GetOutput(0), AsTensor<float>({30}, /*shape=*/{}));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
