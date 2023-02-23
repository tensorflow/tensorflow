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

class RaggedFillEmptyRowsOpTest : public ::tensorflow::OpsTestBase {
 protected:
  const int kValueRowidsOutput = 0;
  const int kValuesOutput = 1;
  const int kEmptyRowIndicatorOutput = 2;
  const int kReverseIndexMapOutput = 3;


  // Builds the tensorflow test graph for the RaggedFillEmptyRows op.
  template <typename T>
  void BuildFillEmptyRowsGraph() {
    const auto& dtype = DataTypeToEnum<T>::v();
    const auto& dtype_int64 = DataTypeToEnum<int64_t>::v();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedFillEmptyRows")
                     .Input(FakeInput(dtype_int64))           // value_rowids
                     .Input(FakeInput(dtype))                 // values
                     .Input(FakeInput(dtype_int64))           // nrows
                     .Input(FakeInput(dtype))                 // default value
                     .Attr("T", dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RaggedFillEmptyRowsOpTest, IntValues) {
  BuildFillEmptyRowsGraph<int>();
  AddInputFromArray<int64_t>(TensorShape({4}), {1, 2, 2, 5}); // value_rowids
  AddInputFromArray<int>(TensorShape({4}), {2, 4, 6, 8});     // values
  AddInputFromArray<int64_t>(TensorShape({}), {7});           // nrows
  AddInputFromArray<int>(TensorShape({}), {-1});              // default value
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kValueRowidsOutput),
      test::AsTensor<int64_t>({0, 1, 2, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({-1, 2, 4, 6, -1, -1, 8, -1}));
}

TEST_F(RaggedFillEmptyRowsOpTest, FloatValues) {
  BuildFillEmptyRowsGraph<float>();
  AddInputFromArray<int64_t>(TensorShape({4}), {1, 2, 2, 5});    // value_rowids
  AddInputFromArray<float>(TensorShape({4}), {2., 4., 6., 8.});  // values
  AddInputFromArray<int64_t>(TensorShape({}), {7});              // nrows
  AddInputFromArray<float>(TensorShape({}), {-1.});  // default value
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kValueRowidsOutput),
      test::AsTensor<int64_t>({0, 1, 2, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<float>(
      *GetOutput(kValuesOutput),
      test::AsTensor<float>({-1., 2., 4., 6., -1., -1., 8., -1.}));
}

}
}  // namespace tensorflow
