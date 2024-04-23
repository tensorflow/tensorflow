/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RangeOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType input_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "Range")
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(input_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class LinSpaceOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType input_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "LinSpace")
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RangeOpTest, Simple_D32) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<int32>(TensorShape({}), {10});
  AddInputFromArray<int32>(TensorShape({}), {2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_INT32, TensorShape({5}));
  test::FillValues<int32>(&expected, {0, 2, 4, 6, 8});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(RangeOpTest, Simple_Float) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {0.3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0.5, 0.8, 1.1, 1.4, 1.7});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RangeOpTest, Large_Double) {
  MakeOp(DT_DOUBLE);

  // Feed and run
  AddInputFromArray<double>(TensorShape({}), {0.0});
  AddInputFromArray<double>(TensorShape({}), {10000});
  AddInputFromArray<double>(TensorShape({}), {0.5});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_DOUBLE, TensorShape({20000}));
  std::vector<double> result;
  for (int32_t i = 0; i < 20000; ++i) result.push_back(i * 0.5);
  test::FillValues<double>(&expected, absl::Span<const double>(result));
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Simple_D32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {3.0});
  AddInputFromArray<float>(TensorShape({}), {7.0});
  AddInputFromArray<int32>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {3.0, 5.0, 7.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Exact_Endpoints) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run. The particular values 0., 1., and 42 are chosen to test that
  // the last value is not calculated via an intermediate delta as (1./41)*41,
  // because for IEEE 32-bit floats that returns 0.99999994 != 1.0.
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<int32>(TensorShape({}), {42});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor output = *GetOutput(0);
  float expected_start = 0.0;
  float start = output.flat<float>()(0);
  EXPECT_EQ(expected_start, start) << expected_start << " vs. " << start;
  float expected_stop = 1.0;
  float stop = output.flat<float>()(output.NumElements() - 1);
  EXPECT_EQ(expected_stop, stop) << expected_stop << " vs. " << stop;
}

TEST_F(LinSpaceOpTest, Single_D64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {9.0});
  AddInputFromArray<float>(TensorShape({}), {100.0});
  AddInputFromArray<int64_t>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&expected, {9.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Simple_Double) {
  MakeOp(DT_DOUBLE, DT_INT32);

  // Feed and run
  AddInputFromArray<double>(TensorShape({}), {5.0});
  AddInputFromArray<double>(TensorShape({}), {6.0});
  AddInputFromArray<int32>(TensorShape({}), {6});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_DOUBLE, TensorShape({6}));
  test::FillValues<double>(&expected, {5.0, 5.2, 5.4, 5.6, 5.8, 6.0});
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow
