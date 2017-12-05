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
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"


namespace tensorflow {


namespace {

class SparseTileOpTest : public OpsTestBase {
  protected:
    template <typename T>
    void MakeOp() {
      DataType value_type = tensorflow::DataTypeToEnum<T>::value;
      TF_ASSERT_OK(
          NodeDefBuilder("sparse_tile_like", "SparseTileLike")
              .Input(FakeInput(DT_INT64))
              .Input(FakeInput(value_type))
              .Input(FakeInput(DT_INT64))
              .Input(FakeInput(DT_INT64))
              .Input(FakeInput(value_type))
              .Input(FakeInput(DT_INT64))
              .Input(FakeInput(DT_INT32))
              .Attr("T", value_type)
              .Finalize(node_def()));
      TF_ASSERT_OK(InitOp());
    }
};


TEST_F(SparseTileOpTest, Basic1) {
  MakeOp<float>();

  // a
  const auto a_indices_shape = TensorShape({2, 1});
  std::initializer_list<int64> a_in{ 0, 2 };
  const gtl::ArraySlice<int64> a_indices(a_in);
  std::initializer_list<int64> a_sh{ 3 };
  const gtl::ArraySlice<int64> a_shape(a_sh);
  // b
  const auto b_indices_shape = TensorShape({4, 2});
  std::initializer_list<int64> b_in{ 0, 1, 1, 0, 2, 0, 2, 1 };
  const gtl::ArraySlice<int64> b_indices(b_in);
  std::initializer_list<int64> b_sh{ 3, 2 };
  const gtl::ArraySlice<int64> b_shape(b_sh);

  AddInputFromArray<int64>(a_indices_shape, a_indices);
  AddInputFromArray<float>(TensorShape({2}), {7, 8});
  AddInputFromArray<int64>(TensorShape({1}), a_shape);
  AddInputFromArray<int64>(b_indices_shape, b_indices);
  AddInputFromArray<float>(TensorShape({4}), {2, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), b_shape);
  AddInputFromArray<int32>(TensorShape({1}), {1});  // reduction axes

  TF_ASSERT_OK(RunOpKernel());
  LOG(INFO) << "File test";

  Tensor expected_indices(allocator(), DT_INT64, TensorShape({3, 2}));
  std::initializer_list<int64> b_in_exp{ 0, 1, 2, 0, 2, 1 };
  test::FillValues<int64>(&expected_indices, b_in_exp);
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected_values, {7, 8, 8});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, b_shape);
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(SparseTileOpTest, Basic2) {
  MakeOp<float>();

  // a
  const auto a_indices_shape = TensorShape({4, 2});
  std::initializer_list<int64> a_in{ 0, 0, 1, 1, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> a_indices(a_in);
  std::initializer_list<int64> a_sh{ 4, 2};
  const gtl::ArraySlice<int64> a_shape(a_sh);
  // b
  const auto b_indices_shape = TensorShape({7, 3});
  std::initializer_list<int64> b_in{ 0, 0, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 0, 2, 1, 2, 0, 1, 2 };
  const gtl::ArraySlice<int64> b_indices(b_in);
  std::initializer_list<int64> b_sh{ 4, 2, 3 };
  const gtl::ArraySlice<int64> b_shape(b_sh);

  AddInputFromArray<int64>(a_indices_shape, a_indices);
  AddInputFromArray<float>(TensorShape({4}), {7, 8, 9, 6});
  AddInputFromArray<int64>(TensorShape({2}), a_shape);
  AddInputFromArray<int64>(b_indices_shape, b_indices);
  AddInputFromArray<float>(TensorShape({7}), {1, 2, 1, 2, 1, 2, 1});
  AddInputFromArray<int64>(TensorShape({3}), b_shape);
  AddInputFromArray<int32>(TensorShape({1}), {2});  // reduction axes

  TF_ASSERT_OK(RunOpKernel());
  //LOG(INFO) << "File test" << GetOutput(0)->matrix<int64>();

  Tensor expected_indices(allocator(), DT_INT64, TensorShape({6, 3}));
  std::initializer_list<int64> b_in_exp{ 0, 0, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 0, 2, 1, 2 };
  test::FillValues<int64>(&expected_indices, b_in_exp);
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected_values, {7, 7, 8, 8, 9, 6});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {3});
  test::FillValues<int64>(&expected_shape, b_shape);
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}



}}
