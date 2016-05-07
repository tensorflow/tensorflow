/* Copyright 2016 Google Inc. All Rights Reserved.

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

static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
  EXPECT_TRUE(StringPiece(s).contains(expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

class SparseDenseCDivTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("cdiv", "SparseDenseCwiseDiv")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class SparseDenseCMulTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("cmul", "SparseDenseCwiseMul")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseDenseCDivTest, DoNotBroadcastSparse_FewerDims) {
  MakeOp<float>();
  // [1] op [2, 1]
  AddInputFromArray<int64>(TensorShape({1, 1}), {0});       // indices
  AddInputFromArray<float>(TensorShape({1}), {1618});       // values
  AddInputFromArray<int64>(TensorShape({1}), {1});          // shape
  AddInputFromArray<float>(TensorShape({2, 1}), {17, 19});  // dense

  ExpectHasSubstr(RunOpKernel().ToString(), "broadcasts dense to sparse only");
}

TEST_F(SparseDenseCDivTest, DoNotBroadcastSparse_SameDims) {
  MakeOp<float>();
  // [1, 1] op [2, 1]
  AddInputFromArray<int64>(TensorShape({1, 2}), {0, 0});
  AddInputFromArray<float>(TensorShape({1}), {1618});
  AddInputFromArray<int64>(TensorShape({2}), {1, 1});
  AddInputFromArray<float>(TensorShape({2, 1}), {17, 19});

  ExpectHasSubstr(RunOpKernel().ToString(), "broadcasts dense to sparse only");
}

TEST_F(SparseDenseCDivTest, SameShape) {
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: same shape, all 1's]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  // Tensor dense(DT_FLOAT, TensorShape({3, 1}));
  Tensor dense(DT_FLOAT, TensorShape(shape));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape(shape), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCDivTest, BroadcastDenseSameDims) {
  // No broadcast.
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: shape [3,1], all 1's]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  Tensor dense(DT_FLOAT, TensorShape({3, 1}));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({3, 1}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCDivTest, BroadcastDenseFewerDims) {
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: shape [2]]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  Tensor dense(DT_FLOAT, TensorShape({2}));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({2}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCMulTest, BroadcastDense) {
  MakeOp<float>();
  // [    1]
  // [2    ] (shape [3,2])  cmul  [0.5  0] (shape [2])
  // [3   4]
  //
  // Result:
  // [?   0]
  // [1   ?]  where ? remains implicitly zero.
  // [1.5 0]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  Tensor dense(DT_FLOAT, TensorShape({2}));
  auto dense_flat = dense.flat<float>();
  dense_flat(0) = 0.5;
  dense_flat(1) = 0;

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({2}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {0, 1, 1.5, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace

}  // namespace tensorflow
