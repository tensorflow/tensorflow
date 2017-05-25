/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

class SparseAddOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    DataType thresh_type = value_type;
    if (std::is_same<T, std::complex<float>>::value) {
      thresh_type = DT_FLOAT;
    } else if (std::is_same<T, std::complex<double>>::value) {
      thresh_type = DT_DOUBLE;
    }

    TF_ASSERT_OK(NodeDefBuilder("sparseadd", "SparseAdd")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(thresh_type))
                     .Attr("Treal", thresh_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseAddOpTest, TwoD_AddSparseTensorWithSelf) {
  MakeOp<float>();

  // [    1]
  // [2    ]
  // [3   4]

  const auto indices_shape = TensorShape({4, 2});
  std::initializer_list<int64> in{ 0, 1, 1, 0, 2, 0, 2, 1 };
  const gtl::ArraySlice<int64> indices(in);
  std::initializer_list<int64> sh{ 3, 2 };
  const gtl::ArraySlice<int64> shape(sh);

#define ADD_TENSOR_INPUT()                                  \
  AddInputFromArray<int64>(indices_shape, indices);         \
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4}); \
  AddInputFromArray<int64>(TensorShape({2}), shape);

  ADD_TENSOR_INPUT();
  ADD_TENSOR_INPUT();
  AddInputFromArray<float>(TensorShape({}), {0.0});
#undef ADD_TENSOR_INPUT

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, indices_shape);
  test::FillValues<int64>(&expected_indices, indices);
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, {4});
  test::FillValues<float>(&expected_values, {2, 4, 6, 8});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64,
                        {static_cast<int64>(shape.size())});
  test::FillValues<int64>(&expected_shape, shape);
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

// [    1]     [5    ]      [5   1]
// [2    ]  +  [    6]  ==  [2   6]
// [3   4]     [     ]      [3   4]
#define RUN_TEST(VALTYPE)                                                   \
  TEST_F(SparseAddOpTest, TwoD_AddSparseTensorsWithDiffIndices_##VALTYPE) { \
    MakeOp<VALTYPE>();                                                      \
    DataType val_dtype = tensorflow::DataTypeToEnum<VALTYPE>::value;        \
                                                                            \
    const auto indices_shape = TensorShape({4, 2});                         \
    std::initializer_list<int64> in{0, 1, 1, 0, 2, 0, 2, 1};                \
    const gtl::ArraySlice<int64> indices(in);                               \
    std::initializer_list<int64> sh{3, 2};                                  \
    const gtl::ArraySlice<int64> shape(sh);                                 \
                                                                            \
    AddInputFromArray<int64>(indices_shape, indices);                       \
    AddInputFromArray<VALTYPE>(TensorShape({4}), {1, 2, 3, 4});             \
    AddInputFromArray<int64>(TensorShape({2}), shape);                      \
                                                                            \
    AddInputFromArray<int64>(TensorShape({2, 2}), {0, 0, 1, 1});            \
    AddInputFromArray<VALTYPE>(TensorShape({2}), {5, 6});                   \
    AddInputFromArray<int64>(TensorShape({2}), shape);                      \
                                                                            \
    if (val_dtype == DT_COMPLEX64) {                                        \
      AddInputFromArray<float>(TensorShape({}), {0});                       \
    } else if (val_dtype == DT_COMPLEX128) {                                \
      AddInputFromArray<double>(TensorShape({}), {0});                      \
    } else {                                                                \
      AddInputFromArray<VALTYPE>(TensorShape({}), {0});                     \
    }                                                                       \
                                                                            \
    TF_ASSERT_OK(RunOpKernel());                                            \
                                                                            \
    const int expected_nnz = 6;                                             \
    Tensor expected_indices(allocator(), DT_INT64,                          \
                            TensorShape({expected_nnz, 2}));                \
    test::FillValues<int64>(&expected_indices,                              \
                            {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1});          \
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));        \
                                                                            \
    Tensor expected_values(allocator(), val_dtype, {expected_nnz});         \
    test::FillValues<VALTYPE>(&expected_values, {5, 1, 2, 6, 3, 4});        \
    test::ExpectTensorEqual<VALTYPE>(expected_values, *GetOutput(1));       \
                                                                            \
    Tensor expected_shape(allocator(), DT_INT64,                            \
                          {static_cast<int64>(shape.size())});              \
    test::FillValues<int64>(&expected_shape, shape);                        \
    test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));          \
  }

RUN_TEST(int64);
RUN_TEST(float);
RUN_TEST(double);
RUN_TEST(complex64);
RUN_TEST(complex128);
#undef RUN_TEST

// Adding
//    [    1]
//    [2    ]
//    [3   4]
// to its cwise negation.
#define RUN_TEST(VALTYPE, THRESH)                                        \
  TEST_F(SparseAddOpTest, TwoD_SmallValuesShouldVanish_##VALTYPE) {      \
    MakeOp<VALTYPE>();                                                   \
    DataType val_dtype = tensorflow::DataTypeToEnum<VALTYPE>::value;     \
    const auto indices_shape = TensorShape({4, 2});                      \
    std::initializer_list<int64> in{0, 1, 1, 0, 2, 0, 2, 1};             \
    const gtl::ArraySlice<int64> indices(in);                            \
    std::initializer_list<int64> sh{3, 2};                               \
    const gtl::ArraySlice<int64> shape(sh);                              \
                                                                         \
    auto AddSparseTensor = [indices, indices_shape, shape,               \
                            this](bool negate) {                         \
      AddInputFromArray<int64>(indices_shape, indices);                  \
      if (!negate) {                                                     \
        AddInputFromArray<VALTYPE>(TensorShape({4}), {1, 2, 3, 4});      \
      } else {                                                           \
        AddInputFromArray<VALTYPE>(TensorShape({4}), {-1, -2, -3, -4});  \
      }                                                                  \
      AddInputFromArray<int64>(TensorShape({2}), shape);                 \
    };                                                                   \
    AddSparseTensor(false);                                              \
    AddSparseTensor(true);                                               \
    if (val_dtype == DT_COMPLEX64) {                                     \
      AddInputFromArray<float>(TensorShape({}), {THRESH});               \
    } else if (val_dtype == DT_COMPLEX128) {                             \
      AddInputFromArray<double>(TensorShape({}), {THRESH});              \
    } else {                                                             \
      AddInputFromArray<VALTYPE>(TensorShape({}), {THRESH});             \
    }                                                                    \
                                                                         \
    TF_ASSERT_OK(RunOpKernel());                                         \
                                                                         \
    Tensor expected_indices(allocator(), DT_INT64, TensorShape({0, 2})); \
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));     \
                                                                         \
    Tensor expected_values(allocator(), val_dtype, TensorShape({0}));    \
    test::ExpectTensorEqual<VALTYPE>(expected_values, *GetOutput(1));    \
                                                                         \
    Tensor expected_shape(allocator(), DT_INT64,                         \
                          {static_cast<int64>(shape.size())});           \
    test::FillValues<int64>(&expected_shape, shape);                     \
    test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));       \
  }

RUN_TEST(int64, 1);
RUN_TEST(float, 1e-3f);
RUN_TEST(double, 1e-3f);
RUN_TEST(complex64, 1e-3f);
RUN_TEST(complex128, 1e-3f);
#undef RUN_TEST

}  // namespace

}  // namespace tensorflow
