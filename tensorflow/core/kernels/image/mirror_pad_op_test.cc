/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class MirrorPadOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(const string& mode) {
    TF_EXPECT_OK(NodeDefBuilder("mirror_pad_op", "MirrorPad")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_INT32))
                     .Attr("mode", mode)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST(T)                                                     \
  TEST_F(MirrorPadOpTest, TestMirrorPadReflect##T) {                         \
    MakeOp<T>("REFLECT");                                                    \
    AddInputFromArray<T>(TensorShape({1, 2, 3, 1}), {1, 2, 3, 4, 5, 6});     \
    AddInputFromArray<int32>(TensorShape({4, 2}), {0, 0, 1, 1, 2, 2, 0, 0}); \
    TF_ASSERT_OK(RunOpKernel());                                             \
                                                                             \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                   \
                    TensorShape({1, 4, 7, 1}));                              \
    test::FillValues<T>(&expected,                                           \
                        {6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1,           \
                         6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1});         \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                     \
  }                                                                          \
                                                                             \
  TEST_F(MirrorPadOpTest, TestMirrorPadSymmetric##T) {                       \
    MakeOp<T>("SYMMETRIC");                                                  \
    AddInputFromArray<T>(TensorShape({1, 2, 1, 3}), {1, 2, 3, 4, 5, 6});     \
    AddInputFromArray<int32>(TensorShape({4, 2}), {1, 1, 0, 0, 0, 0, 2, 2}); \
    TF_ASSERT_OK(RunOpKernel());                                             \
                                                                             \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                   \
                    TensorShape({3, 2, 1, 7}));                              \
    test::FillValues<T>(                                                     \
        &expected,                                                           \
        {2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2,      \
         5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5});    \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                     \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(quint8)
REGISTER_TEST(qint8)
REGISTER_TEST(qint32)
REGISTER_TEST(uint8)
REGISTER_TEST(uint16)
REGISTER_TEST(int8)
REGISTER_TEST(int16)
REGISTER_TEST(int32)
REGISTER_TEST(int64)

#undef REGISTER_TEST

TEST_F(MirrorPadOpTest, TestMirrorPadReflectLargeInput) {
  MakeOp<float>("REFLECT");
  // Generate a relatively large input
  const int kInput = 1000;
  const int kPad = 10;
  const int kOutput = kInput + 2 * kPad;

  // Input:
  //  0, 1, 2, ..., 999
  //  0, 1, 2, ..., 999
  //  ... (altogether 1000 lines)
  //  0, 1, 2, ..., 999
  AddInput<float>(TensorShape({1, kInput, kInput, 1}),
                  [=](int i) -> float { return i % kInput; });
  AddInputFromArray<int32>(TensorShape({4, 2}),
                           {0, 0, kPad, kPad, kPad, kPad, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, kOutput, kOutput, 1}));
  test::FillFn<float>(&expected, [=](int i) -> float {
    i = i % kOutput;
    if (0 <= i && i < kPad)
      return kPad - i;
    else if (kPad <= i && i < kInput + kPad)
      return i - kPad;
    else if (kInput + kPad <= i && i < kOutput)
      return 2 * kInput + kPad - 2 - i;
    else
      return -1;
  });

  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(MirrorPadOpTest, TestMirrorPadSymmetricLargeInput) {
  MakeOp<float>("SYMMETRIC");
  // Generate a relatively large input
  const int kInput = 1000;
  const int kPad = 10;
  const int kOutput = kInput + 2 * kPad;

  // Input:
  //  0, 1, 2, ..., 999
  //  0, 1, 2, ..., 999
  //  ... (altogether 1000 lines)
  //  0, 1, 2, ..., 999
  AddInput<float>(TensorShape({1, kInput, kInput, 1}),
                  [=](int i) -> float { return i % kInput; });
  AddInputFromArray<int32>(TensorShape({4, 2}),
                           {0, 0, kPad, kPad, kPad, kPad, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, kOutput, kOutput, 1}));
  test::FillFn<float>(&expected, [=](int i) -> float {
    i = i % kOutput;
    if (0 <= i && i < kPad)
      return kPad - i - 1;
    else if (kPad <= i && i < kInput + kPad)
      return i - kPad;
    else if (kInput + kPad <= i && i < kOutput)
      return 2 * kInput + kPad - 1 - i;
    else
      return -1;
  });

  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

class MirrorPadGradOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(const string& mode) {
    TF_EXPECT_OK(NodeDefBuilder("mirror_pad_grad_op", "MirrorPadGrad")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_INT32))
                     .Attr("mode", mode)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST(T)                                                      \
  TEST_F(MirrorPadGradOpTest, TestMirrorPadGradReflect##T) {                  \
    MakeOp<T>("REFLECT");                                                     \
    AddInput<T>(TensorShape({1, 4, 7, 1}), [](int i) -> T { return i % 7; }); \
    AddInputFromArray<int32>(TensorShape({4, 2}), {0, 0, 1, 1, 2, 2, 0, 0});  \
    TF_ASSERT_OK(RunOpKernel());                                              \
                                                                              \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                    \
                    TensorShape({1, 2, 3, 1}));                               \
    test::FillValues<T>(&expected, {16, 18, 8, 16, 18, 8});                   \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                      \
  }                                                                           \
                                                                              \
  TEST_F(MirrorPadGradOpTest, TestMirrorPadGradSymmetric##T) {                \
    MakeOp<T>("SYMMETRIC");                                                   \
    AddInput<T>(TensorShape({3, 2, 1, 7}), [](int i) -> T { return i % 7; }); \
    AddInputFromArray<int32>(TensorShape({4, 2}), {1, 1, 0, 0, 0, 0, 2, 2});  \
    TF_ASSERT_OK(RunOpKernel());                                              \
                                                                              \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                    \
                    TensorShape({1, 2, 1, 3}));                               \
    test::FillValues<T>(&expected, {9, 27, 27, 9, 27, 27});                   \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                      \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(uint8)
REGISTER_TEST(uint16)
REGISTER_TEST(int8)
REGISTER_TEST(int16)
REGISTER_TEST(int32)
REGISTER_TEST(int64)

#undef REGISTER_TEST

}  // namespace tensorflow
