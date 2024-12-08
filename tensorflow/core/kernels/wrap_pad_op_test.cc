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

class WrapPadOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("wrap_pad_op", "WrapPad")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST(T)                                                     \
  TEST_F(WrapPadOpTest, TestWrapPad##T) {                                    \
    MakeOp<T>();                                                             \
    AddInputFromArray<T>(TensorShape({1, 2, 3, 1}), {1, 2, 3, 4, 5, 6});     \
    AddInputFromArray<int32>(TensorShape({4, 2}), {0, 0, 1, 1, 2, 2, 0, 0}); \
    TF_ASSERT_OK(RunOpKernel());                                             \
                                                                             \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                   \
                    TensorShape({1, 4, 7, 1}));                              \
    test::FillValues<T>(&expected,                                           \
                        {5, 6, 4, 5, 6, 4, 5, 2, 3, 1, 2, 3, 1, 2,           \
                         5, 6, 4, 5, 6, 4, 5, 2, 3, 1, 2, 3, 1, 2});         \
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
REGISTER_TEST(int64_t)

#undef REGISTER_TEST

TEST_F(WrapPadOpTest, TestWrapPadLargeInput) {
  MakeOp<float>();
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
      return kInput - kPad + i;
    else if (kPad <= i && i < kInput + kPad)
      return i - kPad;
    else if (kInput + kPad <= i && i < kOutput)
      return i - kInput - kPad;
    else
      return -1;
  });

  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

class WrapPadGradOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("wrap_pad_grad_op", "WrapPadGrad")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST(T)                                                      \
  TEST_F(WrapPadGradOpTest, TestWrapPadGrad##T) {                             \
    MakeOp<T>();                                                              \
    AddInput<T>(TensorShape({1, 4, 7, 1}), [](int i) -> T { return i % 7; }); \
    AddInputFromArray<int32>(TensorShape({4, 2}), {0, 0, 1, 1, 2, 2, 0, 0});  \
    TF_ASSERT_OK(RunOpKernel());                                              \
                                                                              \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                    \
                    TensorShape({1, 2, 3, 1}));                               \
    test::FillValues<T>(&expected, {14, 18, 10, 14, 18, 10});                 \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                      \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(uint8)
REGISTER_TEST(uint16)
REGISTER_TEST(int8)
REGISTER_TEST(int16)
REGISTER_TEST(int32)
REGISTER_TEST(int64_t)

#undef REGISTER_TEST

}  // namespace tensorflow
