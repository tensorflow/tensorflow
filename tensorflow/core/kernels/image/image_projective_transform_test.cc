/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
// #include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
class ImageProjectiveTransformV3OpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(const string& interpolation, const string& fill_mode) {
    TF_EXPECT_OK(NodeDefBuilder("image_projective_transform_v3_op",
                                "ImageProjectiveTransformV3")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_FLOAT))  // transform
                     .Input(FakeInput(DT_INT32))  // output shape
                     .Input(FakeInput(DT_FLOAT))  // fill_value
                     .Attr("interpolation", interpolation)
                     .Attr("fill_mode", fill_mode)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST(T)                                                    \
  TEST_F(ImageProjectiveTransformV3OpTest, TestConstantFill##T##nearest) {  \
    constexpr uint8 FILL_VALUE = 42;                                        \
    MakeOp<T>("NEAREST", "CONSTANT");                                       \
    /* Input:      */                                                       \
    /* [[1, 1, 1]  */                                                       \
    /*  [1, 1, 1]  */                                                       \
    /*  [1, 1, 1]] */                                                       \
    AddInputFromArray<T>(TensorShape({1, 3, 3, 1}),                         \
                         {1, 1, 1, 1, 1, 1, 1, 1, 1});                      \
                                                                            \
    /* Rotation 45 degrees */                                               \
    AddInputFromArray<float>(TensorShape({1, 8}),                           \
                             {0.70710677, -0.70710677, 1., 0.70710677,      \
                              0.70710677, -0.41421354, 0., 0.});            \
    AddInputFromArray<int32>(TensorShape({2}), {3, 3});                     \
    AddInputFromArray<float>(TensorShape({}), {FILL_VALUE});                \
    TF_ASSERT_OK(RunOpKernel());                                            \
                                                                            \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                  \
                    TensorShape({1, 3, 3, 1}));                             \
    /* Output (C = fill_value): */                                          \
    /* [[C, 1, C]  */                                                       \
    /*  [1, 1, 1]  */                                                       \
    /*  [C, 1, C]] */                                                       \
    test::FillValues<T>(&expected, {FILL_VALUE, 1, FILL_VALUE, 1, 1, 1,     \
                                    FILL_VALUE, 1, FILL_VALUE});            \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                    \
  }                                                                         \
                                                                            \
  TEST_F(ImageProjectiveTransformV3OpTest, TestConstantFill##T##bilinear) { \
    constexpr uint8 FILL_VALUE = 42;                                        \
    MakeOp<T>("BILINEAR", "CONSTANT");                                      \
    /* Input:      */                                                       \
    /* [[1, 1, 1]  */                                                       \
    /*  [1, 1, 1]  */                                                       \
    /*  [1, 1, 1]] */                                                       \
    AddInputFromArray<T>(TensorShape({1, 3, 3, 1}),                         \
                         {1, 1, 1, 1, 1, 1, 1, 1, 1});                      \
                                                                            \
    /* Rotation 45 degrees */                                               \
    AddInputFromArray<float>(TensorShape({1, 8}),                           \
                             {0.70710677, -0.70710677, 1., 0.70710677,      \
                              0.70710677, -0.41421354, 0., 0.});            \
    AddInputFromArray<int32>(TensorShape({2}), {3, 3});                     \
    AddInputFromArray<float>(TensorShape({}), {FILL_VALUE});                \
    TF_ASSERT_OK(RunOpKernel());                                            \
                                                                            \
    Tensor expected(allocator(), DataTypeToEnum<T>::value,                  \
                    TensorShape({1, 3, 3, 1}));                             \
    /* Output (C = fill_value): */                                          \
    /* [[C, 1, C]  */                                                       \
    /*  [1, 1, 1]  */                                                       \
    /*  [C, 1, C]] */                                                       \
    test::FillValues<T>(&expected, {FILL_VALUE, 1, FILL_VALUE, 1, 1, 1,     \
                                    FILL_VALUE, 1, FILL_VALUE});            \
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));                    \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(uint8)
REGISTER_TEST(int32)
REGISTER_TEST(int64)

#undef REGISTER_TEST
}  // namespace tensorflow
