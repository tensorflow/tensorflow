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

#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ReverseTest : public ClientLibraryTestBase {};

// Tests the reverse operation on a scalar.
XLA_TEST_F(ReverseTest, ReverseScalar) {
  ComputationBuilder b(client_, TestName());
  float input = 3.5f;
  b.Rev(b.ConstantR0<float>(input), {});
  ComputeAndCompareR0<float>(&b, input, {});
}

// Tests the reverse operation on a 0x0 float array on both dimensions.
XLA_TEST_F(ReverseTest, Reverse0x0FloatArray) {
  ComputationBuilder b(client_, TestName());
  b.Rev(b.ConstantR2FromArray2D<float>(Array2D<float>(0, 0)), {0, 1});
  ComputeAndCompareR2<float>(&b, Array2D<float>(0, 0), {});
}

// Tests the reverse operation on a 0x1 float array on both dimensions.
XLA_TEST_F(ReverseTest, Reverse0x1FloatArray) {
  ComputationBuilder b(client_, TestName());
  b.Rev(b.ConstantR2FromArray2D<float>(Array2D<float>(0, 1)), {0, 1});
  ComputeAndCompareR2<float>(&b, Array2D<float>(0, 1), {});
}

// Tests the reverse operation on a 1x0 float array on both dimensions.
XLA_TEST_F(ReverseTest, Reverse1x0FloatArray) {
  ComputationBuilder b(client_, TestName());
  b.Rev(b.ConstantR2FromArray2D<float>(Array2D<float>(1, 0)), {0, 1});
  ComputeAndCompareR2<float>(&b, Array2D<float>(1, 0), {});
}

// Tests the reverse operation on a 1x1 float array on both dimensions.
XLA_TEST_F(ReverseTest, Reverse1x1FloatArray) {
  ComputationBuilder b(client_, TestName());
  Array2D<float> input({{3.5f}});
  b.Rev(b.ConstantR2FromArray2D<float>(input), {0, 1});
  ComputeAndCompareR2<float>(&b, input, {});
}

XLA_TEST_F(ReverseTest, Reverse2x0x4x3FloatArrayDim02) {
  ComputationBuilder b(client_, TestName());
  b.Rev(b.ConstantR4FromArray4D<float>(Array4D<float>(2, 0, 4, 3)), {0, 2});
  ComputeAndCompareR4<float>(&b, Array4D<float>(2, 0, 4, 3), {});
}

XLA_TEST_F(ReverseTest, Reverse2x0x4x3FloatArrayDim13) {
  ComputationBuilder b(client_, TestName());
  b.Rev(b.ConstantR4FromArray4D<float>(Array4D<float>(2, 0, 4, 3)), {1, 3});
  ComputeAndCompareR4<float>(&b, Array4D<float>(2, 0, 4, 3), {});
}

// Tests the reverse operation on a 4D U8 array on dimension 0 and 3.
XLA_TEST_F(ReverseTest, Reverse4DU8ArrayOnDim23) {
  ComputationBuilder b(client_, TestName());
  // Input shape is U8[1x2x3x4].
  // clang-format off
  Array4D<uint8> input({{
    {{1, 2, 3, 4},
     {5, 6, 7, 8},
     {9, 10, 11, 12}},
    {{13, 14, 15, 16},
     {17, 18, 19, 20},
     {21, 22, 23, 24}},
  }});
  // clang-format on

  b.Rev(b.ConstantR4FromArray4D<uint8>(input), {0, 3});

  // clang-format off
  Array4D<uint8> expected({{
    {{4, 3, 2, 1},
     {8, 7, 6, 5},
     {12, 11, 10, 9}},
    {{16, 15, 14, 13},
     {20, 19, 18, 17},
     {24, 23, 22, 21}},
  }});
  // clang-format on
  ComputeAndCompareR4<uint8>(&b, expected, {});
}

// Tests the reverse operation on a 4D float array on dimension 0 and 1.
TEST_F(ReverseTest, Reverse4DFloatArrayOnDim01) {
  ComputationBuilder b(client_, TestName());
  // Input shape is float[4x3x2x1].
  // clang-format off
  Array4D<float> input({
    {{{1.0f}, {2.0f}},
     {{3.0f}, {4.0f}},
     {{5.0f}, {6.0f}}},
    {{{7.0f}, {8.0f}},
     {{9.0f}, {10.0f}},
     {{11.0f}, {12.0f}}},
    {{{13.0f}, {14.0f}},
     {{15.0f}, {16.0f}},
     {{17.0f}, {18.0f}}},
    {{{19.0f}, {20.0f}},
     {{21.0f}, {22.0f}},
     {{23.0f}, {24.0f}}},
  });
  // clang-format on

  b.Rev(b.ConstantR4FromArray4D<float>(input), {0, 1});

  // clang-format off
  Array4D<float> expected({
    {{{23.0f}, {24.0f}},
     {{21.0f}, {22.0f}},
     {{19.0f}, {20.0f}}},
    {{{17.0f}, {18.0f}},
     {{15.0f}, {16.0f}},
     {{13.0f}, {14.0f}}},
    {{{11.0f}, {12.0f}},
     {{9.0f}, {10.0f}},
     {{7.0f}, {8.0f}}},
    {{{5.0f}, {6.0f}},
     {{3.0f}, {4.0f}},
     {{1.0f}, {2.0f}}},
  });
  // clang-format on
  ComputeAndCompareR4<float>(&b, expected, {}, ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
