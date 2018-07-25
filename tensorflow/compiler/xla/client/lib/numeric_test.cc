/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class NumericTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  void TestMatrixDiagonal();
};

// TODO(b/64798317): Delete this test case once xla::IotaGen is converted to
// xla::Iota. This test is already implemented for xla::IotaGen in
// xla/tests/iota_test.cc.
XLA_TEST_F(NumericTest, Iota) {
  XlaBuilder builder(TestName());
  Iota(&builder, S32, 10);

  ComputeAndCompareR1<int32>(&builder, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {});
}

XLA_TEST_F(NumericTest, Triangle) {
  XlaBuilder builder(TestName());
  Array3D<int32> input(2, 3, 4);
  input.FillIota(0);

  XlaOp a;
  auto a_data = CreateR3Parameter<int32>(input, 0, "a", &builder, &a);
  LowerTriangle(a);
  Array3D<int32> expected({{{0, 0, 0, 0}, {4, 5, 0, 0}, {8, 9, 10, 0}},
                           {{12, 0, 0, 0}, {16, 17, 0, 0}, {20, 21, 22, 0}}});

  ComputeAndCompareR3<int32>(&builder, expected, {a_data.get()});
}

template <typename T>
void NumericTest::TestMatrixDiagonal() {
  XlaBuilder builder("GetMatrixDiagonal");
  Array3D<T> input(2, 3, 4);
  input.FillIota(0);

  XlaOp a;
  auto a_data = CreateR3Parameter<T>(input, 0, "a", &builder, &a);
  GetMatrixDiagonal(a);
  Array2D<T> expected({{0, 5, 10}, {12, 17, 22}});

  ComputeAndCompareR2<T>(&builder, expected, {a_data.get()});
}

XLA_TEST_F(NumericTest, GetMatrixDiagonal_S32) { TestMatrixDiagonal<int32>(); }

XLA_TEST_F(NumericTest, GetMatrixDiagonal_S64) { TestMatrixDiagonal<int64>(); }

XLA_TEST_F(NumericTest, GetMatrixDiagonal_F32) { TestMatrixDiagonal<float>(); }

}  // namespace
}  // namespace xla
