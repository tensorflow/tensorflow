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

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

#include <cmath>
#include <initializer_list>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class ArithmeticTest : public ClientLibraryTestBase {
 public:
  template <typename NativeT, typename IndexT>
  void TestArgMin(std::initializer_list<std::initializer_list<NativeT>> input,
                  absl::Span<IndexT const> expected_output, int axis,
                  bool tie_low) {
    TestArgMinMax(input, expected_output, axis, /*is_min=*/true, tie_low);
  }

  template <typename NativeT, typename IndexT>
  void TestArgMax(std::initializer_list<std::initializer_list<NativeT>> input,
                  absl::Span<IndexT const> expected_output, int axis,
                  bool tie_low) {
    TestArgMinMax(input, expected_output, axis, /*is_min=*/false, tie_low);
  }

 private:
  // Test ArgMin/ArgMax implementation, both single- and two- pass.
  template <typename NativeT, typename IndexT>
  void TestArgMinMax(
      std::initializer_list<std::initializer_list<NativeT>> input,
      absl::Span<IndexT const> expected_output, int axis, bool is_min,
      bool tie_low) {
    if (is_min) {
      TestArgMinMaxImpl(
          input, expected_output, [=](XlaOp op, PrimitiveType type) {
            return ArgMin(op, type, axis, /*stable=*/true, tie_low);
          });
      TestArgMinMaxImpl(input, expected_output,
                        [=](XlaOp op, PrimitiveType type) {
                          return ArgMinTwoPass(op, type, axis, tie_low);
                        });
    } else {
      TestArgMinMaxImpl(
          input, expected_output, [=](XlaOp op, PrimitiveType type) {
            return ArgMax(op, type, axis, /*stable=*/true, tie_low);
          });
      TestArgMinMaxImpl(input, expected_output,
                        [=](XlaOp op, PrimitiveType type) {
                          return ArgMaxTwoPass(op, type, axis, tie_low);
                        });
    }
  }

  template <typename NativeT, typename IndexT>
  void TestArgMinMaxImpl(
      std::initializer_list<std::initializer_list<NativeT>> input,
      absl::Span<IndexT const> expected_output,
      std::function<void(XlaOp, PrimitiveType)> MinMaxImpl) {
    XlaBuilder builder(TestName());
    XlaOp x = ConstantR2<NativeT>(&builder, input);
    MinMaxImpl(x, primitive_util::NativeToPrimitiveType<IndexT>());
    ComputeAndCompareR1<IndexT>(&builder, expected_output, {});
  }
};

XLA_TEST_F(ArithmeticTest, ArgMinR2Axis0) {
  TestArgMin<int32, int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 2},
                           /*axis=*/0, /*tie_low=*/true);
  TestArgMin<int32, int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 2, 2},
                           /*axis=*/0, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMinR2Axis1) {
  TestArgMin<int32, int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 1},
                           /*axis=*/1, /*tie_low=*/true);
  TestArgMin<int32, int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 2},
                           /*axis=*/1, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMinFloatR2Axis1) {
  TestArgMin<float, int32>(
      {
          {NAN, 6, 7, -1, 4, NAN, -50},
      },
      {6}, /*axis=*/1, /*tie_low=*/true);
}

XLA_TEST_F(ArithmeticTest, ArgMinComplexR2Axis0) {
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(9.f, -1.f), complex64(3.f, 0.f)}},
                               {0, 1},
                               /*axis=*/0, /*tie_low=*/true);
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(9.f, -1.f), complex64(3.f, 0.f)}},
                               {0, 1},
                               /*axis=*/0, /*tie_low=*/false);
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(NAN, 2.f), complex64(3.f, NAN)}},
                               {0, 0}, /*axis=*/0, /*tie_low=*/true);
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(NAN, 2.f), complex64(3.f, NAN)}},
                               {0, 0}, /*axis=*/0, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMinComplexR2Axis1) {
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(1.f, 2.f)},
                                {complex64(9.f, -1.f), complex64(3.f, 0.f)}},
                               {0, 1},
                               /*axis=*/1, /*tie_low=*/true);
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(1.f, 2.f)},
                                {complex64(9.f, -1.f), complex64(3.f, 0.f)}},
                               {1, 1},
                               /*axis=*/1, /*tie_low=*/false);
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(NAN, 2.f), complex64(3.f, NAN)}},
                               {0, 0}, /*axis=*/1, /*tie_low=*/true);
  TestArgMin<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(NAN, 2.f), complex64(3.f, NAN)}},
                               {0, 0}, /*axis=*/1, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMaxR2Axis0) {
  TestArgMax<int32, int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {2, 0, 1},
                           /*axis=*/0, /*tie_low=*/true);
}

XLA_TEST_F(ArithmeticTest, ArgMaxR2Axis1) {
  TestArgMax<int32, int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {1, 0, 0},
                           /*axis=*/1, /*tie_low=*/true);
}

XLA_TEST_F(ArithmeticTest, ArgMaxFloat32R2Axis1) {
  TestArgMax<float, int32>(
      {
          {NAN, 6, 7, 7, 4, NAN, -50},
      },
      {2}, /*axis=*/1, /*tie_low=*/true);
  TestArgMax<float, int32>(
      {
          {NAN, 6, 7, 7, 4, NAN, -50},
      },
      {3}, /*axis=*/1, /*tie_low=*/false);
  TestArgMax<float, int32>(
      {
          {NAN, NAN, NAN},
      },
      {0}, /*axis=*/1, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMaxFloat16R2Axis1) {
  TestArgMax<half, int32>(
      {
          {half(NAN), half(6.f), half(7.f), half(7.f), half(4.f), half(NAN),
           half(-50.f)},
      },
      {2}, /*axis=*/1, /*tie_low=*/true);
}

XLA_TEST_F(ArithmeticTest, ArgMaxComplexR2Axis0) {
  TestArgMax<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(9.f, -1.f), complex64(3.f, 0.f)}},
                               {1, 0},
                               /*axis=*/0, /*tie_low=*/true);
  TestArgMax<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(9.f, -1.f), complex64(3.f, 0.f)}},
                               {1, 0},
                               /*axis=*/0, /*tie_low=*/false);
  TestArgMax<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(NAN, 2.f), complex64(3.f, NAN)}},
                               {0, 0}, /*axis=*/0, /*tie_low=*/true);
  TestArgMax<complex64, int32>({{complex64(1.f, 2.f), complex64(3.f, 10.f)},
                                {complex64(NAN, 2.f), complex64(3.f, NAN)}},
                               {0, 0}, /*axis=*/0, /*tie_low=*/false);
}

}  // namespace
}  // namespace xla
