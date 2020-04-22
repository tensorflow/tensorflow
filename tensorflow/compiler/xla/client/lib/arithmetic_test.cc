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
  template <typename NativeT>
  void TestArgMin(std::initializer_list<std::initializer_list<NativeT>> input,
                  absl::Span<NativeT const> expected_output, int axis,
                  bool tie_low) {
    TestArgMinMax(input, expected_output, axis, /*is_min=*/true, tie_low);
  }

  template <typename NativeT>
  void TestArgMax(std::initializer_list<std::initializer_list<NativeT>> input,
                  absl::Span<NativeT const> expected_output, int axis,
                  bool tie_low) {
    TestArgMinMax(input, expected_output, axis, /*is_min=*/false, tie_low);
  }

 private:
  // Test ArgMin/ArgMax implementation, both single- and two- pass.
  template <typename NativeT>
  void TestArgMinMax(
      std::initializer_list<std::initializer_list<NativeT>> input,
      absl::Span<NativeT const> expected_output, int axis, bool is_min,
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

  template <typename NativeT>
  void TestArgMinMaxImpl(
      std::initializer_list<std::initializer_list<NativeT>> input,
      absl::Span<NativeT const> expected_output,
      std::function<void(XlaOp, PrimitiveType)> MinMaxImpl) {
    XlaBuilder builder(TestName());
    XlaOp x = ConstantR2<NativeT>(&builder, input);
    MinMaxImpl(x, primitive_util::NativeToPrimitiveType<NativeT>());
    ComputeAndCompareR1<NativeT>(&builder, expected_output, {});
  }
};

XLA_TEST_F(ArithmeticTest, ArgMinR2Axis0) {
  TestArgMin<int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 2},
                    /*axis=*/0, /*tie_low=*/true);
  TestArgMin<int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 2, 2},
                    /*axis=*/0, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMinR2Axis1) {
  TestArgMin<int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 1},
                    /*axis=*/1, /*tie_low=*/true);
  TestArgMin<int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 2},
                    /*axis=*/1, /*tie_low=*/false);
}

XLA_TEST_F(ArithmeticTest, ArgMaxR2Axis0) {
  TestArgMax<int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {2, 0, 1},
                    /*axis=*/0, /*tie_low=*/true);
}

XLA_TEST_F(ArithmeticTest, ArgMaxR2Axis1) {
  TestArgMax<int32>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {1, 0, 0},
                    /*axis=*/1, /*tie_low=*/true);
}

}  // namespace
}  // namespace xla
