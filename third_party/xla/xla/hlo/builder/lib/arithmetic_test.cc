/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/arithmetic.h"

#include <cstdint>
#include <functional>
#include <initializer_list>

#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ArithmeticTest : public ClientLibraryTestRunnerMixin<
                           HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 public:
  template <typename NativeT>
  void TestArgMin(std::initializer_list<std::initializer_list<NativeT>> input,
                  absl::Span<NativeT const> expected_output, int axis) {
    TestArgMinMax(input, expected_output, axis, /*is_min=*/true);
  }

  template <typename NativeT>
  void TestArgMax(std::initializer_list<std::initializer_list<NativeT>> input,
                  absl::Span<NativeT const> expected_output, int axis) {
    TestArgMinMax(input, expected_output, axis, /*is_min=*/false);
  }

 private:
  // Test ArgMin/ArgMax implementation, both single- and two- pass.
  template <typename NativeT>
  void TestArgMinMax(
      std::initializer_list<std::initializer_list<NativeT>> input,
      absl::Span<NativeT const> expected_output, int axis, bool is_min) {
    XlaBuilder builder(TestName());
    XlaOp x = ConstantR2<NativeT>(&builder, input);
    ArgMinMax(x, primitive_util::NativeToPrimitiveType<NativeT>(), axis,
              /*is_min=*/is_min);
    ComputeAndCompareR1<NativeT>(&builder, expected_output, {});
  }

  template <typename NativeT>
  void TestArgMinMaxImpl(
      std::initializer_list<std::initializer_list<NativeT>> input,
      absl::Span<NativeT const> expected_output,
      std::function<void(XlaOp, PrimitiveType)> MinMaxImpl) {}
};

TEST_F(ArithmeticTest, ArgMinR2Axis0) {
  TestArgMin<int32_t>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 2},
                      /*axis=*/0);
}

TEST_F(ArithmeticTest, ArgMinR2Axis1) {
  TestArgMin<int32_t>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {0, 1, 1},
                      /*axis=*/1);
}

TEST_F(ArithmeticTest, ArgMaxR2Axis0) {
  TestArgMax<int32_t>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {2, 0, 1},
                      /*axis=*/0);
}

TEST_F(ArithmeticTest, ArgMaxR2Axis1) {
  TestArgMax<int32_t>({{1, 7, 4}, {6, 3, 5}, {8, 3, 3}}, {1, 0, 0},
                      /*axis=*/1);
}

}  // namespace
}  // namespace xla
