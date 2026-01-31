/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;

class ReplayTest : public ClientLibraryTestRunnerMixin<
                       HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {};

TEST_F(ReplayTest, TwoPlusTwoReplay) {
  // Make 2+2 computation.
  XlaBuilder builder(TestName());
  auto two = ConstantR0<int32_t>(&builder, 2);
  Add(two, two);
  XlaComputation computation = builder.Build().value();

  // Serialize it out.
  std::unique_ptr<HloSnapshot> module = computation.Snapshot().value();

  // Replay it.
  XlaComputation replayed(module->hlo().hlo_module());

  // Check signature is the same.
  ASSERT_OK_AND_ASSIGN(ProgramShape original_shape,
                       computation.GetProgramShape());
  ASSERT_OK_AND_ASSIGN(ProgramShape replayed_shape, replayed.GetProgramShape());
  ASSERT_THAT(replayed_shape.ToProto(), EqualsProto(original_shape.ToProto()));

  // Run it.
  ASSERT_OK_AND_ASSIGN(Literal literal,
                       ExecuteAndTransfer(replayed, /*arguments=*/{}));

  // Expect 4.
  LiteralTestUtil::ExpectR0Equal<int32_t>(4, literal);
}

TEST_F(ReplayTest, XPlusYReplayWithParameters) {
  // Make computation.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(S32, {}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(S32, {}), "y");
  Add(x, y);
  XlaComputation computation = builder.Build().value();

  // Serialize it out.
  std::unique_ptr<HloSnapshot> module = computation.Snapshot().value();

  // Replay it.
  XlaComputation replayed(module->hlo().hlo_module());

  // Check signature is the same.
  ASSERT_OK_AND_ASSIGN(ProgramShape original_shape,
                       computation.GetProgramShape());
  ASSERT_OK_AND_ASSIGN(ProgramShape replayed_shape, replayed.GetProgramShape());
  ASSERT_THAT(replayed_shape.ToProto(), EqualsProto(original_shape.ToProto()));

  // Run it.
  Literal x_data = LiteralUtil::CreateR0<int32_t>(2);
  Literal y_data = LiteralUtil::CreateR0<int32_t>(3);
  ASSERT_OK_AND_ASSIGN(Literal literal,
                       ExecuteAndTransfer(replayed,
                                          /*arguments=*/{&x_data, &y_data}));

  // Expect 5.
  LiteralTestUtil::ExpectR0Equal<int32_t>(5, literal);
}

TEST_F(ReplayTest, MapPlusTwoOverR1) {
  // As above, but with map(+2) over some constant array.
  XlaBuilder plus_two_builder("plus two");
  auto input =
      Parameter(&plus_two_builder, 0, ShapeUtil::MakeShape(S32, {}), "input");
  Add(input, ConstantR0<int32_t>(&plus_two_builder, 2));
  XlaComputation plus_two = plus_two_builder.Build().value();

  XlaBuilder mapper_builder(TestName());
  auto original = ConstantR1<int32_t>(&mapper_builder, {1, 2, 3});
  Map(&mapper_builder, {original}, plus_two, {0});

  XlaComputation computation = mapper_builder.Build().value();

  // Serialize it out.
  std::unique_ptr<HloSnapshot> module = computation.Snapshot().value();

  // Replay it.
  XlaComputation replayed(module->hlo().hlo_module());

  // Check signature is the same.
  ASSERT_OK_AND_ASSIGN(ProgramShape original_shape,
                       computation.GetProgramShape());
  ASSERT_OK_AND_ASSIGN(ProgramShape replayed_shape, replayed.GetProgramShape());
  ASSERT_THAT(replayed_shape.ToProto(), EqualsProto(original_shape.ToProto()));

  // Run it.
  ASSERT_OK_AND_ASSIGN(Literal literal,
                       ExecuteAndTransfer(replayed, /*arguments=*/{}));

  // Expect result.
  LiteralTestUtil::ExpectR1Equal<int32_t>({3, 4, 5}, literal);
}

}  // namespace
}  // namespace xla
