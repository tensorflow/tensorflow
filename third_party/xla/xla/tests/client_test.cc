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
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ClientTest : public ClientLibraryTestBase {};

TEST_F(ClientTest, ExecuteWithLayout) {
  XlaBuilder b(TestName());

  std::vector<std::vector<int64_t>> layouts = {{0, 1}, {1, 0}};
  for (const std::vector<int64_t>& execute_layout : layouts) {
    for (const std::vector<int64_t>& transfer_layout : layouts) {
      Add(ConstantR2<int32_t>(&b, {{1, 2}, {3, 4}}),
          ConstantR2<int32_t>(&b, {{10, 20}, {30, 40}}));
      TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());

      ExecutionOptions execution_options = execution_options_;
      *execution_options.mutable_shape_with_output_layout() =
          ShapeUtil::MakeShapeWithDenseLayout(S32, /*dimensions=*/{2, 2},
                                              execute_layout)
              .ToProto();
      TF_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<GlobalData> data,
          client_->Execute(computation, {}, &execution_options));

      Literal expected_literal = LiteralUtil::CreateR2WithLayout<int32_t>(
          {{11, 22}, {33, 44}}, LayoutUtil::MakeLayout(transfer_layout));

      TF_ASSERT_OK_AND_ASSIGN(
          auto computed, client_->Transfer(*data, &expected_literal.shape()));

      ASSERT_THAT(
          computed.shape().ToProto(),
          tsl::proto_testing::EqualsProto(expected_literal.shape().ToProto()));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
    }
  }
}

TEST_F(ClientTest, ExecuteWithTupleLayout) {
  XlaBuilder b(TestName());

  Tuple(&b, {ConstantR2<int32_t>(&b, {{1, 2}, {3, 4}}),
             ConstantR2<int32_t>(&b, {{10, 20}, {30, 40}})});

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());

  ExecutionOptions execution_options = execution_options_;
  // Create a result shape with one element column major and the other row
  // major.
  *execution_options.mutable_shape_with_output_layout() =
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShapeWithDenseLayout(S32, /*dimensions=*/{2, 2},
                                               /*minor_to_major=*/{0, 1}),
           ShapeUtil::MakeShapeWithDenseLayout(S32, /*dimensions=*/{2, 2},
                                               /*minor_to_major=*/{1, 0})})
          .ToProto();

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      client_->ExecuteAndTransfer(computation, {}, &execution_options));
  LiteralTestUtil::ExpectR2Equal<int32_t>({{1, 2}, {3, 4}},
                                          LiteralSlice(result, {0}));
  LiteralTestUtil::ExpectR2Equal<int32_t>({{10, 20}, {30, 40}},
                                          LiteralSlice(result, {1}));

  EXPECT_TRUE(result.shape().IsTuple());
  EXPECT_EQ(2, ShapeUtil::TupleElementCount(result.shape()));

  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::GetTupleElementShape(result.shape(), 0),
      ShapeUtil::MakeShapeWithDenseLayout(S32, /*dimensions=*/{2, 2},
                                          /*minor_to_major=*/{0, 1})));
  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::GetTupleElementShape(result.shape(), 1),
      ShapeUtil::MakeShapeWithDenseLayout(S32, /*dimensions=*/{2, 2},
                                          /*minor_to_major=*/{1, 0})));
}

}  // namespace
}  // namespace xla
