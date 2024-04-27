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

#include <memory>
#include <vector>

#include "xla/client/global_data.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/test_helpers.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class ClientTest : public ClientLibraryTestBase {};

XLA_TEST_F(ClientTest, ExecuteWithLayout) {
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

      ASSERT_TRUE(LiteralTestUtil::EqualShapesAndLayouts(
          expected_literal.shape(), computed.shape()));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
    }
  }
}

XLA_TEST_F(ClientTest, ExecuteWithTupleLayout) {
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

// Disabled for interpreter since ExecuteAsyncOnStream is not implemented on
// interpreter backend.
XLA_TEST_F(ClientTest,
           DISABLED_ON_INTERPRETER(DISABLED_ON_GPU(ExecuteParallel))) {
  XlaComputation add_with_one_arg, mul_with_two_args, dot_with_one_arg;
  Shape shape = ShapeUtil::MakeShape(S32, {2, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> const_arg,
      client_->TransferToServer(
          LiteralUtil::CreateR2<int32_t>({{5, 6}, {7, 8}})));

  XlaBuilder b(TestName() + ".add");
  Add(Parameter(&b, 0, shape, "param_0"),
      ConstantR2<int32_t>(&b, {{1, 2}, {3, 4}}));
  TF_ASSERT_OK_AND_ASSIGN(add_with_one_arg, b.Build());

  // We can't really test parallel execution on CPU since all of the cores in a
  // CPU are presented as a single device.  So for now we test "parallel"
  // execution on a single device.
  std::vector<Client::XlaComputationInstance> computation_instances;
  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::DeviceHandle> devices,
                          client_->GetDeviceHandles(1));
  ASSERT_EQ(devices.size(), 1);

  ExecutionOptions options = execution_options_;
  *options.add_device_handles() = devices[0];
  computation_instances.push_back(Client::XlaComputationInstance(
      add_with_one_arg, {const_arg.get()}, options, nullptr));

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          client_->ExecuteParallel(computation_instances));
  auto expected_result = LiteralUtil::CreateR2<int32_t>({{6, 8}, {10, 12}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result_literal,
      client_->Transfer(*results[0], &expected_result.shape()));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result_literal));
}

}  // namespace
}  // namespace xla
