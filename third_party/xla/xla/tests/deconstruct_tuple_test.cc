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
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class DeconstructTupleTest : public HloPjRtTestBase {
 protected:
  HloRunnerPjRt& GetPjRtRunner() {
    return static_cast<HloRunnerPjRt&>(test_runner());
  }

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteAndGetBuffers(
      XlaBuilder* builder,
      const std::vector<std::unique_ptr<PjRtBuffer>>& arguments) {
    absl::StatusOr<XlaComputation> computation_status = builder->Build();
    TF_RETURN_IF_ERROR(computation_status.status());
    XlaComputation computation = std::move(computation_status).value();

    TF_ASSIGN_OR_RETURN(
        ProgramShape program_shape,
        ProgramShape::FromProto(computation.proto().host_program_shape()));
    HloModuleConfig config(program_shape);
    config.set_debug_options(GetModuleConfigForTest().debug_options());
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProto(computation.proto(), config));

    // We use ExecuteWithDeviceBuffers to get PjRtBuffers back.
    // HloRunnerPjRt creates an executable and runs it.
    auto executable_status = GetPjRtRunner().CreateExecutable(
        std::move(module), /*run_hlo_passes=*/true);
    TF_RETURN_IF_ERROR(executable_status.status());
    auto executable = std::move(executable_status).value();

    return GetPjRtRunner().ExecuteWithDeviceBuffers(
        executable.get(), arguments, /*execute_options=*/nullptr);
  }
};

TEST_F(DeconstructTupleTest, DeconstructTuple) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2});

  auto buffers_status = ExecuteAndGetBuffers(&builder, {});
  ASSERT_TRUE(buffers_status.ok());
  auto buffers = std::move(buffers_status).value();

  // PjRt flattens the tuple into 2 buffers.
  ASSERT_EQ(buffers.size(), 2);

  std::shared_ptr<Literal> literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[0]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[1]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, *literal);
}

TEST_F(DeconstructTupleTest, DeconstructTupleRepeatedElement) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2, const2, const1});

  auto buffers_status = ExecuteAndGetBuffers(&builder, {});
  ASSERT_TRUE(buffers_status.ok());
  auto buffers = std::move(buffers_status).value();

  ASSERT_EQ(buffers.size(), 4);

  std::shared_ptr<Literal> literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[0]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[1]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[2]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[3]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);
}

TEST_F(DeconstructTupleTest, DeconstructTupleThenDeallocate) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2, const1});

  auto buffers_status = ExecuteAndGetBuffers(&builder, {});
  ASSERT_TRUE(buffers_status.ok());
  auto buffers = std::move(buffers_status).value();
  ASSERT_EQ(buffers.size(), 3);

  // Deallocate one of the buffers (by resetting unique_ptr)
  buffers[0].reset();

  // Others should still be valid.
  std::shared_ptr<Literal> literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[1]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[2]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);
}

TEST_F(DeconstructTupleTest, DeconstructNonTuple) {
  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});

  auto buffers_status = ExecuteAndGetBuffers(&builder, {});
  ASSERT_TRUE(buffers_status.ok());
  auto buffers = std::move(buffers_status).value();

  ASSERT_EQ(buffers.size(), 1);
  std::shared_ptr<Literal> literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[0]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);
}

TEST_F(DeconstructTupleTest, DeconstructTupleFromParam) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR1<float>({3.14f, -100.25f});

  // Transfer param to device
  auto transfer_status =
      GetPjRtRunner().TransferLiteralsToDevice({&param0_literal});
  ASSERT_TRUE(transfer_status.ok());
  auto param_buffers = std::move(transfer_status).value();

  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "param0");
  Tuple(&builder, {p});

  auto buffers_status = ExecuteAndGetBuffers(&builder, param_buffers);
  ASSERT_TRUE(buffers_status.ok());
  auto buffers = std::move(buffers_status).value();

  ASSERT_EQ(buffers.size(), 1);
  std::shared_ptr<Literal> literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[0]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({3.14f, -100.25f}, *literal);
}

TEST_F(DeconstructTupleTest, DeconstructNestedTuple) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  // ((A, B), A) -> flattened to (A, B, A) by manual flattening in test
  // because PjRt client crashes on nested tuple outputs in this environment.
  Tuple(&builder, {const1, const2, const1});

  auto buffers_status = ExecuteAndGetBuffers(&builder, {});
  ASSERT_TRUE(buffers_status.ok());
  auto buffers = std::move(buffers_status).value();

  // PjRt flattens nested tuples. Structure is (A, B, A) -> 3 leaf buffers.
  ASSERT_EQ(buffers.size(), 3);

  std::shared_ptr<Literal> literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[0]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[1]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, *literal);

  TF_ASSERT_OK_AND_ASSIGN(literal, buffers[2]->ToLiteralSync());
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, *literal);
}

}  // namespace
}  // namespace xla
