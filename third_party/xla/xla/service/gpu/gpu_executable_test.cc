/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_executable.h"

#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::IsOkAndHolds;

TEST(GpuExecutableTest, OuputInfoToAndFromProto) {
  const GpuExecutable::OutputInfo output_info0{/*allocation_index=*/42,
                                               /*passthrough=*/true,
                                               /*alias_config=*/std::nullopt};
  EXPECT_THAT(output_info0.ToProto(), EqualsProto(R"pb(
                allocation_index: 42,
                passthrough: true
              )pb"));
  EXPECT_THAT(GpuExecutable::OutputInfo::FromProto(output_info0.ToProto()),
              IsOkAndHolds(output_info0));

  const GpuExecutable::OutputInfo output_info1{
      /*allocation_index=*/43,
      /*passthrough=*/false,
      /*alias_config=*/
      HloInputOutputAliasConfig::Alias{
          /*parameter_number=*/89, /*parameter_index=*/ShapeIndex{1, 2, 3, 4},
          /*kind=*/HloInputOutputAliasConfig::kMustAlias}};
  EXPECT_THAT(output_info1.ToProto(), EqualsProto(R"pb(
                allocation_index: 43,
                alias_config {
                  parameter_number: 89,
                  parameter_shape_index: [ 1, 2, 3, 4 ],
                  kind: MUST_ALIAS
                }
              )pb"));
  EXPECT_THAT(GpuExecutable::OutputInfo::FromProto(output_info1.ToProto()),
              IsOkAndHolds(output_info1));

  const GpuExecutable::OutputInfo output_info2{
      /*allocation_index=*/44,
      /*passthrough=*/true,
      /*alias_config=*/
      HloInputOutputAliasConfig::Alias{
          /*parameter_number=*/0, /*parameter_index=*/ShapeIndex{},
          /*kind=*/HloInputOutputAliasConfig::kMayAlias}};
  EXPECT_THAT(output_info2.ToProto(), EqualsProto(R"pb(
                allocation_index: 44,
                passthrough: true,
                alias_config { kind: MAY_ALIAS }
              )pb"));
  EXPECT_THAT(GpuExecutable::OutputInfo::FromProto(output_info2.ToProto()),
              IsOkAndHolds(output_info2));
}

TEST(GpuExecutableTest, EntryComputationLayout) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.program_shape.AddParameter(ShapeUtil::MakeShape(F32, {1, 2, 3}), "p0");
  params.program_shape.AddParameter(ShapeUtil::MakeShape(U8, {1}), "p1");
  *params.program_shape.mutable_result() = ShapeUtil::MakeShape(F64, {2});
  params.executable =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{}, ThunkSequence{});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          GpuExecutable::Create(std::move(params)));
  EXPECT_THAT(executable->entry_computation_layout().parameter_layouts(),
              ElementsAre(ShapeLayout(ShapeUtil::MakeShape(F32, {1, 2, 3})),
                          ShapeLayout(ShapeUtil::MakeShape(U8, {1}))));
  EXPECT_EQ(executable->entry_computation_layout().result_layout(),
            ShapeLayout(ShapeUtil::MakeShape(F64, {2})));
}

}  // namespace
}  // namespace xla::gpu
