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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::Property;
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
              absl_testing::IsOkAndHolds(output_info0));

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
              absl_testing::IsOkAndHolds(output_info1));

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
              absl_testing::IsOkAndHolds(output_info2));
}

TEST(GpuExecutableTest, RunThunkPasses) {
  const std::string dump_dir = testing::TempDir();
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_dump_to(dump_dir);
  debug_options.set_xla_gpu_experimental_enable_command_buffer_on_thunks(true);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  int execution_count = 0;
  auto create_executable = [&]() {
    Thunk::ThunkInfo thunk_info;
    BufferAllocation alloc(0, 1024, 0);
    BufferAllocation::Slice slice(&alloc, 0, 1024);

    ThunkSequence thunk_sequence;
    thunk_sequence.push_back(std::make_unique<KernelThunk>(
        thunk_info, "test_kernel",
        emitters::KernelArguments(std::vector<emitters::KernelArgument>()),
        LaunchDimensions(), std::nullopt, 0));
    thunk_sequence.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        thunk_info, slice, slice, 1024));

    GpuExecutable::Params params;
    params.executable = std::make_unique<SequentialThunk>(
        thunk_info, std::move(thunk_sequence));
    params.debug_options = debug_options;

    params.module_name = absl::StrCat("test_module", execution_count++);
    se::DeviceDescription device_description;
    device_description.set_gpu_compute_capability(
        se::CudaComputeCapability::Volta());
    device_description.set_driver_version({12, 3, 0});
    device_description.set_runtime_version({12, 3, 0});
    params.device_description = device_description;
    params.enable_debug_info_manager = false;
    params.debug_module =
        std::make_unique<HloModule>(params.module_name, HloModuleConfig());
    params.debug_module->mutable_config().set_debug_options(debug_options);
    return GpuExecutable::Create(std::move(params));
  };

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          create_executable());
  const ThunkSequence& thunks = executable->GetThunk().thunks();
  EXPECT_THAT(
      thunks,
      ElementsAre(Pointee(Property(&Thunk::kind, Thunk::kCommandBuffer))));

  std::vector<std::string> dump_files;
  ASSERT_TRUE(tsl::Env::Default()
                  ->GetMatchingPaths(
                      tsl::io::JoinPath(
                          dump_dir, "*thunk_sequence_after_thunk_passes*.txt"),
                      &dump_files)
                  .ok());

  EXPECT_EQ(dump_files.size(), 1);
}

TEST(GpuExecutableTest, ComputeComputationLayout) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.program_shape.AddParameter(ShapeUtil::MakeShape(F32, {1, 2, 3}), "p0");
  params.program_shape.AddParameter(ShapeUtil::MakeShape(U8, {1}), "p1");
  *params.program_shape.mutable_result() = ShapeUtil::MakeShape(F64, {2});
  params.executable =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{}, ThunkSequence{});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          GpuExecutable::Create(std::move(params)));
  EXPECT_THAT(executable->compute_computation_layout().parameter_layouts(),
              ElementsAre(ShapeLayout(ShapeUtil::MakeShape(F32, {1, 2, 3})),
                          ShapeLayout(ShapeUtil::MakeShape(U8, {1}))));
  EXPECT_EQ(executable->compute_computation_layout().result_layout(),
            ShapeLayout(ShapeUtil::MakeShape(F64, {2})));
}

}  // namespace
}  // namespace xla::gpu
