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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;

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
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_dump_to(dump_dir.path());
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
        thunk_info,
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata()));
    thunk_sequence.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        thunk_info, slice, slice, 1024));

    GpuExecutable::Params params;
    params.executable = std::make_unique<SequentialThunk>(
        thunk_info, std::move(thunk_sequence));
    params.debug_options = debug_options;

    params.module_name = absl::StrCat("test_module", execution_count++);
    se::DeviceDescription device_description;
    device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
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
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_dir.path(),
                        "*thunk_sequence_after_thunk_passes*.txt"),
      &dump_files));

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

TEST(GpuExecutableTest, ExecutableName) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{}, ThunkSequence{});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          GpuExecutable::Create(std::move(params)));
  EXPECT_THAT(executable->name(), "test_module");
}

TEST(GpuExecutableTest, GetMlirAllocations) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{}, ThunkSequence{});

  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(0, 1024, 0);
  allocations.emplace_back(1, 2048, 0);

  const BufferAllocation* expected_ptr0 = &allocations[0];
  const BufferAllocation* expected_ptr1 = &allocations[1];

  params.mlir_allocations = std::move(allocations);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          GpuExecutable::Create(std::move(params)));

  // The pointers must match exactly because the allocations may have Slice
  // objects which hold pointers to the parent allocations.
  EXPECT_THAT(executable->GetAllocations(),
              ElementsAre(expected_ptr0, expected_ptr1));
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
MakeNonEmptyBufferAssignment() {
  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      a = f32[128] parameter(0)
      b = f32[128] parameter(1)
      ROOT c = f32[128] add(a, b)
    })";
  TF_ASSIGN_OR_RETURN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));

  AliasInfo alias_info;
  TF_ASSIGN_OR_RETURN(
      auto buffer_assignment,
      BufferAssigner::Run(
          hlo.get(), std::make_unique<DependencyHloOrdering>(hlo.get()),
          [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
          },
          &alias_info, [](LogicalBuffer::Color) { return /*alignment=*/1; }));
  EXPECT_FALSE(buffer_assignment->Allocations().empty());
  return buffer_assignment;
}

TEST(GpuExecutableTest, GetBufferAssignmentAllocations) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                          MakeNonEmptyBufferAssignment());

  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{}, ThunkSequence{});

  std::vector<const BufferAllocation*> expected_allocs;
  expected_allocs.reserve(buffer_assignment->Allocations().size());
  for (const auto& alloc : buffer_assignment->Allocations()) {
    expected_allocs.push_back(&alloc);
  }

  params.buffer_assignment = std::move(buffer_assignment);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          GpuExecutable::Create(std::move(params)));

  // The pointers must match exactly because the allocations may have Slice
  // objects which hold pointers to the parent allocations.
  EXPECT_THAT(executable->GetAllocations(), ElementsAreArray(expected_allocs));
}

TEST(GpuExecutableTest, MlirAllocationsArePreferred) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                          MakeNonEmptyBufferAssignment());

  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{}, ThunkSequence{});

  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(0, 1024, 0);
  allocations.emplace_back(1, 2048, 0);

  const BufferAllocation* expected_ptr0 = &allocations[0];
  const BufferAllocation* expected_ptr1 = &allocations[1];

  params.buffer_assignment = std::move(buffer_assignment);
  params.mlir_allocations = std::move(allocations);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                          GpuExecutable::Create(std::move(params)));

  // Expect that the allocations from mlir_allocations are returned.
  EXPECT_THAT(executable->GetAllocations(),
              ElementsAre(expected_ptr0, expected_ptr1));
}

TEST(GpuExecutableTest, ThunkChecksumPassAddsAllocation) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice slice(&alloc, 0, 1024);

  // Set up a thunk graph with a kernel that has some buffers that should be
  // checked, otherwise the pass is a no-op and doesn't need to allocate.
  auto make_test_thunk_sequence = [&]() {
    Thunk::ThunkInfo thunk_info;
    ThunkSequence thunk_sequence;
    thunk_sequence.push_back(std::make_unique<KernelThunk>(
        thunk_info,
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/
        emitters::KernelArguments({
            emitters::KernelArgument(
                ShapeUtil::MakeShape(F32, /*dimensions=*/{16}), slice),
        }),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata()));
    return thunk_sequence;
  };
  auto make_test_hlo_module = []() {
    HloComputation::Builder builder("test_computation");
    HloInstruction* root = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(1)));
    auto hlo_module =
        std::make_unique<HloModule>("test_module", HloModuleConfig());
    hlo_module->AddEntryComputation(builder.Build(/*root_instruction=*/root));
    return hlo_module;
  };

  GpuExecutable::Params params_without_pass;
  params_without_pass.debug_module = make_test_hlo_module();
  params_without_pass.executable = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo{}, make_test_thunk_sequence());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable_without_pass,
      GpuExecutable::Create(std::move(params_without_pass)));
  size_t allocations_without_pass =
      executable_without_pass->GetAllocations().size();

  GpuExecutable::Params params_with_pass;
  params_with_pass.debug_module = make_test_hlo_module();
  params_with_pass.executable = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo{}, make_test_thunk_sequence());
  params_with_pass.debug_options
      .set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(true);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable_with_pass,
                          GpuExecutable::Create(std::move(params_with_pass)));
  EXPECT_EQ(executable_with_pass->GetAllocations().size(),
            allocations_without_pass + 1);
}

TEST(GpuExecutableTest, DumpsMetadataListProto) {
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_dump_to(dump_dir.path());

  int execution_count = 0;
  auto create_executable = [&]() {
    Thunk::ThunkInfo thunk_info;
    thunk_info.thunk_id = 123;
    BufferAllocation alloc(0, 1024, 0);
    BufferAllocation::Slice slice(&alloc, 0, 1024);

    ThunkSequence thunk_sequence;
    thunk_sequence.push_back(std::make_unique<KernelThunk>(
        thunk_info,
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata()));
    thunk_info.thunk_id = 456;
    thunk_sequence.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        thunk_info, slice, slice, 1024));

    GpuExecutable::Params params;
    thunk_info.thunk_id = 789;
    params.executable = std::make_unique<SequentialThunk>(
        thunk_info, std::move(thunk_sequence));
    params.debug_options = debug_options;

    params.module_name = absl::StrCat("test_module", execution_count++);
    se::DeviceDescription device_description;
    device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
    device_description.set_driver_version({12, 3, 0});
    device_description.set_runtime_version({12, 3, 0});
    params.device_description = device_description;
    params.enable_debug_info_manager = false;
    params.debug_module =
        std::make_unique<HloModule>(params.module_name, HloModuleConfig());
    params.debug_module->mutable_config().set_debug_options(debug_options);
    return GpuExecutable::Create(std::move(params));
  };

  TF_ASSERT_OK(create_executable());

  std::vector<std::string> dump_files;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_dir.path(), "*thunk_metadata.txt"), &dump_files));
  ASSERT_THAT(dump_files, SizeIs(1));

  ThunkMetadataListProto metadata_list_proto;
  TF_ASSERT_OK(tsl::ReadTextProto(tsl::Env::Default(), dump_files.front(),
                                  &metadata_list_proto));

  EXPECT_THAT(metadata_list_proto, EqualsProto(R"pb(
                thunk_metadata {
                  thunk_info { thunk_id: 789 }
                  thunk_kind: "kSequential"
                }
                thunk_metadata {
                  thunk_info { thunk_id: 123 }
                  thunk_kind: "kKernel"
                }
                thunk_metadata {
                  thunk_info { thunk_id: 456 }
                  thunk_kind: "kCopy"
                }
              )pb"));
}

TEST(GpuExecutableTest, ProtoConversion) {
  se::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
  device_description.set_driver_version({12, 3, 0});
  device_description.set_runtime_version({12, 3, 0});

  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = 123;

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::make_unique<KernelThunk>(
      thunk_info,
      /*kernel_name=*/"test_kernel", emitters::KernelArguments({}),
      LaunchDimensions(),
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0, se::gpu::TmaMetadata()));

  GpuExecutable::Params params;
  params.asm_text = "test_asm_text";
  params.binary = {1, 2, 3};
  params.dnn_compiled_graphs = {{"test_dnn_compiled_graph", "test_json"}};

  thunk_info.thunk_id = 456;
  params.executable =
      std::make_unique<SequentialThunk>(thunk_info, std::move(thunk_sequence));
  params.device_description = device_description;

  params.module_name = "test_module";
  params.enable_debug_info_manager = false;
  params.mlir_allocations = {BufferAllocation(0, 1024, 0)};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> reference_executable,
                          GpuExecutable::Create(std::move(params)));
  TF_ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto,
                          reference_executable->ToProto());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> reconstructed_executable,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM"));
  EXPECT_THAT(reconstructed_executable->text(), "test_asm_text");
  EXPECT_THAT(reconstructed_executable->binary(), ElementsAre(1, 2, 3));
  EXPECT_THAT(
      reconstructed_executable->dnn_compiled_graphs(),
      UnorderedElementsAre(Pair("test_dnn_compiled_graph", "test_json")));
  EXPECT_THAT(reconstructed_executable->GetThunk().thunks(),
              ElementsAre(Pointee(Property(&Thunk::kind, Thunk::kKernel))));
  EXPECT_THAT(reconstructed_executable->GetAllocations(),
              ElementsAre(Pointee(Property(&BufferAllocation::size, 1024))));
  EXPECT_THAT(reconstructed_executable->name(), "test_module");
}

}  // namespace
}  // namespace xla::gpu
