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

#include "xla/service/gpu/gpu_aot_compilation_result.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "riegeli/bytes/string_reader.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel_symbol_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::stream_executor::DeviceDescription;
using ::stream_executor::GpuComputeCapability;
using ::stream_executor::MockPlatform;
using ::stream_executor::MockStreamExecutor;
using ::testing::AnyOf;
using ::testing::Return;
using ::testing::ReturnRef;
using ::tsl::proto_testing::EqualsProto;

PLATFORM_DEFINE_ID(kDummyPlatformId, dummy_platform);

DeviceDescription GetDeviceDescription() {
  DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      GpuComputeCapability{::stream_executor::CudaComputeCapability::Volta()});
  device_description.set_driver_version({12, 3, 0});
  device_description.set_runtime_version({12, 3, 0});
  device_description.set_compile_time_toolkit_version({12, 3, 0});
  return device_description;
}

class GpuAotCompilationResultTest : public ::testing::Test {
 public:
  GpuAotCompilationResultTest() : device_description_(GetDeviceDescription()) {
    EXPECT_CALL(executor_, GetDeviceDescription())
        .WillRepeatedly(ReturnRef(device_description_));
    EXPECT_CALL(executor_, GetPlatform()).WillRepeatedly(Return(&platform_));
    EXPECT_CALL(platform_, Name()).WillRepeatedly(ReturnRef(platform_name_));
    EXPECT_CALL(platform_, id()).WillRepeatedly(Return(platform_id_));
  }

  void* const kCudaSymbol = reinterpret_cast<void*>(0x1234567890);

  // Creates a dummy GpuExecutableProto, the actual values don't matter much.
  absl::StatusOr<GpuExecutableProto> CreateGpuExecutableProto() {
    Thunk::ThunkInfo thunk_info;
    thunk_info.thunk_id = 123;

    ThunkSequence thunk_sequence;
    thunk_sequence.push_back(std::make_unique<KernelThunk>(
        thunk_info,
        /*kernel_name=*/"test_kernel", emitters::KernelArguments({}),
        LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0, ::stream_executor::gpu::TmaMetadata()));
    CustomKernel custom_kernel{
        "custom_kernel_name",
        stream_executor::KernelLoaderSpec::
            CreateSerializableInProcessSymbolSpec(
                "persistent_kernel_name", kCudaSymbol, "test_custom_kernel",
                /*arity=*/42),
        stream_executor::BlockDim(), stream_executor::ThreadDim(),
        /*shared_memory_bytes=*/23};
    thunk_sequence.push_back(std::make_unique<CustomKernelThunk>(
        thunk_info, custom_kernel, emitters::KernelArguments({})));

    auto hlo_module = std::make_unique<HloModule>("test_module_with_shape",
                                                  HloModuleConfig());
    auto builder = HloComputation::Builder("entry");
    auto constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
    hlo_module->AddEntryComputation(builder.Build(constant));

    GpuExecutable::Params params;
    params.debug_module = std::move(hlo_module);
    params.asm_text = "test_asm_text";
    params.binary = {1, 2, 3};
    params.dnn_compiled_graphs = {{"test_dnn_compiled_graph", "test_json"}};

    thunk_info.thunk_id = 456;
    params.executable =
        std::make_unique<ThunkExecutor>(std::move(thunk_sequence));
    params.device_description = device_description_;

    params.module_name = "test_module";
    params.enable_debug_info_manager = false;
    params.mlir_allocations = {BufferAllocation(0, 1024, 0)};
    ASSIGN_OR_RETURN(
        params.executable_abi_version,
        stream_executor::ExecutableAbiVersion::FromDeviceDescription(
            device_description_));

    params.buffer_assignment_proto =
        tsl::proto_testing::ParseTextProtoOrDie<BufferAssignmentProto>(R"pb(
          buffer_allocations { size: 1024 }
          logical_buffers { size: 1024 }
          heap_simulator_traces {
            events { kind: ALLOC }
            events { kind: FREE }
          }
        )pb");

    ASSIGN_OR_RETURN(std::unique_ptr<GpuExecutable> executable,
                     GpuExecutable::Create(std::move(params)));
    return executable->ToProto();
  }

  void EnsureCudaSymbolIsRegistered() {
    // This test has to rely on the global registry, because
    // `GpuAotCompilationResult` uses the global registry to look up symbols.
    // That means different test cases can affect each other. Therefore we check
    // if the symbol is registered, and only register it if it's not.
    stream_executor::KernelSymbolRegistry& registry =
        stream_executor::KernelSymbolRegistry::GetGlobalInstance();
    ASSERT_THAT(registry.FindSymbol("persistent_kernel_name", platform_id_),
                AnyOf(IsOkAndHolds(kCudaSymbol),
                      StatusIs(absl::StatusCode::kNotFound)));
    if (!registry.FindSymbol("persistent_kernel_name", platform_id_).ok()) {
      TF_ASSERT_OK(registry.RegisterSymbol("persistent_kernel_name",
                                           platform_id_, kCudaSymbol));
    }
  }

  DeviceDescription device_description_;
  MockStreamExecutor executor_;
  MockPlatform platform_;
  const std::string platform_name_ = "gpu";
  stream_executor::Platform::Id platform_id_ = kDummyPlatformId;
};

TEST_F(GpuAotCompilationResultTest, CreateAndSerialize) {
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto reference_executable,
                       CreateGpuExecutableProto());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuAotCompilationResult> result,
      GpuAotCompilationResult::FromProto(reference_executable));

  ASSERT_OK_AND_ASSIGN(std::string serialized_result,
                       result->SerializeAsString());

  GpuExecutableProto deserialized_executable;
  ASSERT_OK(ReadSplitProto(
      std::make_unique<riegeli::StringReader<>>(serialized_result),
      deserialized_executable));

  // Module IDs are re-created during deserialization so ignore them
  deserialized_executable.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->clear_id();
  reference_executable.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->clear_id();
  EXPECT_THAT(deserialized_executable, EqualsProto(reference_executable));
}

TEST_F(GpuAotCompilationResultTest, LoadExecutable) {
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto reference_executable,
                       CreateGpuExecutableProto());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuAotCompilationResult> result,
      GpuAotCompilationResult::FromProto(reference_executable));

  {
    ASSERT_OK_AND_ASSIGN(
        stream_executor::ExecutableAbiVersion executable_abi_version,
        result->GetExecutableAbiVersion());
    EXPECT_EQ(executable_abi_version.platform_name(), "CUDA");
    EXPECT_EQ(executable_abi_version.proto()
                  .cuda_platform_version()
                  .cuda_toolkit_version(),
              "12.3.0");
  }

  EnsureCudaSymbolIsRegistered();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*result).LoadExecutable(platform_.id(), GetDeviceDescription(),
                                        DebugOptions()));

  {
    ASSERT_OK_AND_ASSIGN(
        stream_executor::ExecutableAbiVersion executable_abi_version,
        executable->GetExecutableAbiVersion());
    EXPECT_EQ(executable_abi_version.platform_name(), "CUDA");
    EXPECT_EQ(executable_abi_version.proto()
                  .cuda_platform_version()
                  .cuda_toolkit_version(),
              "12.3.0");
  }

  auto* gpu_executable = dynamic_cast<GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_executable, nullptr) << "Executable is not a GpuExecutable.";

  ASSERT_OK_AND_ASSIGN(GpuExecutableProto executable_proto,
                       gpu_executable->ToProto());
  // HLO module is re-created from proto, and will have a new ID, so we clear
  // it for comparison purposes.
  executable_proto.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->clear_id();
  reference_executable.mutable_hlo_module_with_config()
      ->mutable_hlo_module()
      ->clear_id();
  EXPECT_THAT(executable_proto, EqualsProto(reference_executable));
}

TEST_F(GpuAotCompilationResultTest, GetCompiledMemoryStats) {
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto reference_executable,
                       CreateGpuExecutableProto());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuAotCompilationResult> result,
      GpuAotCompilationResult::FromProto(reference_executable));

  ASSERT_OK_AND_ASSIGN(CompiledMemoryStats memory_stats,
                       result->GetCompiledMemoryStats());
  EXPECT_EQ(memory_stats.peak_memory_in_bytes, 1024);
  EXPECT_EQ(memory_stats.serialized_buffer_assignment,
            reference_executable.buffer_assignment().SerializeAsString());
}

}  // namespace
}  // namespace xla::gpu
