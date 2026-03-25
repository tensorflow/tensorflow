/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"

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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mock_pjrt_client.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/mock_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Optional;
using ::testing::Property;
using ::testing::Return;

constexpr absl::string_view kFakeDeviceName = "Fake_device";

constexpr absl::string_view kProgram = R"(HloModule Computation

ENTRY Computation() -> s32[] {
  ROOT result = s32[] constant(2)
})";

constexpr absl::string_view mlir_str = R"mlir(
  module {
    func.func @main() -> tensor<i32> {
      %0 = mhlo.constant dense<2> : tensor<i32>
      return %0 : tensor<i32>
    }
  })mlir";

absl::StatusOr<xla::XlaComputation> GetXlaComputation(
    absl::string_view program) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      xla::ParseAndReturnUnverifiedModule(program, {}));

  return XlaComputation(hlo_module->ToProto());
}

std::shared_ptr<xla::GpuTopology> GetGpuTopology(
    absl::string_view platform_version, int num_partitions,
    int num_hosts_per_partition, int num_devices_per_host,
    int core_count_per_chip) {
  return std::make_shared<xla::GpuTopology>(platform_version, num_partitions,
                                            num_hosts_per_partition,
                                            num_devices_per_host);
}

TEST(StreamExecutorGpuCompilerTest, NoClientXla) {
  StreamExecutorGpuCompiler compiler(CudaId());
  StreamExecutorGpuTopologyDescription topology(
      CudaId(), CudaName(), GetGpuTopology(kFakeDeviceName, 1, 1, 2, 10));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, GetXlaComputation(kProgram));
  EXPECT_THAT(compiler.Compile(xla::CompileOptions(), computation, topology,
                               /*client=*/nullptr),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(StreamExecutorGpuCompilerTest, TopologyNotSameXla) {
  StreamExecutorGpuCompiler compiler(CudaId());
  StreamExecutorGpuTopologyDescription topology(
      CudaId(), CudaName(), GetGpuTopology(kFakeDeviceName, 1, 1, 2, 10));

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto computation, GetXlaComputation(kProgram));
  EXPECT_THAT(compiler.Compile(xla::CompileOptions(), computation, topology,
                               client.get()),
              absl_testing::StatusIs(absl::StatusCode::kOk));
}

TEST(StreamExecutorGpuCompilerTest, SuccessXla) {
  StreamExecutorGpuCompiler compiler(CudaId());

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto computation, GetXlaComputation(kProgram));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      client->CompileAndLoad(computation, xla::CompileOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          loaded_executable->Execute(
                              /*argument_handles=*/{{}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  ASSERT_EQ(result_buffers.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result_buffers[0]->ToLiteral().Await());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0(2), *result_literal));
}

TEST(StreamExecutorGpuCompilerTest, NoClientMlir) {
  StreamExecutorGpuCompiler compiler(CudaId());

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();

  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, context.get());

  StreamExecutorGpuTopologyDescription topology(
      CudaId(), CudaName(), GetGpuTopology(kFakeDeviceName, 1, 1, 2, 10));

  EXPECT_THAT(compiler.Compile(xla::CompileOptions(),
                               MaybeOwningMlirModule(std::move(context),
                                                     std::move(mlir_module)),
                               topology,
                               /*client=*/nullptr),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(StreamExecutorGpuCompilerTest, TopologyNotSameMlir) {
  StreamExecutorGpuCompiler compiler(CudaId());

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();

  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, context.get());

  StreamExecutorGpuTopologyDescription topology(
      CudaId(), CudaName(), GetGpuTopology(kFakeDeviceName, 1, 1, 2, 10));

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  EXPECT_THAT(compiler.Compile(xla::CompileOptions(),
                               MaybeOwningMlirModule(std::move(context),
                                                     std::move(mlir_module)),
                               topology, client.get()),
              absl_testing::StatusIs(absl::StatusCode::kOk));
}

TEST(StreamExecutorGpuCompilerTest, SuccessMlir) {
  StreamExecutorGpuCompiler compiler(CudaId());

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();

  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, context.get());

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(mlir_module)),
          xla::CompileOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          loaded_executable->Execute(
                              /*argument_handles=*/{{}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  ASSERT_EQ(result_buffers.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result_buffers[0]->ToLiteral().Await());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0(2), *result_literal));
}

TEST(StreamExecutorGpuCompilerTest, SuccessMlirCanBeSerialized) {
  StreamExecutorGpuCompiler compiler(CudaId());

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();

  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, context.get());

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  StreamExecutorGpuTopologyDescription topology(
      CudaId(), CudaName(), GetGpuTopology(kFakeDeviceName, 1, 1, 2, 10));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtExecutable> executable,
      compiler.Compile(
          xla::CompileOptions(),
          MaybeOwningMlirModule(std::move(context), std::move(mlir_module)),
          topology, client.get()));

  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          executable->SerializeExecutable());
  ASSERT_FALSE(serialized.empty());

  TF_ASSERT_OK_AND_ASSIGN(auto loaded_executable_from_serialized,
                          client->LoadSerializedExecutable(
                              serialized, std::nullopt, xla::LoadOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          loaded_executable_from_serialized->Execute(
                              /*argument_handles=*/{{}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  ASSERT_EQ(result_buffers.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result_buffers[0]->ToLiteral().Await());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0(2), *result_literal));
}

TEST(StreamExecutorGpuCompilerTest, PlatformId) {
  constexpr PjRtPlatformId kPlatformId = PjRtPlatformId(1234);
  StreamExecutorGpuCompiler compiler(kPlatformId);
  EXPECT_EQ(compiler.pjrt_platform_id(), kPlatformId);
}

TEST(StreamExecutorGpuCompilerTest, GetTargetRuntimeAbiVersion) {
  absl::StatusOr<std::string> platform_name =
      xla::PlatformUtil::CanonicalPlatformName("gpu");
  if (!platform_name.ok() || *platform_name != "cuda") {
    GTEST_SKIP() << "GetTargetRuntimeAbiVersion is only supported for CUDA.";
  }

  // We compile a minimal program and check whether the executable's ABI version
  // is compatible with the runtime's ABI version.
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(auto computation, GetXlaComputation(kProgram));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      client->CompileAndLoad(computation, xla::CompileOptions()));

  StreamExecutorGpuCompiler compiler(CudaId());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtRuntimeAbiVersion> runtime_abi_version,
      compiler.GetTargetRuntimeAbiVersion());
  EXPECT_EQ(runtime_abi_version->platform_id(), CudaId());

  ASSERT_OK_AND_ASSIGN(auto executable_abi_version,
                       loaded_executable->GetExecutable()->GetAbiVersion());
  EXPECT_OK(runtime_abi_version->IsCompatibleWith(*executable_abi_version));
}

TEST(StreamExecutorGpuCompilerTest, JITCompilation) {
  auto mock_compiler = std::make_unique<MockCompiler>();
  MockCompiler& mock_compiler_ref = *mock_compiler;

  StreamExecutorGpuCompiler pjrt_compiler(CudaId(), std::move(mock_compiler));

  // We create a GPU topology without a target config to indicate that we
  // want JIT compilation.
  auto gpu_topology =
      std::make_shared<GpuTopology>(/*platform_version=*/"",
                                    /*num_partitions=*/1,
                                    /*num_hosts_per_partition=*/1,
                                    /*num_devices_per_host=*/1,
                                    /*gpu_target_config=*/std::nullopt);
  StreamExecutorGpuTopologyDescription topology_description(
      CudaId(), CudaName(), gpu_topology);

  // We expect that the underlying compiler is not called.
  EXPECT_CALL(mock_compiler_ref, Compile).Times(0);
  EXPECT_CALL(mock_compiler_ref, CompileAheadOfTime).Times(0);

  // We expect the compilation request to be forwarded to the PjRtClient.
  MockPjRtClient mock_client;
  EXPECT_CALL(mock_client, platform_id).WillRepeatedly(Return(CudaId()));
  EXPECT_CALL(mock_client, Compile)
      .Times(1)
      .WillOnce(Return(std::unique_ptr<PjRtExecutable>(nullptr)));

  ASSERT_OK_AND_ASSIGN(XlaComputation computation, GetXlaComputation(kProgram));
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      pjrt_compiler.Compile(CompileOptions(), computation, topology_description,
                            &mock_client));
  EXPECT_EQ(executable, nullptr);
}

TEST(StreamExecutorGpuCompilerTest, DevicelessCompilation) {
  auto mock_compiler = std::make_unique<MockCompiler>();
  MockCompiler& mock_compiler_ref = *mock_compiler;

  StreamExecutorGpuCompiler pjrt_compiler(CudaId(), std::move(mock_compiler));

  ASSERT_OK_AND_ASSIGN(auto gpu_target_config_proto,
                       gpu::GetGpuTargetConfig(gpu::GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto gpu_target_config, gpu::GpuTargetConfig::FromProto(
                                                   gpu_target_config_proto));
  auto gpu_topology = std::make_shared<GpuTopology>(
      GetSingleDeviceGpuTopology(CudaName(), gpu_target_config));

  StreamExecutorGpuTopologyDescription topology_description(
      CudaId(), CudaName(), gpu_topology);

  // We expect that the underlying compiler is called with no executor given.
  EXPECT_CALL(mock_compiler_ref, PlatformId)
      .WillRepeatedly(Return(stream_executor::cuda::kCudaPlatformId));
  EXPECT_CALL(mock_compiler_ref, Compile).Times(0);
  EXPECT_CALL(mock_compiler_ref,
              CompileAheadOfTime(
                  _, Property(&AotCompilationOptions::executor, IsNull())))
      .WillOnce(Return(std::vector<std::unique_ptr<CompiledModule>>{}));

  ASSERT_OK_AND_ASSIGN(XlaComputation computation, GetXlaComputation(kProgram));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      pjrt_compiler.Compile(CompileOptions(), computation, topology_description,
                            /*client=*/nullptr));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(IsEmpty()));
}

TEST(StreamExecutorGpuCompilerTest, CrossCompilation) {
  auto mock_compiler = std::make_unique<MockCompiler>();
  MockCompiler& mock_compiler_ref = *mock_compiler;

  StreamExecutorGpuCompiler pjrt_compiler(CudaId(), std::move(mock_compiler));

  ASSERT_OK_AND_ASSIGN(auto gpu_target_config_proto,
                       gpu::GetGpuTargetConfig(gpu::GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto gpu_target_config, gpu::GpuTargetConfig::FromProto(
                                                   gpu_target_config_proto));
  cpu::TargetMachineOptions host_target_machine_options(
      "some_triple", "some_cpu", "+some_feature,-some_other_feature");
  auto gpu_topology = std::make_shared<GpuTopology>(GetSingleDeviceGpuTopology(
      CudaName(), gpu_target_config, host_target_machine_options));

  StreamExecutorGpuTopologyDescription topology_description(
      CudaId(), CudaName(), gpu_topology);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetStreamExecutorGpuClient(GpuClientOptions()));
  auto se_gpu_client = dynamic_cast<StreamExecutorGpuClient*>(client.get());
  ASSERT_NE(se_gpu_client, nullptr);

  se::StreamExecutor* stream_executor =
      se_gpu_client->client()->backend().default_stream_executor();

  // We expect that the underlying compiler is called with the executor from the
  // PjRt client.
  EXPECT_CALL(mock_compiler_ref, PlatformId)
      .WillRepeatedly(Return(stream_executor::cuda::kCudaPlatformId));
  EXPECT_CALL(mock_compiler_ref, Compile).Times(0);
  EXPECT_CALL(
      mock_compiler_ref,
      CompileAheadOfTime(
          _, ::testing::AllOf(
                 Property(&AotCompilationOptions::executor, stream_executor),
                 Property(&AotCompilationOptions::gpu_topology,
                          Optional(Property(
                              &GpuTopology::host_target_machine_options,
                              Optional(host_target_machine_options)))))))
      .WillOnce(Return(std::vector<std::unique_ptr<CompiledModule>>{}));

  ASSERT_OK_AND_ASSIGN(XlaComputation computation, GetXlaComputation(kProgram));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      pjrt_compiler.Compile(CompileOptions(), computation, topology_description,
                            client.get()));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(IsEmpty()));
}

}  // namespace
}  // namespace xla
