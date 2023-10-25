/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "third_party/protobuf/text_format.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

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

void ValidateResult(
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& result) {
  ASSERT_EQ(result.size(), 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  ASSERT_EQ(result_buffers.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result_buffers[0]->ToLiteralSync());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0(2), *result_literal));
}

TEST(StreamExecutorGpuCompilerTest, SuccessAotCompileMlirAndLoad) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(true, /*allocator_config=*/{},
                                              /*node_id=*/0));
  auto se_client = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorGpuClient*>(client.release()));
  Compiler::TargetConfig gpu_target_config = xla::Compiler::TargetConfig(
      se_client->client()->backend().default_stream_executor());
  StreamExecutorGpuCompiler compiler;

  mlir::MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();
  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context);
  TF_ASSERT_OK_AND_ASSIGN(auto topology, se_client->GetTopologyDescription());
  xla::CompileOptions opts;
  opts.target_config = gpu_target_config;

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          compiler.Compile(opts, mlir_module.get(), *topology,
                                           /*client=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(auto loaded_executable,
                          se_client->Load(std::move(executable)));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      loaded_executable->Execute(/*argument_handles=*/{{}}, {}));
  ValidateResult(result);
}

TEST(StreamExecutorGpuCompilerTest, SuccessAotCompileXlaAndLoad) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(true, /*allocator_config=*/{},
                                              /*node_id=*/0));
  auto se_client = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorGpuClient*>(client.release()));
  auto gpu_compiler = gpu::NVPTXCompiler();
  Compiler::TargetConfig gpu_target_config = gpu_compiler.GetGpuTargetConfig(
      se_client->client()->backend().default_stream_executor());
  StreamExecutorGpuCompiler compiler;

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kProgram));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          se_client->GetTopologyDescription());
  xla::CompileOptions opts;
  opts.target_config = gpu_target_config;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> loaded_executable,
      se_client->Load(std::move(executable)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      loaded_executable->Execute(/*argument_handles=*/{{}}, {}));
  ValidateResult(result);
}

TEST(StreamExecutorGpuCompilerTest, SuccessLoadFromSerializedExecutable) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(true, /*allocator_config=*/{},
                                              /*node_id=*/0));
  auto se_client = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorGpuClient*>(client.release()));
  StreamExecutorGpuCompiler compiler;
  xla::CompileOptions opts;
  opts.target_config = Compiler::TargetConfig(
      se_client->client()->backend().default_stream_executor());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kProgram));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          se_client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(opts, computation, *topology, /*client=*/nullptr));

  // Serialize the executable and load it.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                          executable->SerializeExecutable());
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      se_client->LoadSerialized(serialized_executable, std::nullopt,
                                LoadOptions()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, loaded_executable->Execute(/*argument_handles=*/{{}}, {}));
  ValidateResult(result);
}

}  // namespace
}  // namespace xla
