/* Copyright 2022 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using GpuAotCompilationTest = HloTestBase;

TEST_F(GpuAotCompilationTest, ExportAndLoadExecutable) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  ROOT b = f32[100, 200]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  // Compile AOT.
  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler->CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(compiler, stream_exec));
}

TEST_F(GpuAotCompilationTest, AotCompilationWithoutGpuDevice) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  ROOT b = f32[100, 200]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));

  // Stream executor is not passed as an option.
  Compiler::TargetConfig gpu_target_config(stream_exec);
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_target_config(gpu_target_config);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler->CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(compiler, stream_exec));
}

namespace {

using ::mlir::ArrayRef;
using ::mlir::NamedAttribute;

std::string CreateTritonCustomCallBackendConfig() {
  mlir::MLIRContext context_;
  mlir::Builder builder(&context_);

  // Create the backend_config for the triton custom call.
  const std::string kMLIRText = R"(
  module {
    tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
      %0 = tt.get_program_id x : i32
      %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
      %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
      %cst = arith.constant 1.000000e+00 : f32
      %3 = arith.addf %1, %cst : f32
      tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
      tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
      tt.return
    }
  }
  )";

  NamedAttribute name =
      builder.getNamedAttr("name", builder.getStringAttr("add_one"));
  NamedAttribute ir =
      builder.getNamedAttr("ir", builder.getStringAttr(kMLIRText));
  NamedAttribute num_stages =
      builder.getNamedAttr("num_stages", builder.getI32IntegerAttr(3));
  NamedAttribute num_warps =
      builder.getNamedAttr("num_warps", builder.getI32IntegerAttr(4));
  NamedAttribute grid_x =
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(1));
  NamedAttribute grid_y =
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(1));
  NamedAttribute grid_z =
      builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(1));
  NamedAttribute debug =
      builder.getNamedAttr("debug", builder.getBoolAttr(false));

  std::vector<NamedAttribute> attributes = {
      name, ir, num_stages, num_warps, grid_x, grid_y, grid_z, debug};
  ArrayRef<NamedAttribute> attributesRef(attributes);
  mlir::DictionaryAttr backend_config =
      mlir::DictionaryAttr::get(&context_, attributesRef);

  // Parse the backend_config into a string.
  std::string backend_config_str;
  llvm::raw_string_ostream(backend_config_str) << backend_config;

  return backend_config_str;
}

}  // namespace

TEST_F(GpuAotCompilationTest, ExportAndLoadExecutableWithTriton) {
  auto triton_support =
      EnsureTritonSupportsComputeCapability(backend()
                                                .default_stream_executor()
                                                ->GetDeviceDescription()
                                                .gpu_compute_capability());
  if (!triton_support.ok()) {
    GTEST_SKIP() << triton_support;
  }

  const absl::string_view hlo_string_template = R"(
    HloModule Test

    ENTRY main {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT c = (f32[],f32[]) custom-call(a, b), custom_call_target="__gpu$xla.gpu.triton", backend_config="%s"
    }
    )";

  std::string hlo_string =
      absl::StrFormat(hlo_string_template,
                      absl::CEscape(CreateTritonCustomCallBackendConfig()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto platform_name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(platform_name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  // Compile AOT.
  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler->CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(compiler, stream_exec));

  const xla::Literal literal_1 = xla::LiteralUtil::CreateR0<float>(1.0f);
  const xla::Literal literal_2 = xla::LiteralUtil::CreateR0<float>(2.0f);
  const xla::Literal literal_3 = xla::LiteralUtil::CreateR0<float>(3.0f);

  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          GetHloRunner().value()->ExecuteWithExecutable(
                              executable.get(), {&literal_1, &literal_3}));

  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::MakeTuple({&literal_2, &literal_3}), result));
}

}  // namespace gpu
}  // namespace xla
