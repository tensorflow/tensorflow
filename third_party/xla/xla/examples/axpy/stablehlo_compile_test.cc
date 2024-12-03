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

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::NotNull;

std::string GetTestProgramPath() {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "examples", "axpy",
                           "stablehlo_axpy.mlir");
}

class StableHloAxpyTest : public ::testing::Test {
 protected:
  static std::unique_ptr<PjRtClient> GetCpuClient() {
    const PJRT_Api* c_api = GetPjrtApi();
    EXPECT_THAT(c_api, NotNull());
    absl::StatusOr<std::unique_ptr<PjRtClient>> client =
        xla::WrapClientAroundCApi(c_api);
    CHECK_OK(client);
    return std::move(*client);
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateStableHloProgram(
      absl::string_view program_path) {
    // Register MLIR dialects necessary to parse our program. In our case this
    // is just the Func dialect and StableHLO.
    registry_.insert<mlir::func::FuncDialect>();
    mlir::stablehlo::registerAllDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();

    // Read StableHLO program to string.
    std::string program_string;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(), std::string(program_path), &program_string));

    std::cerr << "Loaded StableHLO program from " << program_path << ":\n"
              << program_string << std::endl;

    return mlir::parseSourceString<mlir::ModuleOp>(program_string, &context_);
  }

 private:
  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
};  // class

TEST_F(StableHloAxpyTest, UsePjrtCppWrapper) {
  std::unique_ptr<PjRtClient> client = GetCpuClient();
  EXPECT_THAT(client, NotNull());
}

TEST_F(StableHloAxpyTest, RegisterAPIAndRetrieve) {
  const char* device = "MyFancyCPUDevice";
  EXPECT_THAT(GetCApiClient(device), ::testing::Not(::tsl::testing::IsOk()));
  EXPECT_THAT(pjrt::PjrtApi(device), ::testing::Not(::tsl::testing::IsOk()));

  const PJRT_Api* c_api = GetPjrtApi();
  EXPECT_THAT(c_api, NotNull());
  TF_EXPECT_OK(pjrt::SetPjrtApi(device, c_api));

  EXPECT_THAT(GetCApiClient(device), ::tsl::testing::IsOk());
  EXPECT_THAT(pjrt::PjrtApi(device), ::tsl::testing::IsOk());
}

TEST_F(StableHloAxpyTest, CompileCPUTestProgram) {
  std::unique_ptr<PjRtClient> client = GetCpuClient();

  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> program,
                          CreateStableHloProgram(GetTestProgramPath()));

  // Use our client to compile our StableHLO program to an executable.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client->Compile(*program, CompileOptions{}));
}

TEST_F(StableHloAxpyTest, CompileAndExecuteCPUTestProgram) {
  // TODO(masonchang): Use GetCpuClient() once the C API supports
  // BufferFromHostLiteral.
  xla::CpuClientOptions options;
  options.cpu_device_count = 4;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::GetXlaPjrtCpuClient(options));

  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> program,
                          CreateStableHloProgram(GetTestProgramPath()));

  // Use our client to compile our StableHLO program to an executable.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client->Compile(*program, CompileOptions{}));

  // Create inputs to our computation.
  auto alpha_literal = xla::LiteralUtil::CreateR0<float>(3.14f);
  auto x_literal = xla::LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto y_literal =
      xla::LiteralUtil::CreateR1<float>({10.5f, 20.5f, 30.5f, 40.5f});

  std::cerr << "Computation inputs:" << std::endl;
  std::cerr << "\talpha:" << alpha_literal << std::endl;
  std::cerr << "\tx:" << x_literal << std::endl;
  std::cerr << "\ty:" << y_literal << std::endl;

  PjRtDevice* host_cpu = client->devices()[0];

  // Transfer our literals to buffers. If we were using a GPU, these buffers
  // would correspond to device memory.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> alpha,
      client->BufferFromHostLiteral(alpha_literal, host_cpu));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> x,
                          client->BufferFromHostLiteral(x_literal, host_cpu));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> y,
                          client->BufferFromHostLiteral(y_literal, host_cpu));

  // Do our computation.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> axpy_result,
      executable->Execute({{alpha.get(), x.get(), y.get()}}, /*options=*/{}));

  // Convert result buffer back to literal.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> axpy_result_literal,
                          axpy_result[0][0]->ToLiteralSync());

  // Check to make sure that our results match what we expect.
  xla::LiteralTestUtil::ExpectR1Near<float>({13.64f, 26.78f, 39.92f, 53.06f},
                                            *axpy_result_literal,
                                            xla::ErrorSpec(0.01f));
  std::cerr << "Computation output: " << *axpy_result_literal << std::endl;
}

}  // namespace
}  // namespace xla
