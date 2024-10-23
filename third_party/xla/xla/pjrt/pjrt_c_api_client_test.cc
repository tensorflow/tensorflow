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
#include "xla/pjrt/pjrt_c_api_client.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/Version.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

static void SetUpCpuPjRtApi() {
  std::string device_type = "cpu";
  auto status = ::pjrt::PjrtApi(device_type);
  if (!status.ok()) {
    TF_ASSERT_OK(
        pjrt::SetPjrtApi(device_type, ::pjrt::cpu_plugin::GetCpuPjrtApi()));
  }
}

TEST(PjRtCApiClientTest, IsDynamicDimension) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  // Prepare input buffer and executable.
  std::vector<int32_t> data0{1, 2, 3, 4, 5, 6};
  Shape shape0 = ShapeUtil::MakeShape(S32, {2, 3});
  TF_ASSERT_OK_AND_ASSIGN(
      auto param0,
      client->BufferFromHostBuffer(
          data0.data(), shape0.element_type(), shape0.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));
  std::vector<int32_t> data1{2};
  Shape shape1 = ShapeUtil::MakeShape(S32, {});
  TF_ASSERT_OK_AND_ASSIGN(
      auto param1,
      client->BufferFromHostBuffer(
          data1.data(), shape1.element_type(), shape1.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));
  XlaBuilder builder("DynamicReshape");
  auto inp_0 = Parameter(&builder, 0, shape0, "input0");
  auto inp_1 = Parameter(&builder, 1, shape1, "input1");
  std::vector<bool> dims_are_dynamic = {false, true};
  auto reshaped =
      DynamicReshape(inp_0, {inp_1, inp_1}, {2, 3}, dims_are_dynamic);
  auto computation = builder.Build(reshaped).value();
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->Compile(computation, CompileOptions()).value();
  ExecuteOptions execute_options;
  execute_options.non_donatable_input_indices = {0};
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results =
      executable->Execute({{param0.get(), param1.get()}}, execute_options)
          .value();
  ASSERT_EQ(results[0].size(), 1);
  auto* result_buffer = results[0][0].get();

  auto is_dynamic_dimension = result_buffer->is_dynamic_dimension();

  EXPECT_THAT(is_dynamic_dimension,
              ::testing::ElementsAreArray(dims_are_dynamic));
}

TEST(PjRtCApiClientTest, PlatformId) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  EXPECT_EQ(client->platform_name(), xla::CpuName());
  EXPECT_EQ(client->platform_id(), xla::CpuId());
}

TEST(PjRtCApiClientTest, EmptyExecutableFingerprint) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  XlaBuilder builder("sum");
  auto inp_0 = Parameter(&builder, 0, shape, "input0");
  auto inp_1 = Parameter(&builder, 1, shape, "input1");
  auto sum = Add(inp_0, inp_1);
  builder.SetUpAlias({}, 0, {});
  auto computation = builder.Build(sum).value();
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->Compile(computation, CompileOptions()).value();

  PjRtCApiClient* c_client = dynamic_cast<PjRtCApiClient*>(client.get());
  ASSERT_NE(c_client, nullptr);
  if (c_client->pjrt_c_api()->pjrt_api_version.minor_version >= 35) {
    // Empty executable should return an error status.
    EXPECT_FALSE(executable->FingerprintExecutable().ok());
  } else {
    // TODO(yeounoh): To be removed after 01/20/2024.
    EXPECT_EQ(executable->FingerprintExecutable().status().code(),
              absl::StatusCode::kUnimplemented);
  }
}

TEST(PjRtClientTest, CreateViewAndCopyToDeviceAsyncExternalCpuOnly) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  ASSERT_GT(client->addressable_devices().size(), 1);
  alignas(cpu_function_runtime::MinAlign()) std::array<int32_t, 4> data;
  data.fill(0);
  auto* data_ptr = data.data();
  Shape shape = ShapeUtil::MakeShape(S32, {4});

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateViewOfDeviceBuffer(
          data_ptr, shape, client->addressable_devices()[0],
          /*on_delete_callback=*/[data = std::move(data)]() mutable {
            (void)data;
          }));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> result,
      buffer->CopyToDevice(client->addressable_devices()[1]));
  buffer.reset();
  ASSERT_TRUE(result);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

  std::vector<int32_t> expected(4, 0);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(PjRtClientTest, CompileUsesStableHloVersion) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(const PJRT_Api* c_api, pjrt::PjrtApi("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  static auto PJRT_Client_Compile_Orig = c_api->PJRT_Client_Compile;
  constexpr char kProgram[] = "func.func @main() {return}";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  const_cast<PJRT_Api*>(c_api)->PJRT_Client_Compile =
      [](PJRT_Client_Compile_Args* args) -> PJRT_Error* {
    mlir::vhlo::Version version = mlir::vhlo::Version::getCurrentVersion();
    std::string version_string = absl::StrFormat(
        "%d.%d.%d", version.getMajor(), version.getMinor(), version.getPatch());
    // MLIR doesn't have any functionality for retrieving the producer of
    // bytecode files, so just scan the raw string.
    EXPECT_TRUE(llvm::StringRef(args->program->code, args->program->code_size)
                    .contains(version_string));
    return PJRT_Client_Compile_Orig(args);
  };
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->Compile(*module, CompileOptions()).value();
  const_cast<PJRT_Api*>(c_api)->PJRT_Client_Compile = PJRT_Client_Compile_Orig;
}

TEST(PjRtClientTest, DeserializeExecutableWithDifferentDeviceAssignment) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  ASSERT_GT(client->addressable_devices().size(), 1);

  XlaBuilder builder("Identity");
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});
  auto input = Parameter(&builder, 0, shape, "input");
  auto computation = builder.Build(input).value();

  auto compile_options_for_device = [](int id) -> xla::CompileOptions {
    xla::DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = id;
    xla::CompileOptions options;
    options.executable_build_options.set_device_assignment(device_assignment);
    return options;
  };

  // Compile the executable for device 0 and serialize it.
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->Compile(computation, compile_options_for_device(0)).value();
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                          executable->SerializeExecutable());

  // Deserialize the executable for device 1.
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_executable,
      client->DeserializeExecutable(serialized_executable,
                                    compile_options_for_device(1)));

  // Check that the executable's compile options were overridden
  // with device id 1.
  EXPECT_EQ(
      deserialized_executable->addressable_devices()[0]->global_device_id(), 1);
}

}  // namespace
}  // namespace xla
