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
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/Version.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

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

TEST(PjRtCApiClientTest, FulfillAliasBuffer) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});
  TF_ASSERT_OK_AND_ASSIGN(
      auto alias_buffer,
      client->CreateAliasBuffer(shape, client->memory_spaces()[0]));

  // Create a buffer from host data.
  TF_ASSERT_OK_AND_ASSIGN(
      auto param,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  // Define a simple "add one" kernel.
  XlaBuilder builder("add_one");
  auto input = Parameter(&builder, 0, shape, "input");
  auto one = ConstantR0<int32_t>(&builder, 1);
  auto add = Add(input, one);
  auto computation = builder.Build(add).value();

  // Compile and load the executable.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      client->CompileAndLoad(computation, CompileOptions()));

  // Execute the kernel.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
      executable->Execute({{param.get()}}, ExecuteOptions()));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  std::unique_ptr<PjRtBuffer>& result_buffer = results[0][0];

  // Wait for the result buffer to be ready.
  TF_ASSERT_OK(result_buffer->GetReadyFuture().Await());

  // Fulfill the alias buffer with the result of the add one kernel.
  ASSERT_NE(alias_buffer.second, nullptr);
  TF_ASSERT_OK(std::move(alias_buffer.second)(result_buffer.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto alias_literal,
                          alias_buffer.first->ToLiteral().Await());

  // Expected result: data + 1
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{2, 3, 4}, {5, 6, 7}}), *alias_literal));
}

TEST(PjRtCApiClientTest, CreateErrorBuffer) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  absl::Status error = absl::InternalError("Test Error");
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});

  TF_ASSERT_OK_AND_ASSIGN(
      auto error_buffer,
      client->CreateErrorBuffer(error, shape, client->memory_spaces()[0]));

  absl::Status awaited_status = error_buffer->GetReadyFuture().Await();
  EXPECT_TRUE(absl::IsInternal(awaited_status));
  EXPECT_THAT(awaited_status.message(), HasSubstr("Test Error"));
}

TEST(PjRtCApiClientTest, ConcurrentGetReadyFuture) {
  const PJRT_Api* c_api = ::pjrt::cpu_plugin::GetCpuPjrtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          WrapClientAroundCApi(c_api));

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(
      tsl::Env::Default(), "GetReadyWithConcurrentUsage", kNumThreads);

  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});

  // Create a buffer from host data.
  TF_ASSERT_OK_AND_ASSIGN(
      auto param,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  // Define a simple "add one" kernel.
  XlaBuilder builder("add_one");
  auto input = Parameter(&builder, 0, shape, "input");
  auto one = ConstantR0<int32_t>(&builder, 1);
  auto add = Add(input, one);
  auto computation = builder.Build(add).value();

  // Compile and load the executable.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      client->CompileAndLoad(computation, CompileOptions()));
  for (size_t i = 0; i < 100; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
        executable->Execute({{param.get()}}, ExecuteOptions()));
    auto buffer = std::move(results[0][0]);

    absl::BlockingCounter blocking_counter(kNumThreads);
    for (size_t j = 0; j < kNumThreads; ++j) {
      thread_pool.Schedule([&, buffer = buffer.get()]() {
        TF_EXPECT_OK(buffer->GetReadyFuture().Await());
        blocking_counter.DecrementCount();
      });
    }
    blocking_counter.Wait();
  }
}

TEST(PjRtCApiClientTest, GetReadyFutureDeletedBuffer) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  std::vector<int32_t> data{1};
  Shape shape = ShapeUtil::MakeShape(S32, {});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  buffer->Delete();
  EXPECT_TRUE(buffer->IsDeleted());

  auto future = buffer->GetReadyFuture();
  EXPECT_THAT(future.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("deleted or donated")));
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
          client->memory_spaces()[0], /*device_layout=*/nullptr));
  std::vector<int32_t> data1{2};
  Shape shape1 = ShapeUtil::MakeShape(S32, {});
  TF_ASSERT_OK_AND_ASSIGN(
      auto param1,
      client->BufferFromHostBuffer(
          data1.data(), shape1.element_type(), shape1.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));
  XlaBuilder builder("DynamicReshape");
  auto inp_0 = Parameter(&builder, 0, shape0, "input0");
  auto inp_1 = Parameter(&builder, 1, shape1, "input1");
  std::vector<bool> dims_are_dynamic = {false, true};
  auto reshaped =
      DynamicReshape(inp_0, {inp_1, inp_1}, {2, 3}, dims_are_dynamic);
  auto computation = builder.Build(reshaped).value();
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->CompileAndLoad(computation, CompileOptions()).value();
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
  EXPECT_EQ(result_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {2, 3}, {false, true}));
  EXPECT_EQ(*result_buffer->logical_on_device_shape(),
            ShapeUtil::MakeShape(S32, {2, 2}, {false, true}));
}

TEST(PjRtCApiClientTest, OnDeviceShape) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  for (PrimitiveType t : {F32, F16, S8, BF16}) {
    Shape shape = ShapeUtil::MakeShape(t, {3, 2});
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        client->BufferFromHostBuffer(
            data.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
            client->memory_spaces()[0], /*device_layout=*/nullptr));
    EXPECT_EQ(buffer->on_device_shape(), shape);
    EXPECT_EQ(*buffer->logical_on_device_shape(), shape);
  }
}

TEST(PjRtCApiClientTest, ClientPlatformIdAndName) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  EXPECT_EQ(client->platform_name(), xla::CpuName());
  EXPECT_EQ(client->platform_id(), xla::CpuId());
}

TEST(PjRtCApiClientTest, TopologyPlatformIdAndName) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());

  ASSERT_NE(topology, nullptr);
  EXPECT_EQ(topology->platform_name(), xla::CpuName());
  EXPECT_EQ(topology->platform_id(), xla::CpuId());
}

TEST(PjRtCApiClientTest, TopologyGetDefaultLayout) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  ASSERT_NE(topology, nullptr);

  std::vector<int64_t> dims = {2, 3, 4};
  TF_ASSERT_OK_AND_ASSIGN(Layout layout,
                          topology->GetDefaultLayout(PrimitiveType::F32, dims));

  Layout expected_layout = LayoutUtil::MakeDescendingLayout(dims.size());
  EXPECT_EQ(layout, expected_layout);
}

TEST(PjRtCApiClientTest, NonEmptyExecutableFingerprint) {
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
      client->CompileAndLoad(computation, CompileOptions()).value();

  PjRtCApiClient* c_client = dynamic_cast<PjRtCApiClient*>(client.get());
  ASSERT_NE(c_client, nullptr);
  if (c_client->pjrt_c_api()->pjrt_api_version.minor_version >= 58) {
    EXPECT_TRUE(executable->FingerprintExecutable().ok());
  } else if (c_client->pjrt_c_api()->pjrt_api_version.minor_version >= 35) {
    EXPECT_FALSE(executable->FingerprintExecutable().ok());
  } else {
    // TODO(yeounoh): To be removed after 01/20/2024.
    EXPECT_EQ(executable->FingerprintExecutable().status().code(),
              absl::StatusCode::kUnimplemented);
  }
}

TEST(PjRtCApiClientTest, GetCompileOptions) {
  SetUpCpuPjRtApi();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetCApiClient("cpu"));
  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  XlaBuilder builder("sum");
  auto inp_0 = Parameter(&builder, 0, shape, "input0");
  auto inp_1 = Parameter(&builder, 1, shape, "input1");
  auto sum = Add(inp_0, inp_1);
  builder.SetUpAlias({}, 0, {});
  auto computation = builder.Build(sum).value();

  CompileOptions options;
  options.compile_portable_executable = !options.compile_portable_executable;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                       client->CompileAndLoad(computation, options));

  ASSERT_OK_AND_ASSIGN(CompileOptions retrieved_options,
                       executable->GetCompileOptions());
  EXPECT_EQ(retrieved_options.compile_portable_executable,
            options.compile_portable_executable);
}

TEST(PjRtCApiClientTest, CreateBuffersForAsyncHostToDeviceWithShape) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  xla::Shape host_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, /*dimensions=*/{2, 2, 2},
      /*minor_to_major=*/{1, 0, 2});
  std::vector<xla::Shape> host_shapes = {host_shape};
  auto status_or_transfer_manager = client->CreateBuffersForAsyncHostToDevice(
      absl::MakeSpan(host_shapes), client->memory_spaces()[0]);
  EXPECT_TRUE(status_or_transfer_manager.ok())
      << status_or_transfer_manager.status();
}

TEST(PjRtClientTest, CreateViewAndCopyToDeviceAsyncExternalCpuOnly) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  ASSERT_GT(client->addressable_devices().size(), 1);
  alignas(cpu::MinAlign()) std::array<int32_t, 4> data;
  data.fill(0);
  auto* data_ptr = data.data();
  Shape shape = ShapeUtil::MakeShape(S32, {4});

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateViewOfDeviceBuffer(
          data_ptr, shape, client->memory_spaces()[0],
          /*on_delete_callback=*/[data = std::move(data)]() mutable {
            (void)data;
          }));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> result,
      buffer->CopyToMemorySpace(client->memory_spaces()[1]));
  buffer.reset();
  ASSERT_TRUE(result);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());

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
      client->CompileAndLoad(*module, CompileOptions()).value();
  const_cast<PJRT_Api*>(c_api)->PJRT_Client_Compile = PJRT_Client_Compile_Orig;
}

TEST(PjRtClientTest, CompileWorksInplace) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  constexpr char kProgram[] = "func.func @main() {return}";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  CompileOptions options;
  options.allow_in_place_mlir_modification = true;
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->CompileAndLoad(*module, options).value();
  EXPECT_NE(executable.get(), nullptr);
}

TEST(PjRtClientTest, CanQueryMemoryDescriptions) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices =
      topology->DeviceDescriptions();
  for (std::unique_ptr<const PjRtDeviceDescription>& device : devices) {
    for (const PjRtMemorySpaceDescription* memory : device->memory_spaces()) {
      // TODO: CPU doesn't currently have memory descriptions, so the
      //       code below doesn't get triggered yet.
      EXPECT_NE(memory, nullptr);
      EXPECT_GT(memory->kind().size(), 0);
      EXPECT_GE(memory->kind_id(), 0);
    }
  }
}

TEST(PjRtCApiClientTest, GetDeviceAssignment) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  ASSERT_GT(client->addressable_devices().size(), 1);

  XlaBuilder builder("Identity");
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});
  auto input = Parameter(&builder, 0, shape, "input");
  auto computation = builder.Build(input).value();

  DeviceAssignment device_assignment(1, 2);
  device_assignment(0, 0) = 0;
  device_assignment(0, 1) = 1;

  CompileOptions options;
  options.executable_build_options.set_device_assignment(device_assignment);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client->CompileAndLoad(computation, options));

  const DeviceAssignment& retrieved_assignment =
      executable->device_assignment();
  EXPECT_EQ(retrieved_assignment.replica_count(), 1);
  EXPECT_EQ(retrieved_assignment.computation_count(), 2);
  EXPECT_EQ(retrieved_assignment(0, 0), 0);
  EXPECT_EQ(retrieved_assignment(0, 1), 1);
}

TEST(PjRtCApiClientTest, WrapClientAroundCApi) {
  const PJRT_Api* c_api = ::pjrt::cpu_plugin::GetCpuPjrtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          WrapClientAroundCApi(c_api));
  EXPECT_EQ(client->platform_name(), xla::CpuName());
  EXPECT_EQ(client->platform_id(), xla::CpuId());
}

// User-defined data type to be passed to FFI handler via the execute context
// side channel.
struct MemsetValue {
  explicit MemsetValue(float value) : value(value) {}
  float value;
};

static absl::Status MemsetFromValue(
    ffi::Result<ffi::BufferR1<PrimitiveType::F32>> result,
    MemsetValue* memset_value) {
  for (size_t i = 0; i < result->element_count(); ++i) {
    result->typed_data()[i] = memset_value->value;
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromValue, MemsetFromValue,
                       ffi::Ffi::Bind()
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>()
                           .Ctx<ffi::UserData<MemsetValue>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromValue", "HOST",
                         kMemsetFromValue);

TEST(PjRtCApiClientTest, ForwardExecuteContext) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  const PJRT_Api* c_api = ::pjrt::cpu_plugin::GetCpuPjrtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          WrapClientAroundCApi(c_api));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(XlaComputation(hlo_module->ToProto()), {}));

  ExecuteContext context;
  TF_ASSERT_OK(context.ffi_context().Emplace<MemsetValue>(42.0f));

  ExecuteOptions options;
  options.context = &context;

  auto result = executable->Execute(/*argument_handles=*/{{}}, options);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result->at(0).at(0)->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({42.0f, 42.0f, 42.0f, 42.0f}),
      *result_literal));
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
      client->CompileAndLoad(computation, compile_options_for_device(0))
          .value();
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                          executable->SerializeExecutable());

  // Deserialize the executable for device 1.
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_executable,
      client->LoadSerializedExecutable(
          serialized_executable, compile_options_for_device(1), LoadOptions{}));

  // Check that the executable's compile options were overridden
  // with device id 1.
  EXPECT_EQ(
      deserialized_executable->addressable_devices()[0]->global_device_id(), 1);
}

TEST(PjRtCApiClientTest, GetOutputShapes) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  const PJRT_Api* c_api = ::pjrt::cpu_plugin::GetCpuPjrtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          WrapClientAroundCApi(c_api));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(XlaComputation(hlo_module->ToProto()), {}));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Shape> output_shapes,
                          executable->GetOutputShapes());
  EXPECT_EQ(output_shapes.size(), 1);
  Shape expected_shape = ShapeUtil::MakeShape(F32, {4});
  EXPECT_EQ(output_shapes[0], expected_shape);
}

TEST(PjRtCApiClientTest, GetParameterAndOutputShardings) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  XlaBuilder builder("sum");
  auto inp_0 = Parameter(&builder, 0, shape, "input0");
  auto inp_1 = Parameter(&builder, 1, shape, "input1");
  auto sum = Add(inp_0, inp_1);
  auto computation = builder.Build(sum).value();

  CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client->CompileAndLoad(computation, options));

  // CPU usually returns nullopt for shardings if not explicitly set.
  auto parameter_shardings = executable->GetParameterShardings();
  auto output_shardings = executable->GetOutputShardings();

  // We expect them to be either nullopt or some default shardings.
  // For CPU with default options, they are typically nullopt.
  if (parameter_shardings.has_value()) {
    EXPECT_EQ(parameter_shardings->size(), 2);
  }
  if (output_shardings.has_value()) {
    EXPECT_EQ(output_shardings->size(), 1);
  }
}

TEST(PjRtCApiClientTest, GetParameterAndOutputLayouts) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  XlaBuilder builder("sum");
  auto inp_0 = Parameter(&builder, 0, shape, "input0");
  auto inp_1 = Parameter(&builder, 1, shape, "input1");
  auto sum = Add(inp_0, inp_1);
  auto computation = builder.Build(sum).value();

  CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client->CompileAndLoad(computation, options));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> parameter_layouts,
      executable->GetParameterLayouts());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> output_layouts,
      executable->GetOutputLayouts());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::Layout expected_layout,
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));

  // We expect parameter and output layouts to be a default layout because we
  // didn't specify any layouts in the HLO.
  EXPECT_EQ(parameter_layouts.size(), 2);
  for (const std::shared_ptr<const PjRtLayout>& parameter_layout :
       parameter_layouts) {
    EXPECT_EQ(parameter_layout->xla_layout(), expected_layout);
  }
  EXPECT_EQ(output_layouts.size(), 1);
  for (const std::shared_ptr<const PjRtLayout>& output_layout :
       output_layouts) {
    EXPECT_EQ(output_layout->xla_layout(), expected_layout);
  }
}

TEST(PjRtClientTest, BufferFromLiteralInt4) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  xla::Shape shape = xla::ShapeUtil::MakeShape(S4, {128, 256});
  TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteral().Await());
  EXPECT_THAT(received_literal->data<s4>(),
              ElementsAreArray(literal.data<s4>()));
}

TEST(PjRtCApiClientTest, AsyncHostToDeviceTransferManagerTransferLiteral) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithType<int32_t>({4});
  std::vector<int32_t> data = {1, 2, 3, 4};
  xla::Literal literal = xla::LiteralUtil::CreateR1<int32_t>(data);

  std::vector<xla::Shape> host_shapes = {shape};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(absl::MakeSpan(host_shapes),
                                                client->memory_spaces()[0]));

  xla::Future<> future = transfer_manager->TransferLiteralToBuffer(
      /*buffer_index=*/0, literal, /*on_done=*/[]() {});
  TF_ASSERT_OK(future.Await());

  std::unique_ptr<PjRtBuffer> buffer =
      transfer_manager->RetrieveBuffer(/*buffer_index=*/0);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          buffer->ToLiteral().Await());

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, *result_literal));
}

TEST(PjRtCApiClientTest, CopyRawToHostFuture) {
  SetUpCpuPjRtApi();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetCApiClient("cpu"));
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> recv_data(4);
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));
  auto [dst_promise, dst_future] = MakePromise<void*>();
  ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  auto result = buffer->CopyRawToHostFuture(dst_future, 0, size);

  // Fulfill the promise with a valid host buffer.
  dst_promise.Set(recv_data.data());
  EXPECT_OK(result.Await());
  ASSERT_EQ(recv_data.size(), data.size());
  EXPECT_THAT(recv_data, ElementsAreArray(data));

  // Test error case.
  auto [error_dst_promise, error_dst_future] = MakePromise<void*>();
  result = buffer->CopyRawToHostFuture(error_dst_future, 0, size);
  error_dst_promise.Set(absl::InternalError("Future error"));
  absl::Status status = result.Await();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, "Future error"));
}

TEST(PjRtCApiClientTest, PoisonExecution) {
  SetUpCpuPjRtApi();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetCApiClient("cpu"));

  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnUnverifiedModule(R"(
HloModule Identity
ENTRY Identity() -> f32[2, 2] {
    ROOT %result = f32[2, 2] parameter(0)
})",
                                                                       {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                       client->CompileAndLoad(xla_computation, {}));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {shape}, client->memory_spaces()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  const int32_t kLaunchId = 123;
  ExecuteOptions opts;
  opts.launch_id = kLaunchId;
  // PoisonExecution only works for asynchronous executions. Synchronous
  // executions are executed inline and will not be poisonable.
  opts.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;

  auto result =
      pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}}, opts);
  ASSERT_OK(result);

  // Poisoning the execution should succeed because the execution has not
  // started with the input buffer not defined yet.
  auto poison_result = client->addressable_devices().front()->PoisonExecution(
      kLaunchId, Internal("foobar1"));
  ASSERT_THAT(poison_result, IsOkAndHolds(true));

  // The buffer is expected to be poisoned with the error.
  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->at(0).size(), 1);
  EXPECT_THAT(result->at(0).at(0)->ToLiteral().Await(),
              StatusIs(tsl::error::INTERNAL, HasSubstr("foobar1")));

  // A later error (propagated from the input buffer) would not affect the
  // already poisoned output buffer.
  transfer_manager->SetBufferError(0, Internal("foobar2"));

  EXPECT_THAT(result->at(0).at(0)->ToLiteral().Await(),
              StatusIs(tsl::error::INTERNAL, HasSubstr("foobar1")));

  // Attempting to poison a non-existent execution should fail.
  poison_result = client->addressable_devices().front()->PoisonExecution(
      kLaunchId + 12, Internal("foobar3"));
  EXPECT_THAT(poison_result, IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
