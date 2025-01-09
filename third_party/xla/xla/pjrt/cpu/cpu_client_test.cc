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

#include "xla/pjrt/cpu/cpu_client.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsFalse;
using ::tsl::testing::IsOkAndHolds;

static absl::Status TestError(ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer>,
                              ffi::Result<ffi::AnyBuffer>) {
  return absl::InternalError("test error.");
}

XLA_FFI_DEFINE_HANDLER(kTestError, TestError,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()  // in
                           .Ret<ffi::AnyBuffer>()  // out0
                           .Ret<ffi::AnyBuffer>()  // out1
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$TestError", "Host",
                         kTestError);

TEST(TfrtCpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  ASSERT_EQ(client->memory_spaces().size(),
            client->addressable_devices().size());
  for (auto* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    EXPECT_THAT(device->memory_spaces(), ElementsAre(memory_space));
    EXPECT_EQ(memory_space->kind(), UnpinnedHostMemorySpace::kKind);
    EXPECT_EQ(memory_space->kind_id(), UnpinnedHostMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(UnpinnedHostMemorySpace::kKind),
                IsOkAndHolds(memory_space));
  }
}

TEST(TfrtCpuClientTest, DonationWithExecutionError) {
  static constexpr char kProgram[] =
      R"(
HloModule DonationWithExecutionError,
          input_output_alias={ {}: (0, {}, must-alias) }

ENTRY DonationWithExecutionError() -> f32[2, 2] {
    %input = f32[2, 2] parameter(0)
    %custom-call = (f32[2, 2], u8[0]) custom-call(%input),
                      custom_call_target="__xla_test$$TestError",
                      api_version=API_VERSION_TYPED_FFI,
                      output_to_operand_aliasing={{0}: (0, {})}
    ROOT %result = f32[2, 2] get-tuple-element(%custom-call), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->Compile(xla_computation, {}));

  TF_ASSERT_OK_AND_ASSIGN(auto fingerprint,
                          pjrt_executable->FingerprintExecutable());
  ASSERT_TRUE(!fingerprint.empty());

  std::vector<float> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));

  auto result = pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}},
                                         /*options=*/{});
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(), HasSubstr("test error."));

  result = pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}},
                                    /*options=*/{});
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("buffer has been deleted or donated."));
}

TEST(TfrtCpuClientTest, HloSnapshot) {
  static constexpr char kProgram[] = R"(
    HloModule add
    ENTRY add {
      x = f32[3,2] parameter(0)
      y = f32[3,2] parameter(1)
      ROOT add = f32[3,2] add(x, y)
    })";

  CpuClientOptions cpu_options;
  cpu_options.cpu_device_count = 1;
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetTfrtCpuClient(std::move(cpu_options)));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));

  std::string dir = tsl::testing::TmpDir();
  xla::CompileOptions options;
  auto* debug_opts = options.executable_build_options.mutable_debug_options();
  debug_opts->set_xla_dump_to(dir);
  debug_opts->set_xla_dump_hlo_snapshots(true);
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->Compile(xla_computation, options));

  std::vector<float> data1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<float> data2{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer1,
      client->BufferFromHostBuffer(
          data1.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer2,
      client->BufferFromHostBuffer(
          data2.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));

  auto result = pjrt_executable->Execute(
      /*argument_handles=*/{{buffer1.get(), buffer2.get()}},
      /*options=*/{});
  ASSERT_TRUE(result.ok());

  tsl::FileSystem* fs;
  ASSERT_TRUE(tsl::Env::Default()->GetFileSystemForFile(dir, &fs).ok());

  std::vector<std::string> paths;
  ASSERT_TRUE(fs->GetMatchingPaths(dir + "/*.snapshot.*.pb", &paths).ok());
  ASSERT_EQ(paths.size(), 1);

  HloSnapshot snapshot;
  ASSERT_TRUE(
      tsl::ReadBinaryProto(tsl::Env::Default(), paths[0], &snapshot).ok());

  ASSERT_EQ(*Literal::CreateFromProto(snapshot.arguments(0)),
            LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}));
  ASSERT_EQ(
      *Literal::CreateFromProto(snapshot.arguments(1)),
      LiteralUtil::CreateR2<float>({{10.0, 20.0}, {30.0, 40.0}, {50.0, 60.0}}));
  ASSERT_EQ(
      *Literal::CreateFromProto(snapshot.result()),
      LiteralUtil::CreateR2<float>({{11.0, 22.0}, {33.0, 44.0}, {55.0, 66.0}}));
}

TEST(TfrtCpuClientTest, AsyncTransferRawData) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  auto ready_future = buffer->GetReadyFuture();
  EXPECT_THAT(ready_future.IsReady(), IsFalse());
  constexpr size_t raw_data_size = 3 * 2 * 4;
  char raw_data[raw_data_size];
  std::fill(raw_data, raw_data + raw_data_size, 0x42);
  absl::string_view raw_data_view(raw_data, raw_data_size);
  TF_ASSERT_OK(transfer_manager->TransferRawDataToBuffer(
      0, absl::string_view(raw_data, raw_data_size), []() {}));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteralSync());
  ASSERT_EQ(literal->element_count(), 3 * 2);
  EXPECT_THAT(literal->data<uint32_t>(), Each(0x42424242));
}

TEST(TfrtCpuClientTest, AsyncTransferLiteral) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = xla::ShapeUtil::MakeShape(F32, {128, 256});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  auto ready_future = buffer->GetReadyFuture();
  EXPECT_THAT(ready_future.IsReady(), IsFalse());
  TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToBuffer(0, literal, []() {}));
  TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteralSync());
  EXPECT_THAT(received_literal->data<float>(),
              ElementsAreArray(literal.data<float>()));
}

TEST(TfrtCpuClientTest, AsyncTransferLiteralInt4) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = xla::ShapeUtil::MakeShape(S4, {128, 256});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  auto ready_future = buffer->GetReadyFuture();
  EXPECT_THAT(ready_future.IsReady(), IsFalse());
  TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToBuffer(0, literal, []() {}));
  TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteralSync());
  EXPECT_THAT(received_literal->data<s4>(),
              ElementsAreArray(literal.data<s4>()));
}

TEST(TfrtCpuClientTest, BufferFromLiteralInt4) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = xla::ShapeUtil::MakeShape(S4, {128, 256});
  TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostLiteral(literal, client->addressable_devices()[0]));
  TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteralSync());
  EXPECT_THAT(received_literal->data<s4>(),
              ElementsAreArray(literal.data<s4>()));
}

TEST(TfrtCpuClientTest, AsyncTransferCallsOnDone) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(F32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  auto ready_future = buffer->GetReadyFuture();
  EXPECT_THAT(ready_future.IsReady(), IsFalse());
  char raw_data[3 * 2 * 4] = {0};
  absl::string_view raw_data_view(raw_data, sizeof(raw_data));
  absl::Notification done;
  auto mark_done = [&]() { done.Notify(); };
  TF_ASSERT_OK(
      transfer_manager->TransferRawDataToBuffer(0, raw_data_view, mark_done));
  done.WaitForNotification();
}

TEST(TfrtCpuClientTest, AsyncTransferNeverTransferred) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  transfer_manager.reset();
  EXPECT_THAT(
      buffer->ToLiteralSync(),
      tsl::testing::StatusIs(tsl::error::INTERNAL,
                             HasSubstr("Async transfer object was deleted "
                                       "before transfers completed.")));
}

TEST(TfrtCpuClientTest, AsyncTransferBufferCount) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  EXPECT_EQ(transfer_manager->buffer_count(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      transfer_manager, client->CreateBuffersForAsyncHostToDevice(
                            {shape, shape}, client->addressable_devices()[0]));
  EXPECT_EQ(transfer_manager->buffer_count(), 2);
}

TEST(TfrtCpuClientTest, AsyncTransferBufferSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  EXPECT_EQ(transfer_manager->buffer_size(0), 3 * 2 * 4);
}

TEST(TfrtCpuClientTest, AsyncTransferDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({shape}, device));
  EXPECT_EQ(transfer_manager->device(), device);
}

TEST(TfrtCpuClientTest, AsyncTransferSetBufferError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  transfer_manager->SetBufferError(0, Internal("foobar"));
  EXPECT_THAT(
      buffer->ToLiteralSync(),
      tsl::testing::StatusIs(tsl::error::INTERNAL, HasSubstr("foobar")));
}

TEST(TfrtCpuClientTest, CreateErrorBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->CreateErrorBuffer(Internal("foobar"), shape,
                                             client->addressable_devices()[0]));
  EXPECT_THAT(
      buffer->ToLiteralSync(),
      tsl::testing::StatusIs(tsl::error::INTERNAL, HasSubstr("foobar")));
}

TEST(TfrtCpuClientTest, AsyncTransferRawDataToSubBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  auto ready_future = buffer->GetReadyFuture();
  EXPECT_THAT(ready_future.IsReady(), IsFalse());
  constexpr size_t raw_data_size = 3 * 2 * 4;
  char raw_data[raw_data_size];
  std::fill(raw_data, raw_data + raw_data_size, 0x42);
  absl::string_view raw_data_view(raw_data, raw_data_size);
  TF_ASSERT_OK(transfer_manager->TransferRawDataToSubBuffer(
      0, raw_data_view.data(), 0, raw_data_size - 1, /*is_last_transfer=*/false,
      []() {}));
  TF_ASSERT_OK(transfer_manager->TransferRawDataToSubBuffer(
      0, raw_data_view.data(), raw_data_size - 1, 1, /*is_last_transfer=*/true,
      []() {}));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteralSync());
  ASSERT_EQ(literal->element_count(), 3 * 2);
  EXPECT_THAT(literal->data<uint32_t>(), Each(0x42424242));
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

TEST(TfrtCpuClientTest, ForwardUserDataToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->Compile(xla_computation, {}));

  ExecuteContext context;
  TF_ASSERT_OK(context.ffi_context().Emplace<MemsetValue>(42.0f));

  ExecuteOptions opts;
  opts.context = &context;

  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result->at(0).at(0)->ToLiteralSync());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({42.0f, 42.0f, 42.0f, 42.0f}),
      *result_literal));
}

static absl::Status MemsetFromAttr(
    float attr, ffi::Result<ffi::BufferR1<PrimitiveType::F32>> result) {
  for (size_t i = 0; i < result->element_count(); ++i) {
    result->typed_data()[i] = attr;
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromAttr, MemsetFromAttr,
                       ffi::Ffi::Bind()
                           .Attr<float>("attr")
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromAttr", "HOST",
                         kMemsetFromAttr);

TEST(TfrtCpuClientTest, PassAttrToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
          custom_call_target="MemsetFromAttr",
          api_version=API_VERSION_TYPED_FFI,
          backend_config={"custom_call_config": {"attributes": "{attr = 3.0 : f32}"}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(CpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->Compile(xla_computation, {}));

  ExecuteOptions opts;
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result->at(0).at(0)->ToLiteralSync());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({3.0f, 3.0f, 3.0f, 3.0f}), *result_literal));
}

}  // namespace

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_CreateZeroCopyBuffer(benchmark::State& state) {
  auto client = GetTfrtCpuClient({});
  PjRtDevice* device = (*client)->devices().front();

  alignas(32) float value = 1.0f;

  for (auto _ : state) {
    auto buffer = (*client)->BufferFromHostBuffer(
        &value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr, device);
    CHECK_OK(buffer) << "Failed to create a buffer from a host buffer";
  }
}

BENCHMARK(BM_CreateZeroCopyBuffer);

}  // namespace xla
