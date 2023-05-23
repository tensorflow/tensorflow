/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsFalse;

void TestError(void* out, const void** in, XlaCustomCallStatus* status) {
  static constexpr char kError[] = "test error.";
  XlaCustomCallStatusSetFailure(status, kError, sizeof(kError));
}
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(TestError);

TEST(TfrtCpuClientTest, DonationWithExecutionError) {
  constexpr char kProgram[] =
      R"(HloModule DonationWithExecutionError, input_output_alias={ {}: (0, {}, must-alias) }
ENTRY DonationWithExecutionError() -> f32[2, 2] {
    %input = f32[2, 2] parameter(0)
    %custom-call = (f32[2, 2], u8[0]) custom-call(%input), custom_call_target="TestError", api_version=API_VERSION_STATUS_RETURNING, output_to_operand_aliasing={{0}: (0, {})}
    ROOT %result = f32[2, 2] get-tuple-element(%custom-call), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->Compile(xla_computation, {}));

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
  EXPECT_THAT(result.status().message(), ::testing::HasSubstr("test error."));

  result = pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}},
                                    /*options=*/{});
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("buffer has been deleted or donated."));
}

TEST(TfrtCpuClientTest, HloSnapshot) {
  constexpr char kProgram[] = R"(
    HloModule add
    ENTRY add {
      x = f32[3,2] parameter(0)
      y = f32[3,2] parameter(1)
      ROOT add = f32[3,2] add(x, y)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
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
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
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
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
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

TEST(TfrtCpuClientTest, AsyncTransferCallsOnDone) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
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
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
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
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
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
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  EXPECT_EQ(transfer_manager->buffer_size(0), 3 * 2 * 4);
}

TEST(TfrtCpuClientTest, AsyncTransferDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({shape}, device));
  EXPECT_EQ(transfer_manager->device(), device);
}

TEST(TfrtCpuClientTest, AsyncTransferSetBufferError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));
  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  transfer_manager->SetBufferError(0, InternalError("foobar"));
  EXPECT_THAT(
      buffer->ToLiteralSync(),
      tsl::testing::StatusIs(tsl::error::INTERNAL, HasSubstr("foobar")));
}

}  // namespace
}  // namespace xla
