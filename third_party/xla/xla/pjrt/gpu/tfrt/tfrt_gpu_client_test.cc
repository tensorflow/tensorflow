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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"

#include <stdint.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/protobuf.h"

namespace xla {

class DonationTransactionPeer {
 public:
  static absl::StatusOr<TfrtGpuBuffer::DonationTransaction> AcquireDonation(
      TfrtGpuBuffer* tfrt_buffer) {
    return tfrt_buffer->AcquireDonation();
  }
  static tsl::AsyncValueRef<bool> GetDonationEvent(TfrtGpuBuffer* tfrt_buffer) {
    return tfrt_buffer->GetDonationEvent();
  }
};

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.CompileAndLoad(xla_computation, compile_options);
}

// Given the result of a PjrtExecutable::Execute call (TF-status of vectors of
// vectors), extract the zeroth result from the zeroth device.
absl::StatusOr<std::shared_ptr<xla::Literal>> ExtractSingleResult(
    absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>&
        result) {
  TF_RETURN_IF_ERROR(result.status());
  TF_RET_CHECK(result->size() == 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = (*result)[0];
  TF_RET_CHECK(result_buffers.size() == 1);
  auto literal_or = result_buffers[0]->ToLiteralSync();
  if (!literal_or.status().ok()) return literal_or.status();
  return *literal_or;
}

static constexpr char const* kProgram = R"(HloModule HostTransfer
ENTRY SendRecvSynchronous() -> f32[2] {
  in_chain = token[] after-all()

  data = f32[2] constant({2, 3})
  send = (f32[2], u32[], token[]) send(data, in_chain),
    channel_id=1,
    is_host_transfer=true,
    frontend_attributes={
      _xla_host_transfer_handler_name="undef",
      _xla_host_transfer_rendezvous="undef"
    }
  send-done = token[] send-done(send),
    channel_id=1, is_host_transfer=true

  recv = (f32[2], u32[], token[]) recv(send-done),
    channel_id=2,
    is_host_transfer=true,
    frontend_attributes={
      _xla_host_transfer_handler_name="undef",
      _xla_host_transfer_rendezvous="undef"
    }
  recv-done = (f32[2], token[]) recv-done(recv),
    channel_id=2, is_host_transfer=true

  ROOT result = f32[2] get-tuple-element(recv-done), index=0
})";

TEST(TfrtGpuClientTest, GpuClientOptions) {
  GpuClientOptions options;
  options.platform_name = "cuda";
  options.allowed_devices = {0, 1};
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(options));
  EXPECT_THAT(client->platform_version(), HasSubstr("cuda"));
  EXPECT_EQ(client->device_count(), 2);
}

TEST(TfrtGpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    EXPECT_EQ(memory_space->kind(), TfrtGpuDeviceMemorySpace::kKind);
    EXPECT_EQ(memory_space->kind_id(), TfrtGpuDeviceMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(TfrtGpuDeviceMemorySpace::kKind),
                IsOkAndHolds(memory_space));

    EXPECT_EQ(device->memory_spaces().size(), 2);
    auto* pinned = device->memory_spaces()[1];
    EXPECT_EQ(pinned->kind_id(), PinnedHostMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(PinnedHostMemorySpace::kKind),
                IsOkAndHolds(pinned));
  }
}

TEST(TfrtGpuClientTest, MemorySpacesUniqueIds) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  absl::flat_hash_map<int, std::string> memories;
  for (auto* device : client->devices()) {
    for (auto* memory_space : device->memory_spaces()) {
      std::string debug_string(memory_space->DebugString());
      auto [it, inserted] = memories.insert({memory_space->id(), debug_string});
      EXPECT_TRUE(inserted) << "Duplicate ids for memory spaces '" << it->second
                            << "' and '" << debug_string << "'";
    }
  }
}

TEST(TfrtGpuClientTest, PropagateError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto shape = xla::ShapeUtil::MakeScalarShape(xla::F32);
  absl::Status input_error = absl::InvalidArgumentError("input error");
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateErrorBuffer(
          input_error, shape,
          *client->addressable_devices()[0]->default_memory_space()));

  static constexpr char const* kAddProgram =
      R"(
HloModule Add.6, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

ENTRY %Add.6 (a.1: f32[], b.2: f32[]) -> (f32[], f32[]) {
  %a.1 = f32[] parameter(0)
  %b.2 = f32[] parameter(1)
  %add.3 = f32[] add(f32[] %a.1, f32[] %b.2)
  %add.4 = f32[] add(f32[] %add.3, f32[] %add.3)
  ROOT %tuple.5 = (f32[], f32[]) tuple(f32[] %add.3, f32[] %add.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      executable->Execute({{buffer.get(), buffer.get()}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0]->GetReadyFuture().Await(), input_error);
}

TEST(TfrtGpuClientTest, SendRecvChunked) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  std::array<float, 2> sent_value = {0.0f, 0.0f};

  // Send buffer to host.
  SendCallback send_callback = {
      /*channel_id=*/1, [&](const PjRtTransferMetadata& m, PjRtChunk chunk,
                            int64_t total_size_in_bytes, bool done) {
        float* data = reinterpret_cast<float*>(chunk.data());
        sent_value[0] = data[0];
        sent_value[1] = data[1];
        return absl::OkStatus();
      }};

  // Recv buffer from host.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        auto chunk0 = PjRtChunk::AllocateDefault(sizeof(float));
        *reinterpret_cast<float*>(chunk0.data()) = 5.0f;
        TF_CHECK_OK(stream->AddChunk(std::move(chunk0)).Await());

        auto chunk1 = PjRtChunk::AllocateDefault(sizeof(float));
        *reinterpret_cast<float*>(chunk1.data()) = 6.0f;
        TF_CHECK_OK(stream->AddChunk(std::move(chunk1)).Await());

        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_EQ(sent_value[0], 2.0f);
  EXPECT_EQ(sent_value[1], 3.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<float>({5.0f, 6.0f}),
                                     *result_literal));
}

TEST(TfrtGpuClientTest, SendErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  // Always-failing Send handler.
  SendCallback send_callback = {
      /*channel_id=*/1,
      [&](const PjRtTransferMetadata&, PjRtChunk, int64_t, bool) {
        return Internal("Uh-oh, can send chunk to host");
      }};

  // No-op Recv handler.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  // Check that send error safely rejected and we do not dead lock.
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  EXPECT_THAT(ExtractSingleResult(result).status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Uh-oh, can send chunk to host")));
}

TEST(TfrtGpuClientTest, RecvErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  // No-op Send handler.
  SendCallback send_callback = {
      /*channel_id=*/1, [&](const PjRtTransferMetadata&, PjRtChunk, int64_t,
                            bool) { return absl::OkStatus(); }};

  // Invalid Recv handler that tries to add invalid chunk.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        auto chunk = PjRtChunk::AllocateDefault(10 * sizeof(float));
        stream->AddChunk(std::move(chunk)).Await().IgnoreError();
        // Return ok status to proceed to corresponding recv-done call.
        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  // Check that invalid chunk safely rejected and we do not dead lock.
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  EXPECT_THAT(
      ExtractSingleResult(result).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Adding chunk of size 40 would overflow buffer "
                         "of size 8 (0 already transferred)")));
}

// User-defined data type to be passed to FFI handler via the execute context
// side channel.
struct MemsetValue {
  explicit MemsetValue(float value) : value(value) {}
  float value;
};

static absl::Status MemsetFromValue(
    se::Stream* stream, ffi::Result<ffi::BufferR1<PrimitiveType::F32>> result,
    MemsetValue* memset_value) {
  uint32_t pattern;
  std::memcpy(&pattern, &memset_value->value, sizeof(pattern));

  se::DeviceMemoryBase base = result->device_memory();
  return stream->Memset32(&base, pattern, base.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromValue, MemsetFromValue,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>()
                           .Ctx<ffi::UserData<MemsetValue>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromValue",
                         PlatformUtil::CanonicalPlatformName("GPU").value(),
                         kMemsetFromValue);

TEST(TfrtGpuClientTest, ForwardUserDataToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  ExecuteContext context;
  TF_ASSERT_OK(context.ffi_context().Emplace<MemsetValue>(42.0f));

  ExecuteOptions opts;
  opts.context = &context;

  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({42.0f, 42.0f, 42.0f, 42.0f}),
      *result_literal));
}

static absl::Status MemsetFromAttr(
    se::Stream* stream, float attr,
    ffi::Result<ffi::BufferR1<PrimitiveType::F32>> result) {
  uint32_t pattern;
  std::memcpy(&pattern, &attr, sizeof(pattern));

  se::DeviceMemoryBase base = result->device_memory();
  return stream->Memset32(&base, pattern, base.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromAttr, MemsetFromAttr,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Attr<float>("attr")
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromAttr",
                         PlatformUtil::CanonicalPlatformName("GPU").value(),
                         kMemsetFromAttr);

TEST(TfrtGpuClientTest, PassAttrToFfiHandler) {
  static constexpr char const* kProgram = R"(
  HloModule ffi_handler
  ENTRY main {
    ROOT %custom-call = f32[4] custom-call(),
        custom_call_target="MemsetFromAttr",
        api_version=API_VERSION_TYPED_FFI,
        backend_config={"custom_call_backend_config": {"attributes": "{attr = 3.0 : f32}"}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  ExecuteOptions opts;
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({3.0f, 3.0f, 3.0f, 3.0f}), *result_literal));
}

TEST(TfrtGpuClientTest, AcquireDonation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  // Create TfrtGpuBuffer.
  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(),
                                           device->local_device_id().value(),
                                           size_in_bytes));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      tsl::MakeAvailableAsyncValueRef<GpuEvent>());
  auto memory_space = device->default_memory_space().value();
  auto tfrt_buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  auto donation_transaction =
      DonationTransactionPeer::AcquireDonation(tfrt_buffer.get());
  EXPECT_TRUE(donation_transaction.ok());
  std::move(*donation_transaction).Commit();
  EXPECT_EQ(donation_transaction->device_buffer(), nullptr);
  EXPECT_TRUE(
      DonationTransactionPeer::GetDonationEvent(tfrt_buffer.get()).get());
}

TEST(TfrtGpuClientTest, DonateWithControlDependency) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto literal = LiteralUtil::CreateR2({{1, 2, 3}, {4, 5, 6}});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  PjRtFuture<> future(promise);
  auto blocked_buffer =
      std::move(*(buffer->DonateWithControlDependency(future)));
  EXPECT_TRUE(buffer->IsDeleted());

  buffer.reset();
  absl::Mutex mu;
  auto result_literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(blocked_buffer->on_device_shape()));
  bool got_literal = false;
  blocked_buffer->ToLiteral(result_literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  blocked_buffer.reset();

  EXPECT_FALSE(got_literal);
  promise.Set();
  EXPECT_TRUE(future.IsReady());

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, *result_literal));
}

TEST(TfrtGpuClientTest, ShouldStageHostToDeviceTransfersSetToTrue) {
  GpuClientOptions options_staging;
  options_staging.should_stage_host_to_device_transfers = true;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(options_staging));
  auto* staging_client = tensorflow::down_cast<TfrtGpuClient*>(client.get());
  EXPECT_TRUE(staging_client->should_stage_host_to_device_transfers());
  std::vector<int32_t> data(256);
  std::iota(data.begin(), data.end(), 10);
  Shape shape = ShapeUtil::MakeShape(S32, {256});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      staging_client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *client->addressable_devices()[0]->default_memory_space(),
          /*device_layout=*/nullptr));
  TF_EXPECT_OK(buffer->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                          buffer->ToLiteralSync());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(*literal, LiteralUtil::CreateR1<int32_t>(data)));
}

TEST(TfrtGpuClientTest, ShouldStageHostToDeviceTransfersSetToFalse) {
  GpuClientOptions options_staging;
  options_staging.should_stage_host_to_device_transfers = false;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(options_staging));
  auto* staging_client = tensorflow::down_cast<TfrtGpuClient*>(client.get());
  EXPECT_FALSE(staging_client->should_stage_host_to_device_transfers());
  std::vector<int32_t> data(256);
  std::iota(data.begin(), data.end(), 10);
  Shape shape = ShapeUtil::MakeShape(S32, {256});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      staging_client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *client->addressable_devices()[0]->default_memory_space(),
          /*device_layout=*/nullptr));
  TF_EXPECT_OK(buffer->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                          buffer->ToLiteralSync());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(*literal, LiteralUtil::CreateR1<int32_t>(data)));
}

TEST(TfrtGpuClientTest, BufferFromHostBufferPinnedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto* pinned_memory_space,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          pinned_memory_space, /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(buffer->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteralSync());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(TfrtGpuClientTest, CopyToPinnedHostMemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  auto device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *device->default_memory_space(), /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "device");

  auto* pinned_memory_space = device->memory_spaces()[1];
  EXPECT_EQ(pinned_memory_space->kind_id(), PinnedHostMemorySpace::kKindId);
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(pinned_memory_space));

  EXPECT_EQ(result->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(result->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(TfrtGpuClientTest, CopyToPinnedHostMemorySpaceInt4) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  std::vector<int8_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S4, {4});
  auto device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *device->default_memory_space(), /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "device");

  TF_EXPECT_OK(buffer->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> device_literal,
                          buffer->ToLiteralSync());
  std::vector<xla::s4> expected{xla::s4(1), xla::s4(2), xla::s4(3), xla::s4(4)};
  Literal expected_literal = LiteralUtil::CreateR1<xla::s4>(expected);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *device_literal));

  auto* pinned_memory_space = device->memory_spaces()[1];
  EXPECT_EQ(pinned_memory_space->kind_id(), PinnedHostMemorySpace::kKindId);
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(pinned_memory_space));

  EXPECT_EQ(result->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(result->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *literal));
}

TEST(TfrtGpuClientTest, ToLiteralAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  PjRtDevice* const device = client->addressable_devices()[0];
  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, *device->default_memory_space()));
  std::unique_ptr<PjRtBuffer> buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  buffer.reset();

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  EXPECT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  EXPECT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(TfrtGpuClientTest, ToLiteralAsyncWithNonCompactLayout) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  xla::Shape transposed_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::S32, {2, 3}, /*minor_to_major=*/{0, 1});
  xla::Literal src_literal = xla::LiteralUtil::CreateR2WithLayout<int32_t>(
      {{3, 14, 25}, {36, 47, 58}}, transposed_shape.layout());

  PjRtClient::ShapeSpec spec;
  spec.element_type = src_literal.shape().element_type();
  spec.dims = DimensionVector(src_literal.shape().dimensions().begin(),
                              src_literal.shape().dimensions().end());
  std::vector<std::optional<xla::Layout>> device_layouts = {
      std::make_optional(transposed_shape.layout())};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {spec}, device_layouts,
          client->addressable_devices()[0]->memory_spaces()[0]));
  std::unique_ptr<PjRtBuffer> buffer = transfer_manager->RetrieveBuffer(0);

  absl::Notification n;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    TF_ASSERT_OK(s);
    n.Notify();
  });
  buffer.reset();

  n.WaitForNotification();

  EXPECT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  EXPECT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncWithDifferentMajorToMinor) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::S32, {2, 3}, /*minor_to_major=*/{1, 0});
  xla::Literal src_literal = xla::LiteralUtil::CreateR2WithLayout<int32_t>(
      {{3, 14, 25}, {36, 47, 58}}, shape.layout());

  PjRtClient::ShapeSpec spec;
  spec.element_type = src_literal.shape().element_type();
  spec.dims = DimensionVector(src_literal.shape().dimensions().begin(),
                              src_literal.shape().dimensions().end());
  xla::Shape transposed_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::S32, {2, 3}, /*minor_to_major=*/{0, 1});
  std::vector<std::optional<xla::Layout>> device_layouts = {
      std::make_optional(transposed_shape.layout())};
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {spec}, device_layouts,
          client->addressable_devices()[0]->memory_spaces()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  absl::Notification n;
  auto literal = std::make_shared<Literal>(shape);

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    TF_ASSERT_OK(s);
    n.Notify();
  });
  buffer.reset();

  n.WaitForNotification();

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(TfrtGpuClientTest, ToLiteralAsyncBeforeBufferReady) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  PjRtDevice* const device = client->addressable_devices()[0];
  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, *device->default_memory_space()));
  std::unique_ptr<PjRtBuffer> buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });

  absl::SleepFor(absl::Milliseconds(10));
  ASSERT_FALSE(got_literal);
  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer.reset();

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  EXPECT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  EXPECT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(TfrtGpuClientTest, FromHostAsync) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  PjRtDevice* const device = client->addressable_devices()[0];
  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<float>(i + 10));
    src_literals.push_back(LiteralUtil::CreateR1<float>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          src_shapes, *device->default_memory_space()));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.push_back(transfer_manager->RetrieveBuffer(i));
  }

  for (int i = 0; i < src_shapes.size(); ++i) {
    TF_ASSERT_OK(transfer_manager->TransferRawDataToBuffer(
        i,
        absl::string_view(static_cast<char*>(src_literals[i].untyped_data()),
                          src_literals[i].size_bytes()),
        [&]() {}));
  }

  absl::Mutex mu;
  std::vector<std::shared_ptr<Literal>> literals;
  int got_literal_count = 0;
  int got_callback_count = 0;

  for (auto& buffer : buffers) {
    literals.push_back(std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape())));
    buffer->ToLiteral(literals.back().get()).OnReady([&](absl::Status s) {
      absl::MutexLock l(&mu);
      TF_ASSERT_OK(s);
      ++got_literal_count;
    });
    buffer->GetReadyFuture().OnReady([&](absl::Status s) {
      absl::MutexLock l(&mu);
      TF_ASSERT_OK(s);
      ++got_callback_count;
    });
    buffer.reset();
  }

  {
    auto done = [&]() {
      return got_literal_count == src_literals.size() &&
             got_callback_count == src_literals.size();
    };
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&done));
  }

  for (int i = 0; i < src_literals.size(); ++i) {
    EXPECT_TRUE(
        ShapeUtil::Compatible(src_literals[i].shape(), literals[i]->shape()));
    EXPECT_EQ(
        src_literals[i].data<float>(),
        literals[i]->Relayout(src_literals[i].shape().layout()).data<float>());
  }
}

TEST(TfrtGpuClientTest, FromHostAsyncPinnedHost) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      auto* pinned_memory_space,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<float>(i + 10));
    src_literals.emplace_back(LiteralUtil::CreateR1<float>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              src_shapes, pinned_memory_space));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.emplace_back(transfer_manager->RetrieveBuffer(i));
  }

  for (int i = 0; i < src_shapes.size(); ++i) {
    TF_ASSERT_OK(transfer_manager->TransferRawDataToBuffer(
        i,
        absl::string_view(static_cast<char*>(src_literals[i].untyped_data()),
                          src_literals[i].size_bytes()),
        [&]() {}));
  }
}

TEST(TfrtGpuClientTest, FromHostAsyncPinnedHostChunked) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  std::vector<float> data{1, 3, 5, 7, 11, 13, 17, 19};
  Shape shape = ShapeUtil::MakeShape(F32, {static_cast<int64_t>(data.size())});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
      client->CreateBuffersForAsyncHostToDevice({shape}, memspace));
  std::unique_ptr<PjRtBuffer> buf = txm->RetrieveBuffer(0);
  ASSERT_THAT(buf->GetReadyFuture().IsReady(), Eq(false));

  absl::string_view raw_view(reinterpret_cast<char*>(data.data()),
                             data.size() * sizeof(data[0]));
  int offset = 0;
  while (true) {
    int end = offset + 3;  // unaligned chunk size
    if (end > raw_view.size()) {
      end = raw_view.size();
    }
    int sz = end - offset;
    bool reaches_end = end == raw_view.size();
    TF_ASSERT_OK(txm->TransferRawDataToSubBuffer(
        /*buffer_index=*/0, raw_view.data() + offset, offset, sz, reaches_end,
        /*on_done=*/[]() {}));
    if (reaches_end) {
      break;
    }
    offset = end;
  }
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> lit, buf->ToLiteralSync());
  EXPECT_THAT(lit->data<float>(), ElementsAreArray(data));
}

TEST(TfrtGpuClientTest, DeleteBufferThenFulfillBufferNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  std::vector<float> data{1, 3, 5, 7, 11, 13, 17, 19};
  Shape shape = ShapeUtil::MakeShape(F32, {static_cast<int64_t>(data.size())});
  std::vector<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
      txms;
  for (int i = 0; i < 10000; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
        client->CreateBuffersForAsyncHostToDevice({shape}, memspace));
    std::unique_ptr<PjRtBuffer> buf = txm->RetrieveBuffer(0);
    ASSERT_THAT(buf->GetReadyFuture().IsReady(), Eq(false));
    txms.push_back(std::move(txm));
    // Delete the buffer
  }

  // At this point, we have 10000 buffers pending deallocation.

  absl::string_view raw_view(
      reinterpret_cast<char*>(data.data()),  // REINTERPRET_CAST_OK=test
      data.size() * sizeof(data[0]));
  for (auto& txm : txms) {
    int offset = 0;
    while (true) {
      int end = offset + 3;  // unaligned chunk size
      if (end > raw_view.size()) {
        end = raw_view.size();
      }
      int sz = end - offset;
      bool reaches_end = end == raw_view.size();
      TF_ASSERT_OK(txm->TransferRawDataToSubBuffer(
          /*buffer_index=*/0, raw_view.data() + offset, offset, sz, reaches_end,
          /*on_done=*/[]() {}));
      if (reaches_end) {
        break;
      }
      offset = end;
    }
  }
}

TEST(TfrtGpuClientTest, CreateMixOfErrorBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<float>(i + 10));
    src_literals.push_back(LiteralUtil::CreateR1<float>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          src_shapes, client->addressable_devices()[0]->memory_spaces()[0]));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.push_back(transfer_manager->RetrieveBuffer(i));
  }

  absl::Mutex mu;
  int got_callback_count = 0;
  for (int i = 0; i < 4; ++i) {
    auto& buffer = buffers[i];
    if (i == 0 || i == 3) {
      TF_ASSERT_OK(transfer_manager->TransferLiteralToBuffer(i, src_literals[i],
                                                             [&]() {}));
      buffer->GetReadyFuture().OnReady([&](absl::Status s) {
        absl::MutexLock l(&mu);
        TF_ASSERT_OK(s);
        ++got_callback_count;
      });
    } else {
      absl::Status error = Internal("error %d", i);
      transfer_manager->SetBufferError(i, error);
      buffer->GetReadyFuture().OnReady(
          [error, &mu, &got_callback_count](absl::Status s) {
            absl::MutexLock l(&mu);
            EXPECT_THAT(s.message(), HasSubstr(error.message()));
            ++got_callback_count;
          });
    }
    buffer.reset();
  }

  {
    auto done = [&]() { return got_callback_count == src_literals.size(); };
    absl::MutexLock l(&mu);
    QCHECK(mu.AwaitWithTimeout(absl::Condition(&done), absl::Seconds(60)));
  }
}

TEST(TfrtGpuClientTest, LookupDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 2);
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  TF_ASSERT_OK_AND_ASSIGN(
      auto* looked_up_device,
      client->LookupDevice(PjRtGlobalDeviceId(device->id())));
  EXPECT_EQ(looked_up_device, device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto* addressable_device,
      client->LookupAddressableDevice(device->local_device_id()));
  EXPECT_EQ(addressable_device, device);
}

TEST(TfrtGpuClientTest, CreateViewOfDeviceBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  void* device_ptr = (void*)0x12345678;
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  bool deleted = false;
  auto on_delete_callback = [&]() { deleted = true; };
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          client->CreateViewOfDeviceBuffer(
                              device_ptr, on_device_shape, memory_space,
                              on_delete_callback, /*stream=*/std::nullopt));
  EXPECT_EQ(buffer->on_device_shape(), on_device_shape);
  EXPECT_EQ(buffer->memory_space(), memory_space);
  {
    TF_ASSERT_OK_AND_ASSIGN(auto ref, buffer->AcquireExternalReference());
    EXPECT_EQ(ref->OpaqueDeviceMemoryDataPointer(), device_ptr);
  }
  EXPECT_FALSE(deleted);
  buffer.reset();
  EXPECT_TRUE(deleted);
}

TEST(TfrtGpuClientTest, CopyRawToHostFullBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);

  auto result = buffer->CopyRawToHost(dst, 0, size);
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);
  EXPECT_EQ(*(static_cast<float*>(dst) + 1), 42.0f);

  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(TfrtGpuClientTest, CopyRawToHostSubBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);

  auto result = buffer->CopyRawToHost(dst, 0, sizeof(float));
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);

  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(TfrtGpuClientTest, CopyRawToHostOutOfRange) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);

  auto result = buffer->CopyRawToHost(dst, 1, size);
  EXPECT_THAT(result.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("invalid offset 1")));
  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(TfrtGpuClientTest, CopyRawToHostFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  auto dst_promise = xla::PjRtFuture<void*>::CreatePromise();
  xla::PjRtFuture<void*> dst_future(dst_promise);

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  auto ready = buffer->GetReadyFuture();
  auto result = buffer->CopyRawToHostFuture(dst_future, 0, size);

  // Drop the buffer before fulfilling `dst`. The transfer should still keep
  // the buffer alive.
  buffer.reset();
  ready.OnReady([dst_promise = std::move(dst_promise),
                 size](absl::Status status) mutable {
    void* dst =
        tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);
    dst_promise.Set(dst);
  });

  TF_EXPECT_OK(result.Await());
  TF_ASSERT_OK_AND_ASSIGN(auto* dst, dst_future.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);
  EXPECT_EQ(*(static_cast<float*>(dst) + 1), 42.0f);

  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(GpuTopology, FromProto) {
  GpuTopologyProto msg;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        device_ids: [ 3, 2, 1 ]
        platform_version: "platform_version"
        num_slices: 2
        num_hosts_per_slice: 1
        num_devices_per_host: 3
      )pb",
      &msg));

  std::unique_ptr<const GpuTopology> gpu_topology = GpuTopology::FromProto(msg);
  EXPECT_THAT(gpu_topology->device_ids(), ElementsAre(3, 2, 1));
  EXPECT_THAT(gpu_topology->platform_version(), "platform_version");
  EXPECT_THAT(gpu_topology->num_slices(), 2);
  EXPECT_THAT(gpu_topology->num_hosts_per_slice(), 1);
  EXPECT_THAT(gpu_topology->num_devices_per_host(), 3);
}

TEST(GpuTopology, ToProto) {
  GpuTopology gpu_topology(/*gpu_device_ids=*/{3, 2, 1},
                           /*platform_version=*/"platform_version",
                           /*num_slices=*/2,
                           /*num_hosts_per_slice=*/1,
                           /*num_devices_per_host=*/3);
  GpuTopologyProto msg = gpu_topology.ToProto();
  EXPECT_THAT(msg.device_ids(), ElementsAre(3, 2, 1));
  EXPECT_THAT(msg.platform_version(), "platform_version");
  EXPECT_THAT(msg.num_slices(), 2);
  EXPECT_THAT(msg.num_hosts_per_slice(), 1);
  EXPECT_THAT(msg.num_devices_per_host(), 3);
}

TEST(TfrtGpuClientTest, DistributedInit) {
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "DistributeInit", 4);

  int num_nodes = 2;
  for (int i = 0; i < num_nodes; i++) {
    thread_pool.Schedule([kv_store, i, num_nodes] {
      GpuClientOptions options;
      options.node_id = i;
      options.num_nodes = num_nodes;
      options.kv_store = kv_store;
      TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(options));
      EXPECT_TRUE(client->platform_name() == "cuda" ||
                  client->platform_name() == "rocm");
      EXPECT_EQ(client->addressable_device_count(), 2);
      EXPECT_EQ(client->device_count(), 4);
    });
  }
}

namespace {
constexpr int32_t kData[] = {1, 2, 3, 4};
absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateDeviceBufferForTest(
    xla::PjRtClient* client) {
  auto device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  TF_ASSIGN_OR_RETURN(
      auto input,
      client->BufferFromHostBuffer(
          kData, shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");
  return input;
}
constexpr char const* kD2HProgram = R"(
  HloModule f

  ENTRY main.5 {
    p = s32[4]{0} parameter(0)
    ROOT cc = s32[4] custom-call(p),
        custom_call_target="annotate_device_placement",
        frontend_attributes={_xla_buffer_placement="pinned_host"}
  }
)";

constexpr char const* kD2HProgramTupleOutput = R"(
  HloModule f

  ENTRY main.5 {
    p = s32[4]{0} parameter(0)
    cc = s32[4] custom-call(p),
        custom_call_target="annotate_device_placement",
        frontend_attributes={_xla_buffer_placement="pinned_host"}
    ROOT tuple = (s32[4]{0}, s32[4]{0}) tuple(s32[4]{0} p, s32[4]{0} cc)
  }
)";

}  // namespace

TEST(TfrtGpuClientTest, ExecutablePinnedHostOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "pinned_host");
}

TEST(TfrtGpuClientTest, ExecutablePinnedHostTupleOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  // Build the output shape with the correct memory space set.
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  Shape host_shape = shape;
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  Shape out_shape = ShapeUtil::MakeTupleShape({shape, host_shape});

  // Set the result layout so that the compiler assertions on memory
  // spaces pass.
  xla::CompileOptions options;
  options.executable_build_options.set_result_layout(out_shape);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kD2HProgramTupleOutput, *client, options));

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 2);
  EXPECT_EQ(memory_kinds[0][0], "device");
  EXPECT_EQ(memory_kinds[0][1], "pinned_host");
}

TEST(TfrtGpuClientTest, ExecutePinnedHostOutputTest) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> input,
                          CreateDeviceBufferForTest(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      executable->Execute({{input.get()}}, ExecuteOptions()));

  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "pinned_host");

  TF_ASSERT_OK_AND_ASSIGN(auto memory_stats,
                          executable->GetCompiledMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 16);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                          result_buffers[0]->ToLiteralSync())
  EXPECT_THAT(literal->data<int32_t>(), ElementsAreArray(kData));
}

TEST(TfrtGpuClientTest, ExecutePinnedHostOutputTupleTest) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> input,
                          CreateDeviceBufferForTest(client.get()));

  // Build the output shape with the correct memory space set.
  Shape host_shape = input->on_device_shape();
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  Shape out_shape =
      ShapeUtil::MakeTupleShape({input->on_device_shape(), host_shape});

  // Set the result layout so that the compiler assertions on memory
  // spaces pass.
  xla::CompileOptions options;
  options.executable_build_options.set_result_layout(out_shape);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kD2HProgramTupleOutput, *client, options));

  // Untuple the result so that we get separate buffers.
  // This is how JAX invokes XLA.
  ExecuteOptions execute_options;
  execute_options.untuple_result = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, execute_options));

  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers.size(), 2);
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  EXPECT_EQ(result_buffers[1]->memory_space()->kind(), "pinned_host");

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                          result_buffers[0]->ToLiteralSync())
  EXPECT_THAT(literal->data<int32_t>(), ElementsAreArray(kData));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> another_literal,
                          result_buffers[1]->ToLiteralSync())
  EXPECT_THAT(another_literal->data<int32_t>(), ElementsAreArray(kData));
}

TEST(TfrtGpuClientTest, MlirParameterLayoutFromOptionsIsSetInHlo) {
  constexpr char kMlirCopy[] =
      R"(
      func.func public @main(%arg0: tensor<2x2x2xi32> {
              mhlo.layout_mode = "default"
          }) -> (tensor<2x2x2xi32> {
              jax.result_info = "",
              mhlo.layout_mode = "default"}) {
        return %arg0 : tensor<2x2x2xi32>
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseMlirModuleString(kMlirCopy, context));

  xla::CompileOptions options;
  options.argument_layouts = {
      {ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 2, 2}, {0, 2, 1})}};
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->CompileAndLoad(*module, options));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({0, 2, 1}));
}

TEST(TfrtGpuClientTest, GetDefaultLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  auto shape = ShapeUtil::MakeShape(S4, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto layout,
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));
  EXPECT_EQ(layout.element_size_in_bits(), 4);
}

TEST(TfrtGpuClientTest, AutoLayoutIsSupported) {
  const char* hlo_text = R"(
    HloModule DotLayout,
      entry_computation_layout={(f32[2,3,5],f32[3,4,5])->f32[5,2,4]{2,1,0}}

    ENTRY dot {
      p0 = f32[2,3,5]{2,1,0} parameter(0)
      p1 = f32[3,4,5]{2,1,0} parameter(1)
      ROOT dot.1330.10585 = f32[5,2,4]{2,1,0} dot(p0, p1),
        lhs_batch_dims={2}, lhs_contracting_dims={1},
        rhs_batch_dims={2}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo_text, {}, HloParserOptions().set_fill_missing_layouts(false)));

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_pjrt_allow_auto_layout_in_hlo(true);
  XlaComputation computation = m->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->CompileAndLoad(computation, compile_options));
  TF_ASSERT_OK_AND_ASSIGN(auto layouts, executable->GetParameterLayouts());
  // Check that the assigned layouts are not default.
  EXPECT_NE(layouts[0]->ToString(), "{2,1,0}");
  EXPECT_NE(layouts[1]->ToString(), "{2,1,0}");
}

TEST(TfrtGpuClientTest, CreateUninitializedBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, client->CreateUninitializedBuffer(
                                           on_device_shape, memory_space));
  EXPECT_EQ(*buffer->GetOnDeviceSizeInBytes(), 4 * 4 * 4);
}

TEST(TfrtGpuClientTest, SerializeDeserializeExecutable) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  static constexpr char const* kAddProgram =
      R"(
HloModule Add.6, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

ENTRY %Add.6 (a.1: f32[], b.2: f32[]) -> (f32[], f32[]) {
  %a.1 = f32[] parameter(0)
  %b.2 = f32[] parameter(1)
  %add.3 = f32[] add(f32[] %a.1, f32[] %b.2)
  %add.4 = f32[] add(f32[] %add.3, f32[] %add.3)
  ROOT %tuple.5 = (f32[], f32[]) tuple(f32[] %add.3, f32[] %add.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));
  auto gpu_exe = static_cast<TfrtGpuExecutable*>(std::move(executable).get());
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          gpu_exe->SerializeExecutable());
  EXPECT_TRUE(
      absl::StrContains(serialized, "Generated by LLVM NVPTX Back-End"));

  TF_ASSERT_OK_AND_ASSIGN(auto deserialized, client->DeserializeExecutable(
                                                 serialized, std::nullopt));
  EXPECT_EQ(deserialized->num_replicas(), 1);
  EXPECT_EQ(deserialized->num_partitions(), 1);
  EXPECT_EQ(deserialized->name(), "Add.6");
}

TEST(TfrtGpuClientTest, ValidatesClientName) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  static constexpr char const* kAddProgram =
      R"(
HloModule Add.6, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

ENTRY %Add.6 (a.1: f32[], b.2: f32[]) -> (f32[], f32[]) {
  %a.1 = f32[] parameter(0)
  %b.2 = f32[] parameter(1)
  %add.3 = f32[] add(f32[] %a.1, f32[] %b.2)
  %add.4 = f32[] add(f32[] %add.3, f32[] %add.3)
  ROOT %tuple.5 = (f32[], f32[]) tuple(f32[] %add.3, f32[] %add.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));
  const auto* gpu_exe =
      static_cast<TfrtGpuExecutable*>(std::move(executable).get());
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          gpu_exe->SerializeExecutable());

  ExecutableAndOptionsProto proto;
  proto.ParseFromString(serialized);
  EXPECT_EQ(proto.pjrt_client_name(), "TfrtGpuClient");
  proto.set_pjrt_client_name("SomeGpuClient");
  serialized = proto.SerializeAsString();

  EXPECT_THAT(client->DeserializeExecutable(serialized, std::nullopt),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("PjRt client type expected by the serialized "
                                 "executable: SomeGpuClient")));
}

TEST(TfrtGpuClientTest, CopyToMemorySpace) {
  // TODO(sizhi): Re-enable this test after the feature is implemented.
  GTEST_SKIP() << "Skipping this test.";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  for (auto* memory_space : client->memory_spaces()) {
    xla::Shape shape = xla::ShapeUtil::MakeShape(S32, {128, 256});
    TF_ASSERT_OK_AND_ASSIGN(Literal literal, xla::MakeFakeLiteral(shape));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> buffer,
        client->BufferFromHostLiteral(literal, memory_space));
    TF_ASSERT_OK_AND_ASSIGN(buffer,
                            buffer->CopyToMemorySpace(buffer->memory_space()));
    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> received_literal,
                            buffer->ToLiteralSync());
    EXPECT_THAT(received_literal->data<int32_t>(),
                ElementsAreArray(literal.data<int32_t>()));
  }
}

TEST(TfrtGpuClientTest, AsyncCopyToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  // d0 is the device we will perform local/remote sends from.
  PjRtDevice* const d0 = client->addressable_devices()[0];
  // d1 is the device we will perform local/remote recvs, where the recv
  // sync flag may be contended.
  PjRtDevice* const d1 = client->addressable_devices()[1];

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({src_literal.shape()},
                                                *d0->default_memory_space()));
  std::unique_ptr<PjRtBuffer> src_buffer = transfer_manager->RetrieveBuffer(0);
  // CopyToMemorySpace won't be enqueued until src_buffer is available.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> local_recv_buffer,
      src_buffer->CopyToMemorySpace(*d1->default_memory_space()));

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, []() {}));

  auto literal = std::make_shared<Literal>(src_literal.shape());

  PjRtFuture<> local_recv_literal = local_recv_buffer->ToLiteral(literal.get());
  TF_EXPECT_OK(local_recv_literal.Await());

  EXPECT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  EXPECT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(TfrtGpuClientTest, OnDoneSafelyDestructTransferManagerAsync) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);
  PjRtDevice* const device = client->addressable_devices()[0];

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>
          transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, *device->default_memory_space()));
  std::unique_ptr<PjRtBuffer> buffer = transfer_manager->RetrieveBuffer(0);
  absl::Notification done;
  EXPECT_OK(transfer_manager->TransferLiteralToBuffer(
      0, src_literal,
      /*on_done=*/
      [&done, transfer_manager = std::move(transfer_manager)]() {
        done.Notify();
      }));
  done.WaitForNotification();
}

TEST(TfrtGpuClientTest, DeviceAttributes) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  ASSERT_EQ(client->platform_name(), "cuda");

  for (int device_index = 0;
       device_index < client->addressable_devices().size(); ++device_index) {
    TfrtGpuDevice* device = tensorflow::down_cast<TfrtGpuDevice*>(
        client->addressable_devices()[device_index]);

    // Attribute `compute_capability`.
    auto compute_capability =
        std::get<std::string>(device->Attributes().at("compute_capability"));

    // Gets the expected compute capability.
    const se::Platform* platform = device->executor()->GetPlatform();
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::se::DeviceDescription> desc,
                            platform->DescriptionForDevice(0));
    stream_executor::GpuComputeCapability cc = desc->gpu_compute_capability();
    auto nvcc = std::get<stream_executor::CudaComputeCapability>(cc);
    std::string expected_compute_capability =
        absl::StrCat(nvcc.major, ".", nvcc.minor);
    EXPECT_EQ(compute_capability, expected_compute_capability);

    // Attribute `coords`.
    EXPECT_EQ(device->description().coords()[0], device_index);

    // Attribute `device_vendor`.
    auto device_vendor =
        std::get<std::string>(device->Attributes().at("device_vendor"));
    EXPECT_EQ(device_vendor, desc->device_vendor());

    // Attribute `slice_index`.
    auto slice_index =
        std::get<int64_t>(device->Attributes().at("slice_index"));
    EXPECT_EQ(slice_index, 0);

    // Attribute `core_count`.
    auto core_count = std::get<int64_t>(device->Attributes().at("core_count"));
    EXPECT_EQ(core_count, desc->core_count());
  }
}

TEST(TfrtGpuClientTest, DmaMapUnmap) {
  GpuClientOptions options = GpuClientOptions();
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_client, GetTfrtGpuClient(options));
  auto client = tensorflow::down_cast<TfrtGpuClient*>(gpu_client.get());
  size_t dma_size = 8192;
  size_t alignment = 4096;
  auto host_dma_ptr = xla::AlignedAlloc(alignment, dma_size);

  // DmaMap the first half of the buffer.
  size_t dma_map_size = dma_size / 2;
  char* first_half_ptr = static_cast<char*>(host_dma_ptr.get());
  char* second_half_ptr = first_half_ptr + dma_map_size;
  int offset = 5;
  TF_EXPECT_OK(client->DmaMap(first_half_ptr, dma_map_size));
  EXPECT_TRUE(client->IsDmaMapped(first_half_ptr, dma_map_size));
  EXPECT_TRUE(client->IsDmaMapped(first_half_ptr + offset, 10));
  EXPECT_FALSE(client->IsDmaMapped(first_half_ptr + offset, dma_map_size));
  EXPECT_TRUE(
      client->IsDmaMapped(first_half_ptr + offset, dma_map_size - offset));

  // Verify boundaries.
  EXPECT_TRUE(client->IsDmaMapped(first_half_ptr, 1));
  EXPECT_FALSE(client->IsDmaMapped(first_half_ptr - 1, 1));
  EXPECT_FALSE(client->IsDmaMapped(first_half_ptr + dma_map_size, 1));

  // DmaMap the second half of the buffer.
  TF_EXPECT_OK(client->DmaMap(second_half_ptr, dma_map_size));
  EXPECT_TRUE(client->IsDmaMapped(second_half_ptr, dma_map_size));
  EXPECT_TRUE(client->IsDmaMapped(second_half_ptr + offset, 10));
  EXPECT_FALSE(client->IsDmaMapped(second_half_ptr + offset, dma_map_size));
  EXPECT_TRUE(
      client->IsDmaMapped(second_half_ptr + offset, dma_map_size - offset));

  // Verify boundaries.
  EXPECT_TRUE(client->IsDmaMapped(second_half_ptr, 1));
  EXPECT_TRUE(client->IsDmaMapped(second_half_ptr - 1, 1));
  EXPECT_FALSE(client->IsDmaMapped(second_half_ptr + dma_map_size, 1));

  // Unmap the first half of the buffer.
  TF_EXPECT_OK(client->DmaUnmap(first_half_ptr));
  EXPECT_FALSE(client->IsDmaMapped(first_half_ptr, dma_map_size));
  EXPECT_FALSE(client->IsDmaMapped(first_half_ptr + offset, 10));
  EXPECT_FALSE(client->IsDmaMapped(second_half_ptr - 1, 1));
  EXPECT_TRUE(client->IsDmaMapped(second_half_ptr, 1));

  // Unmap the second half of the buffer.
  TF_EXPECT_OK(client->DmaUnmap(second_half_ptr));
  EXPECT_FALSE(client->IsDmaMapped(second_half_ptr, 1));
}

TEST(TfrtGpuClientTest, MultipleDeviceShareDmaMapping) {
  GpuClientOptions options = GpuClientOptions();

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(options));
  if (client->addressable_devices().size() < 2) {
    GTEST_SKIP() << "Test requires at least two addressable devices.";
  }

  size_t test_length = 0.5l * 1024 * 1024;
  std::vector<int32_t> data(test_length);
  for (int32_t i = 0; i < test_length; ++i) {
    data[i] = i;
  }
  Shape shape = ShapeUtil::MakeShape(S32, {static_cast<int64_t>(data.size())});
  PjRtDevice* const first_device = client->addressable_devices()[0];

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> first_buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr,
          first_device->memory_spaces()[0], /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, first_buffer->GetOnDeviceSizeInBytes());

  size_t dma_size = 2 * 1024 * 1024;
  size_t alignment = 1024;
  auto host_dma_ptr = xla::AlignedAlloc(alignment, dma_size);
  TF_EXPECT_OK(client->DmaMap(host_dma_ptr.get(), dma_size));

  auto result = first_buffer->CopyRawToHost(host_dma_ptr.get(), 0, size);
  TF_EXPECT_OK(result.Await());

  PjRtDevice* const second_device = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, second_device->memory_spaces()[0]));
  auto second_buffer = transfer_manager->RetrieveBuffer(0);

  TF_EXPECT_OK(transfer_manager->TransferRawDataToSubBuffer(
      0, host_dma_ptr.get(), 0, size, true, []() {}));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, second_buffer->ToLiteralSync());
  EXPECT_EQ(literal->element_count(), test_length);
  EXPECT_THAT(literal->data<int32_t>(), ElementsAreArray(data));

  TF_EXPECT_OK(client->DmaUnmap(host_dma_ptr.get()));
}

}  // namespace
}  // namespace xla
