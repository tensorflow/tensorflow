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

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

#include <stdlib.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/client/xla_computation.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream.h"
#include "xla/test.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.Compile(xla_computation, compile_options);
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

TEST(StreamExecutorGpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    EXPECT_THAT(device->memory_spaces(), ElementsAre(memory_space));
    EXPECT_EQ(memory_space->kind(), StreamExecutorGpuHbmMemorySpace::kKind);
    EXPECT_EQ(memory_space->kind_id(),
              StreamExecutorGpuHbmMemorySpace::kKindId);
    EXPECT_THAT(
        device->memory_space_by_kind(StreamExecutorGpuHbmMemorySpace::kKind),
        IsOkAndHolds(memory_space));
  }
}

TEST(StreamExecutorGpuClientTest, PropagateError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
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

TEST(StreamExecutorGpuClientTest, SendRecvChunked) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest, SendErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

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
  EXPECT_TRUE(absl::StrContains(result.status().message(),
                                "Uh-oh, can send chunk to host"));
}

TEST(StreamExecutorGpuClientTest, RecvErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

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
  EXPECT_TRUE(absl::StrContains(result.status().message(),
                                "Adding chunk of size 40 would overflow buffer "
                                "of size 8 (0 already transferred)"));
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

  se::DeviceMemoryBase base = result->data;
  return stream->Memset32(&base, pattern, result->data.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromValue, MemsetFromValue,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>()
                           .Ctx<ffi::UserData<MemsetValue>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromValue",
                         PlatformUtil::CanonicalPlatformName("GPU").value(),
                         kMemsetFromValue);

TEST(StreamExecutorGpuClientTest, ForwardUserDataToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
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

TEST(StreamExecutorGpuClientTest, ToLiteralAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

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

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncBeforeBufferReady) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

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

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, FromHostAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

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
                              src_shapes, client->addressable_devices()[0]));
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
    ASSERT_TRUE(
        ShapeUtil::Compatible(src_literals[i].shape(), literals[i]->shape()));
    ASSERT_EQ(
        src_literals[i].data<float>(),
        literals[i]->Relayout(src_literals[i].shape().layout()).data<float>());
  }
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostFullBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->addressable_devices()[0]));

  void* dst = aligned_alloc(buffer->GetOnDeviceSizeInBytes().value(), 0);

  auto result =
      buffer->CopyRawToHost(dst, 0, buffer->GetOnDeviceSizeInBytes().value());
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);
  EXPECT_EQ(*(static_cast<float*>(dst) + 1), 42.0f);

  free(dst);
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostSubBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->addressable_devices()[0]));
  void* dst = aligned_alloc(buffer->GetOnDeviceSizeInBytes().value(), 0);

  auto result = buffer->CopyRawToHost(dst, 0, sizeof(float));
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);

  free(dst);
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostOutOfRange) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->addressable_devices()[0]));
  void* dst = aligned_alloc(buffer->GetOnDeviceSizeInBytes().value(), 0);

  auto result =
      buffer->CopyRawToHost(dst, 1, buffer->GetOnDeviceSizeInBytes().value());
  EXPECT_THAT(result.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("invalid offset 1")));
  free(dst);
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->addressable_devices()[0]));

  auto dst_promise = xla::PjRtFuture<void*>::CreatePromise();
  xla::PjRtFuture<void*> dst_future(dst_promise);

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  auto ready = buffer->GetReadyFuture();
  auto result = buffer->CopyRawToHostFuture(dst_future, 0, size);

  // Drop the buffer before fulfilling `dst`. The transfer should still keep the
  // buffer alive.
  buffer.reset();
  ready.OnReady([dst_promise = std::move(dst_promise),
                 size](absl::Status status) mutable {
    dst_promise.Set(aligned_alloc(size, 0));
  });

  TF_EXPECT_OK(result.Await());
  TF_ASSERT_OK_AND_ASSIGN(auto* dst, dst_future.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);
  EXPECT_EQ(*(static_cast<float*>(dst) + 1), 42.0f);

  free(dst);
}

TEST(StreamExecutorGpuClientTest, AsyncCopyToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  // d0 is the device we will perform local/remote sends from.
  auto* d0 = client->addressable_devices()[0];
  // d1 is the device we will perform local/remote recvs, where the recv
  // sync flag may be contended.
  auto* d1 = client->addressable_devices()[1];

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({src_literal.shape()}, d0));
  auto src_buffer = transfer_manager->RetrieveBuffer(0);
  // CopyToDevice won't be enqueued until src_buffer is available.
  auto local_recv_buffer = *src_buffer->CopyToDevice(d1);

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, []() {}));

  auto literal = std::make_shared<Literal>(src_literal.shape());

  auto local_recv_literal = local_recv_buffer->ToLiteral(literal.get());
  TF_EXPECT_OK(local_recv_literal.Await());

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, CreateMixOfErrorBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<float>(i + 10));
    src_literals.emplace_back(LiteralUtil::CreateR1<float>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          src_shapes, client->addressable_devices()[0]->memory_spaces()[0]));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.emplace_back(transfer_manager->RetrieveBuffer(i));
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
            ASSERT_EQ(s, error);
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

TEST(StreamExecutorGpuClientTest, DistributedInit) {
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "DistributeInit", 4);

  int num_nodes = 2;
  for (int i = 0; i < num_nodes; i++) {
    thread_pool.Schedule([kv_store, i, num_nodes] {
      GpuClientOptions options;
      options.node_id = i;
      options.num_nodes = num_nodes;
      options.kv_store = kv_store;
      TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
      EXPECT_TRUE(client->platform_name() == "cuda" ||
                  client->platform_name() == "rocm");
      EXPECT_EQ(client->addressable_device_count(), 2);
      EXPECT_EQ(client->device_count(), 4);
    });
  }
}

TEST(StreamExecutorGpuClientTest, GetAllocatorStatsTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  for (auto device : client->addressable_devices()) {
    const xla::Literal literal = xla::LiteralUtil::CreateR0<int32_t>(0);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> buffer,
                            client->BufferFromHostLiteral(literal, device));

    auto stats = device->GetAllocatorStats();
    TF_ASSERT_OK(stats.status());
    ASSERT_GT(stats.value().peak_bytes_in_use, 0);
  }
}

TEST(StreamExecutorGpuClientTest, GpuDeviceDescriptionTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  for (int device_index = 0; device_index < client->device_count();
       device_index++) {
    auto coords =
        static_cast<PjRtStreamExecutorDevice*>(client->devices()[device_index])
            ->description()
            .coords();
    EXPECT_EQ(coords[0], device_index);
  }
  EXPECT_EQ(static_cast<PjRtStreamExecutorDevice*>(client->devices()[0])
                ->description()
                .core_on_chip(),
            0);
}

}  // namespace
}  // namespace xla
