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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "google/protobuf/text_format.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/pjrt/profiling/test_util/mock_device_time_measurement.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::SizeIs;

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
  auto literal_or = result_buffers[0]->ToLiteral().Await();
  if (!literal_or.status().ok()) {
    return literal_or.status();
  }
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

GpuClientOptions DefaultOptions() {
  // Most test cases expect exactly 2 GPUs.
  GpuClientOptions options;
  options.allowed_devices = std::set<int>({0, 1});
  return options;
}

TEST(StreamExecutorGpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    EXPECT_EQ(memory_space->kind(), StreamExecutorGpuHbmMemorySpace::kKind);
    EXPECT_EQ(memory_space->kind_id(),
              StreamExecutorGpuHbmMemorySpace::kKindId);
    EXPECT_THAT(
        device->memory_space_by_kind(StreamExecutorGpuHbmMemorySpace::kKind),
        absl_testing::IsOkAndHolds(memory_space));
    EXPECT_EQ(device->memory_spaces().size(), 2);
    auto* pinned = device->memory_spaces()[1];
    EXPECT_EQ(pinned->kind_id(), PinnedHostMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(PinnedHostMemorySpace::kKind),
                absl_testing::IsOkAndHolds(pinned));
  }
}

TEST(StreamExecutorGpuClientTest, MemorySpacesUniqueIds) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

TEST(StreamExecutorGpuClientTest, NumaNode) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    const auto it = device->Attributes().find("numa_node");
    ASSERT_NE(it, device->Attributes().end());

    const int64_t* value = std::get_if<int64_t>(&it->second);
    ASSERT_NE(value, nullptr);
    EXPECT_NE(*value, tsl::port::kNUMANoAffinity);
  }
}

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
TEST(StreamExecutorGpuClientTest, DonateExternalMem) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto shape = xla::ShapeUtil::MakeScalarShape(xla::F32);

  std::vector<float> data = {1.0f};

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_a,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr,
          client->addressable_devices()[0]->memory_spaces()[0],
          /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_ref,
                          buffer_a->AcquireExternalReference());

  auto device_ptr = buffer_ref->OpaqueDeviceMemoryDataPointer();
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->CreateViewOfDeviceBuffer(
                       device_ptr, shape, buffer_a->memory_space(),
                       [buf = std::shared_ptr<PjRtBuffer::ExternalReference>(
                            std::move(buffer_ref))]() {}));

  static constexpr char const* kAddProgram =
      R"(
HloModule jit_add_one, input_output_alias={ {}: (0, {}, may-alias) }, entry_computation_layout={(f32[])->f32[]}

ENTRY main.5 {
  x = f32[] parameter(0), sharding={replicated}
  constant = f32[] constant(1)
  ROOT result = f32[] add(x, constant)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{buffer.get()}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  TF_EXPECT_OK(result[0][0]->GetReadyFuture().Await());
}
#endif  // defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)

TEST(StreamExecutorGpuClientTest, CreateErrorBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});
  for (PjRtMemorySpace* memory_space : client->memory_spaces()) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        client->CreateErrorBuffer(Internal("foobar"), shape, memory_space));
    EXPECT_THAT(
        buffer->ToLiteral().Await(),
        absl_testing::StatusIs(tsl::error::INTERNAL, HasSubstr("foobar")));
    EXPECT_EQ(buffer->memory_space(), memory_space);
  }
}

TEST(StreamExecutorGpuClientTest, PropagateError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

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
  ASSERT_EQ(result[0].size(), 2);
  for (const auto& b : result[0]) {
    EXPECT_EQ(b->GetReadyFuture().Await(), input_error);
  }
}

// TODO(b/372735047): Fix and reenable.
TEST(StreamExecutorGpuClientTest, DISABLED_DonateWithControlDependency) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

  TF_ASSERT_OK_AND_ASSIGN(
      auto another_buffer,
      client->CreateErrorBuffer(
          input_error, shape,
          *client->addressable_devices()[0]->default_memory_space()));
  TF_ASSERT_OK_AND_ASSIGN(another_buffer,
                          another_buffer->DonateWithControlDependency(
                              result[0][0]->GetReadyFuture()));
  EXPECT_EQ(another_buffer->GetReadyFuture().Await(), input_error);
}

TEST(StreamExecutorGpuClientTest, SendRecvChunked) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

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
        CHECK_OK(stream->AddChunk(std::move(chunk0)).Await());

        auto chunk1 = PjRtChunk::AllocateDefault(sizeof(float));
        *reinterpret_cast<float*>(chunk1.data()) = 6.0f;
        CHECK_OK(stream->AddChunk(std::move(chunk1)).Await());

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
                          GetStreamExecutorGpuClient(DefaultOptions()));

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
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(/*argument_handles=*/{{}}, opts));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  auto status = result[0][0]->GetReadyFuture().Await();
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Uh-oh, can send chunk to host"));
}

TEST(StreamExecutorGpuClientTest, RecvErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

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
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(/*argument_handles=*/{{}}, opts));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  auto status = result[0][0]->GetReadyFuture().Await();
  EXPECT_TRUE(absl::StrContains(status.message(),
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

  se::DeviceAddressBase base = result->device_memory();
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

TEST(StreamExecutorGpuClientTest, ForwardUserDataToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

  se::DeviceAddressBase base = result->device_memory();
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

TEST(StreamExecutorGpuClientTest, PassAttrToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
          custom_call_target="MemsetFromAttr",
          api_version=API_VERSION_TYPED_FFI,
          backend_config={"custom_call_backend_config": {"attributes": "{attr = 3.0 : f32}"}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  ExecuteOptions opts;
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({3.0f, 3.0f, 3.0f, 3.0f}), *result_literal));
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto* d = client->addressable_devices()[0];
  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({src_literal.shape()},
                                                *d->default_memory_space()));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  buffer.reset();

  {
    absl::MutexLock l(mu);
    mu.Await(absl::Condition(&got_literal));
  }

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncWithNonCompactLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {spec}, device_layouts,
          client->addressable_devices()[0]->memory_spaces()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

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

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncWithDifferentMajorToMinor) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncToken) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  xla::Literal literal = xla::LiteralUtil::CreateToken();

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostLiteral(
          literal, client->addressable_devices()[0]->memory_spaces()[0]));
  TF_ASSERT_OK(buffer->GetReadyFuture().Await());

  absl::Notification n;

  buffer->ToLiteral(&literal).OnReady([&](absl::Status s) {
    TF_ASSERT_OK(s);
    n.Notify();
  });
  buffer.reset();

  n.WaitForNotification();
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncBeforeBufferReady) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto* d = client->addressable_devices()[0];
  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({src_literal.shape()},
                                                *d->default_memory_space()));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });

  absl::SleepFor(absl::Milliseconds(10));
  ASSERT_FALSE(got_literal);
  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer.reset();

  {
    absl::MutexLock l(mu);
    mu.Await(absl::Condition(&got_literal));
  }

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, FromHostAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto* d = client->addressable_devices()[0];
  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    absl::c_iota(data, static_cast<float>(i + 10));
    src_literals.emplace_back(LiteralUtil::CreateR1<float>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              src_shapes, *d->default_memory_space()));
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
      absl::MutexLock l(mu);
      TF_ASSERT_OK(s);
      ++got_literal_count;
    });
    buffer->GetReadyFuture().OnReady([&](absl::Status s) {
      absl::MutexLock l(mu);
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
    absl::MutexLock l(mu);
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

TEST(StreamExecutorGpuClientTest, FromHostAsyncPinnedHost) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      auto* pinned_memory_space,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    absl::c_iota(data, static_cast<float>(i + 10));
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

  absl::Mutex mu;
  std::vector<std::shared_ptr<Literal>> literals;
  int got_literal_count = 0;
  int got_callback_count = 0;

  for (auto& buffer : buffers) {
    literals.push_back(std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape())));
    buffer->ToLiteral(literals.back().get()).OnReady([&](absl::Status s) {
      absl::MutexLock l(mu);
      TF_ASSERT_OK(s);
      ++got_literal_count;
    });
    buffer->GetReadyFuture().OnReady([&](absl::Status s) {
      absl::MutexLock l(mu);
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
    absl::MutexLock l(mu);
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

TEST(StreamExecutorGpuClientTest, FromHostAsyncPinnedHostChunked) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> lit,
                          buf->ToLiteral().Await());
  EXPECT_THAT(lit->data<float>(), ElementsAreArray(data));
}

TEST(StreamExecutorGpuClientTest, DeleteBufferThenFulfillBufferNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  std::vector<float> data{1, 3, 5, 7, 11, 13, 17, 19};
  Shape shape = ShapeUtil::MakeShape(F32, {static_cast<int64_t>(data.size())});
  // On ROCm 10k buffers hit vm.max_map_count causing pthread_create to fail
  const int num_buffers =
      client->platform_name() == xla::RocmName() ? 2000 : 10000;
  std::vector<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
      txms;
  for (int i = 0; i < num_buffers; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
        client->CreateBuffersForAsyncHostToDevice({shape}, memspace));
    std::unique_ptr<PjRtBuffer> buf = txm->RetrieveBuffer(0);
    ASSERT_THAT(buf->GetReadyFuture().IsReady(), Eq(false));
    txms.push_back(std::move(txm));
    // Delete the buffer
  }

  // At this point, we have num_buffers buffers pending deallocation.

  absl::string_view raw_view(reinterpret_cast<char*>(data.data()),
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

TEST(StreamExecutorGpuClientTest, CopyRawToHostFullBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst = tsl::port::AlignedMalloc(
      size, static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));

  auto result = buffer->CopyRawToHost(dst, 0, size);
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);
  EXPECT_EQ(*(static_cast<float*>(dst) + 1), 42.0f);

  tsl::port::AlignedSizedFree(
      dst, size,
      static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostSubBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst = tsl::port::AlignedMalloc(
      size, static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));

  auto result = buffer->CopyRawToHost(dst, 0, sizeof(float));
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);

  tsl::port::AlignedSizedFree(
      dst, size,
      static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostOutOfRange) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst = tsl::port::AlignedMalloc(
      size, static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));

  auto result = buffer->CopyRawToHost(dst, 1, size);
  EXPECT_THAT(result.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("invalid offset 1")));
  tsl::port::AlignedSizedFree(
      dst, size,
      static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));

  // The future returned by buffer->CopyRawToHost() may be resolve to an error
  // before the prior buffer->BufferFromHostLiteral() is done. Make sure
  // `literal` is alive long enough to avoid use-after-free. See the comment in
  // PjRtStreamExecutorBuffer::CopyRawToHost() for details.
  TF_EXPECT_OK(buffer->GetReadyFuture().Await());
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto literal = xla::LiteralUtil::CreateR1<float>({41.0f, 42.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  auto [dst_promise, dst_future] = xla::MakePromise<void*>();

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  auto ready = buffer->GetReadyFuture();
  auto result = buffer->CopyRawToHostFuture(dst_future, 0, size);

  // Drop the buffer before fulfilling `dst`. The transfer should still keep
  // the buffer alive.
  buffer.reset();
  ready.OnReady([dst_promise = std::move(dst_promise),
                 size](absl::Status status) mutable {
    void* dst = tsl::port::AlignedMalloc(
        size,
        static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));
    dst_promise.Set(dst);
  });

  TF_EXPECT_OK(result.Await());
  TF_ASSERT_OK_AND_ASSIGN(auto* dst, dst_future.Await());
  EXPECT_EQ(*(static_cast<float*>(dst)), 41.0f);
  EXPECT_EQ(*(static_cast<float*>(dst) + 1), 42.0f);

  tsl::port::AlignedSizedFree(
      dst, size,
      static_cast<std::align_val_t>(tsl::Allocator::kAllocatorAlignment));
}

TEST(StreamExecutorGpuClientTest, AsyncCopyToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  // d0 is the device we will perform local/remote sends from.
  auto* d0 = client->addressable_devices()[0];
  // d1 is the device we will perform local/remote recvs, where the recv
  // sync flag may be contended.
  auto* d1 = client->addressable_devices()[1];

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({src_literal.shape()},
                                                *d0->default_memory_space()));
  auto src_buffer = transfer_manager->RetrieveBuffer(0);
  // CopyToMemorySpace won't be enqueued until src_buffer is available.
  auto local_recv_buffer =
      *src_buffer->CopyToMemorySpace(*d1->default_memory_space());

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, []() {}));

  auto literal = std::make_shared<Literal>(src_literal.shape());

  auto local_recv_literal = local_recv_buffer->ToLiteral(literal.get());
  TF_EXPECT_OK(local_recv_literal.Await());

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, CopyErrorBufferToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto* src_device = client->addressable_devices()[0];
  auto* dst_device = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto* src_memory_space,
                          src_device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(auto* dst_memory_space,
                          dst_device->default_memory_space());

  TF_ASSERT_OK_AND_ASSIGN(
      auto send_buffer,
      client->CreateErrorBuffer(Internal("some error"),
                                ShapeUtil::MakeShape(U32, {3, 2}),
                                src_memory_space));

  TF_ASSERT_OK_AND_ASSIGN(auto recv_buffer,
                          send_buffer->CopyToMemorySpace(dst_memory_space));

  EXPECT_THAT(
      recv_buffer->ToLiteral().Await(),
      absl_testing::StatusIs(tsl::error::INTERNAL, HasSubstr("some error")));
}

TEST(StreamExecutorGpuClientTest, CopyDelayedErrorBufferToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto* src_device = client->addressable_devices()[0];
  auto* dst_device = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto* src_memory_space,
                          src_device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(auto* dst_memory_space,
                          dst_device->default_memory_space());

  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});

  TF_ASSERT_OK_AND_ASSIGN(auto alias_pair,
                          client->CreateAliasBuffer(shape, src_memory_space));
  auto& send_buffer = alias_pair.first;
  auto& fulfill_cb = alias_pair.second;

  TF_ASSERT_OK_AND_ASSIGN(auto recv_buffer,
                          send_buffer->CopyToMemorySpace(dst_memory_space));

  absl::SleepFor(absl::Seconds(3));

  absl::Status error = fulfill_cb(absl::InternalError("delayed error"));

  EXPECT_THAT(recv_buffer->ToLiteral().Await(), error);
}

TEST(StreamExecutorGpuClientTest, CreateMixOfErrorBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<float> data(i + 1);
    absl::c_iota(data, static_cast<float>(i + 10));
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
        absl::MutexLock l(mu);
        TF_ASSERT_OK(s);
        ++got_callback_count;
      });
    } else {
      absl::Status error = Internal("error %d", i);
      transfer_manager->SetBufferError(i, error);
      buffer->GetReadyFuture().OnReady(
          [error, &mu, &got_callback_count](absl::Status s) {
            absl::MutexLock l(mu);
            ASSERT_EQ(s, error);
            ++got_callback_count;
          });
    }
    buffer.reset();
  }

  {
    auto done = [&]() { return got_callback_count == src_literals.size(); };
    absl::MutexLock l(mu);
    QCHECK(mu.AwaitWithTimeout(absl::Condition(&done), absl::Seconds(60)));
  }
}

TEST(GpuTopology, FromProto) {
  GpuTopologyProto msg;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        platform_version: "platform_version"
        num_partitions: 2
        num_hosts_per_partition: 1
        num_devices_per_host: 3
      )pb",
      &msg));

  std::unique_ptr<const GpuTopology> gpu_topology = GpuTopology::FromProto(msg);
  EXPECT_THAT(gpu_topology->platform_version(), "platform_version");
  EXPECT_THAT(gpu_topology->num_partitions(), 2);
  EXPECT_THAT(gpu_topology->num_hosts_per_partition(), 1);
  EXPECT_THAT(gpu_topology->num_devices_per_host(), 3);
}

TEST(GpuTopology, ToProto) {
  GpuTopology gpu_topology(
      /*platform_version=*/"platform_version",
      /*num_partitions=*/2,
      /*num_hosts_per_partition=*/1,
      /*num_devices_per_host=*/3);
  GpuTopologyProto msg = gpu_topology.ToProto();
  EXPECT_THAT(msg.platform_version(), "platform_version");
  EXPECT_THAT(msg.num_partitions(), 2);
  EXPECT_THAT(msg.num_hosts_per_partition(), 1);
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
      EXPECT_TRUE(client->platform_name() == xla::CudaName() ||
                  client->platform_name() == xla::RocmName());
      EXPECT_EQ(client->addressable_device_count(), 2);
      EXPECT_EQ(client->device_count(), 4);
    });
  }
}

TEST(StreamExecutorGpuClientTest, GetAllocatorStatsTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  for (auto device : client->addressable_devices()) {
    const xla::Literal literal = xla::LiteralUtil::CreateR0<int32_t>(0);
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> buffer,
        client->BufferFromHostLiteral(literal, memory_space));
    TF_ASSERT_OK(buffer->GetReadyFuture().Await());

    auto stats = device->GetAllocatorStats();
    TF_ASSERT_OK(stats.status());
    ASSERT_GT(stats.value().peak_bytes_in_use, 0);
  }
}

TEST(StreamExecutorGpuClientTest, GpuDeviceDescriptionTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  for (int device_index = 0; device_index < client->device_count();
       device_index++) {
    auto device =
        static_cast<PjRtStreamExecutorDevice*>(client->devices()[device_index]);
    auto coords = device->description().coords();
    // All devices are in the same partition & process.
    EXPECT_THAT(coords, ElementsAre(0, 0, device->local_device_id().value()));
  }
}

TEST(StreamExecutorGpuClientTest, GpuDeviceSharedMemoryInfo) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  for (const auto& device : client->devices()) {
    auto value = static_cast<PjRtStreamExecutorDevice*>(device)
                     ->description()
                     .Attributes()
                     .find("shared_memory_per_block_optin")
                     ->second;
    int64_t shared_memory_per_block_optin = std::get<int64_t>(value);
    EXPECT_GT(shared_memory_per_block_optin, 0);
  }
}

TEST(StreamExecutorGpuClientTest, GetTopologyDescriptionWithGlobalDevicesTest) {
  const int num_nodes = 4;
  GpuClientOptions options;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  options.mock_gpu_topology = "2x2x2";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  int devices_per_host = client->addressable_device_count();

  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());

  std::vector<std::unique_ptr<const PjRtDeviceDescription>>
      device_descriptions = topology->DeviceDescriptions();
  EXPECT_EQ(client->device_count(), device_descriptions.size());

  for (const auto& device_description : device_descriptions) {
    EXPECT_EQ(device_description->process_index(),
              device_description->id() / devices_per_host);
  }
}

TEST(PjRtCpuClientTest, CopyToMemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  for (auto* memory_space : client->memory_spaces()) {
    xla::Shape shape = xla::ShapeUtil::MakeShape(S32, {128, 256});
    TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer, client->BufferFromHostLiteral(literal, memory_space));
    TF_ASSERT_OK_AND_ASSIGN(buffer,
                            buffer->CopyToMemorySpace(buffer->memory_space()));
    TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteral().Await());
    EXPECT_THAT(received_literal->data<int32_t>(),
                ElementsAreArray(literal.data<int32_t>()));
  }
}

TEST(StreamExecutorGpuClientTest, MockNcclClientTest) {
  GpuClientOptions options = DefaultOptions();
  const int num_nodes = 4;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2);
  EXPECT_EQ(client->device_count(), devices_per_host * num_nodes);
  for (int i = 0; i < client->device_count(); i++) {
    auto device = client->devices()[i];
    auto partition_index = std::get<int64_t>(
        device->description().Attributes().at("partition_index"));
    auto host_index = device->process_index();
    EXPECT_EQ(partition_index, host_index);
  }
}

TEST(StreamExecutorGpuClientTest, ShouldStageHostToDeviceTransfersSetToTrue) {
  GpuClientOptions options_staging = DefaultOptions();
  options_staging.should_stage_host_to_device_transfers = true;
  TF_ASSERT_OK_AND_ASSIGN(auto client_staging,
                          GetStreamExecutorGpuClient(options_staging));

  std::vector<float> data(1024, 1.0f);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});

  auto* staging_client =
      absl::down_cast<StreamExecutorGpuClient*>(client_staging.get());

  EXPECT_TRUE(staging_client->ShouldStageHostToDeviceTransfers(
      data.data(), sizeof(float) * data.size()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client_staging->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr,
          client_staging->addressable_devices()[0]->memory_spaces()[0],
          /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      *literal, LiteralUtil::CreateR1<float>(std::vector<float>(1024, 1.0f))));
}

TEST(StreamExecutorGpuClientTest, ShouldStageHostToDeviceTransfersSetToFalse) {
  GpuClientOptions options_no_staging = DefaultOptions();
  options_no_staging.should_stage_host_to_device_transfers = false;
  TF_ASSERT_OK_AND_ASSIGN(auto client_no_staging,
                          GetStreamExecutorGpuClient(options_no_staging));

  std::vector<float> data(1024, 1.0f);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});

  auto* no_staging_client =
      absl::down_cast<StreamExecutorGpuClient*>(client_no_staging.get());

  EXPECT_FALSE(no_staging_client->ShouldStageHostToDeviceTransfers(
      data.data(), sizeof(float) * data.size()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client_no_staging->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr,
          client_no_staging->addressable_devices()[0]->memory_spaces()[0],
          /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      *literal, LiteralUtil::CreateR1<float>(std::vector<float>(1024, 1.0f))));
}

TEST(StreamExecutorGpuClientTest, MockNcclClientWithGpuTopologyTest) {
  GpuClientOptions options = DefaultOptions();
  options.enable_mock_nccl = true;
  options.num_nodes = 8;
  options.mock_gpu_topology = "2x4x2";
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  TF_ASSERT_OK_AND_ASSIGN(const xla::PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  const StreamExecutorGpuTopologyDescription& gpu_topology =
      absl::down_cast<const StreamExecutorGpuTopologyDescription&>(*topology);

  EXPECT_EQ(gpu_topology.gpu_topology().num_partitions(), 2);
  EXPECT_EQ(gpu_topology.gpu_topology().num_hosts_per_partition(), 4);
  EXPECT_EQ(gpu_topology.gpu_topology().num_devices_per_host(), 2);
}

constexpr char kMlirDistributedSum[] = R"(
module @jit_f attributes {mhlo.num_partitions = 8 : i32,
                          mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8xi32> {
      mhlo.layout_mode = "default",
      mhlo.sharding = "{devices=[8]0,1,2,3,4,5,6,7}"}) -> (tensor<i32> {
          jax.result_info = "",
          mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.reduce(%arg0 init: %c)
        applies stablehlo.add across dimensions = [0]
            : (tensor<8xi32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
})";

TEST(StreamExecutorGpuClientTest, MockNcclClientWithGpuTopologyExecuteTest) {
  GpuClientOptions client_options = DefaultOptions();
  client_options.enable_mock_nccl = true;
  client_options.num_nodes = 4;
  client_options.mock_gpu_topology = "2x2x2";
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(client_options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, xla::ParseMlirModuleString(kMlirDistributedSum, *context));

  xla::CompileOptions options;
  options.executable_build_options.set_num_partitions(8)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)),
          options));

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1}, {0});
  std::vector<std::unique_ptr<PjRtBuffer>> inputs;
  std::vector<std::vector<PjRtBuffer*>> input_ptrs;
  for (int i = 0; i < devices_per_host; i++) {
    auto device = client->addressable_devices()[i];
    std::vector<int32_t> data{i};
    TF_ASSERT_OK_AND_ASSIGN(
        auto input,
        client->BufferFromHostBuffer(
            data.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr,
            *device->default_memory_space(), /*device_layout=*/nullptr));
    input_ptrs.push_back({input.get()});
    inputs.push_back(std::move(input));
  }

  // Test that running the program does not crash/hang.
  TF_ASSERT_OK(
      executable->Execute(absl::MakeSpan(input_ptrs), ExecuteOptions()));
}

TEST(StreamExecutorGpuClientTest, MockNcclClientWithGpuTopologyMismatchTest) {
  GpuClientOptions options = DefaultOptions();
  options.enable_mock_nccl = true;
  options.num_nodes = 16;
  options.mock_gpu_topology = "2x4";
  EXPECT_FALSE(GetStreamExecutorGpuClient(options).ok());
}

TEST(StreamExecutorGpuClientTest, BufferFromHostBufferPinnedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(StreamExecutorGpuClientTest, CopyToPinnedHostMemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(StreamExecutorGpuClientTest, CopyFromPinnedHostMemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  auto device = client->addressable_devices()[0];
  auto* device_memory_space = *device->default_memory_space();
  auto* pinned_memory_space = device->memory_spaces()[1];
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          pinned_memory_space, /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(buffer->IsOnCpu());

  EXPECT_EQ(pinned_memory_space->kind_id(), PinnedHostMemorySpace::kKindId);
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(device_memory_space));

  EXPECT_EQ(result->memory_space()->kind(), "device");

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(StreamExecutorGpuClientTest, CopyToPinnedHostMemorySpaceInt4) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
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

  auto* pinned_memory_space = device->memory_spaces()[1];
  EXPECT_EQ(pinned_memory_space->kind_id(), PinnedHostMemorySpace::kKindId);
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(pinned_memory_space));

  EXPECT_EQ(result->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(result->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());
  std::vector<xla::s4> expected{xla::s4(1), xla::s4(2), xla::s4(3), xla::s4(4)};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<xla::s4>(expected),
                                     *literal));
}

TEST(StreamExecutorGpuClientTest, OpaqueDeviceMemoryDataPointer) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  PjRtDevice* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      device->memory_space_by_kind(PinnedHostMemorySpace::kKind));

  // Create a pinned_host buffer
  std::vector<float> float_data{12.0, 34.0, 56.0, 78.0};
  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buf,
      client->BufferFromHostBuffer(
          static_cast<const void*>(float_data.data()), shape.element_type(),
          shape.dimensions(), /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memspace,
          /*device_layout=*/nullptr));
  ASSERT_THAT(buf->IsOnCpu(), true);
  TF_ASSERT_OK_AND_ASSIGN(size_t buf_sz, buf->GetOnDeviceSizeInBytes());
  ASSERT_THAT(buf_sz, Ge(sizeof(float) * 4));

  // Check that OpaqueDeviceMemoryDataPointer() points to actual data
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer::ExternalReference> ref,
                          buf->AcquireExternalReference());
  TF_ASSERT_OK(buf->GetReadyFuture().Await());
  const float* float_ptr =
      reinterpret_cast<const float*>(ref->OpaqueDeviceMemoryDataPointer());
  EXPECT_THAT(*float_ptr, FloatEq(12.0));
  EXPECT_THAT(*(float_ptr + 1), FloatEq(34.0));
  EXPECT_THAT(*(float_ptr + 2), FloatEq(56.0));
  EXPECT_THAT(*(float_ptr + 3), FloatEq(78.0));

  // Copy raw to device using OpaqueDeviceMemoryDataPointer(), and then read
  // back to host; expect to get back the same data
  TF_ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * default_ms,
                          device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
      client->CreateBuffersForAsyncHostToDevice({shape}, default_ms));
  TF_ASSERT_OK(txm->TransferRawDataToBuffer(
      /*buffer_index=*/0,
      absl::string_view(
          static_cast<const char*>(ref->OpaqueDeviceMemoryDataPointer()),
          buf_sz),
      /*on_done=*/[]() {}));
  std::unique_ptr<PjRtBuffer> hbm_buf = txm->RetrieveBuffer(0);
  EXPECT_THAT(hbm_buf->GetOnDeviceSizeInBytes(),
              absl_testing::IsOkAndHolds(buf_sz));
  EXPECT_THAT(hbm_buf->HostShape(), absl_testing::IsOkAndHolds(shape));
  TF_ASSERT_OK(hbm_buf->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          hbm_buf->ToLiteral().Await());
  EXPECT_THAT(literal->data<float>(), ElementsAreArray(float_data));
}

namespace {

absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateDeviceBufferForTest(
    xla::PjRtClient* client) {
  auto device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());

  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  TF_ASSIGN_OR_RETURN(
      auto input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
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

constexpr char const* kCollectiveMemorySpaceOutput = R"(

  HloModule jit__psum, entry_computation_layout={(s32[1,4]{1,0})->s32[4]{0}}

  region_0.3 {
    Arg_0.0 = s32[] parameter(0)
    Arg_1.0 = s32[] parameter(1)
    ROOT add.0 = s32[] add(Arg_0.0, Arg_1.0)
  }

  ENTRY main.10_spmd {
    param = s32[1,4]{1,0} parameter(0)
    reshape = s32[4]{0} reshape(param)
    ROOT all-reduce = s32[4]{0} all-reduce(reshape), channel_id=1, to_apply=region_0.3
  }

)";

}  // namespace

TEST(StreamExecutorGpuClientTest, ExecutePinnedHostOutputTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateDeviceBufferForTest(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, ExecuteOptions()));

  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "pinned_host");
  TF_ASSERT_OK(result_buffers[0]->GetReadyFuture().Await());

  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_stats, executable->GetExecutable()->GetCompiledMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 16);
  EXPECT_GE(memory_stats.peak_memory_in_bytes, 0);
}

TEST(StreamExecutorGpuClientTest, ExecutePinnedHostOutputTupleTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateDeviceBufferForTest(client.get()));

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, execute_options));

  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  TF_ASSERT_OK(result_buffers[0]->GetReadyFuture().Await());
  EXPECT_EQ(result_buffers.size(), 2);
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  EXPECT_EQ(result_buffers[1]->memory_space()->kind(), "pinned_host");
}

TEST(StreamExecutorGpuClientTest, ExecutablePinnedHostOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetExecutable()->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "pinned_host");
}

TEST(StreamExecutorGpuClientTest,
     GetCompiledMemoryStatsWithTupleAndNcclUserBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  constexpr char const* kProgramWithCollectiveAndTuple = R"(
 HloModule test

 region_0 {
   Arg_0 = f32[] parameter(0)
   Arg_1 = f32[] parameter(1)
   ROOT add = f32[] add(Arg_0, Arg_1)
 }

 ENTRY main {
   p0 = f32[512,128]{1,0} parameter(0)
   p1 = f32[512,32,128]{2,1,0} parameter(1)
   p2 = f32[512,8,128]{2,1,0} parameter(2)
   p3 = f32[512,14336]{1,0} parameter(3)
   p4 = f32[1024]{0} parameter(4)
   p5 = f32[1]{0} parameter(5)

   // All-gather operations that will use memory space 1 with NCCL user buffers
   ag0 = f32[4096,128]{1,0} all-gather(p0), channel_id=1, replica_groups=[1,8]<=[8], dimensions={0}, use_global_device_ids=true
   ag1 = f32[4096,32,128]{2,1,0} all-gather(p1), channel_id=2, replica_groups=[1,8]<=[8], dimensions={0}, use_global_device_ids=true
   ag2 = f32[4096,8,128]{2,1,0} all-gather(p2), channel_id=3, replica_groups=[1,8]<=[8], dimensions={0}, use_global_device_ids=true
   ag3 = f32[4096,14336]{1,0} all-gather(p3), channel_id=4, replica_groups=[1,8]<=[8], dimensions={0}, use_global_device_ids=true

   ar0 = f32[1024]{0} all-reduce(p4), channel_id=5, to_apply=region_0
   ar1 = f32[1]{0} all-reduce(p5), channel_id=6, to_apply=region_0

   // Regular operations with default memory space
   add0 = f32[512,128]{1,0} add(p0, p0)
   add1 = f32[512,32,128]{2,1,0} add(p1, p1)
   add2 = f32[512,8,128]{2,1,0} add(p2, p2)

   // Mix of all-gather results (memory space 1) and regular tensors (memory space 0)
   ROOT tuple = (f32[4096,128]{1,0}, f32[4096,32,128]{2,1,0}, f32[4096,8,128]{2,1,0}, f32[4096,14336]{1,0},
                 f32[1024]{0}, f32[1]{0}, f32[1024]{0}, f32[1]{0},
                 f32[512,128]{1,0}, f32[512,32,128]{2,1,0}, f32[512,8,128]{2,1,0}, f32[512,14336]{1,0},
                 f32[4096,128]{1,0}, f32[4096,32,128]{2,1,0}, f32[4096,8,128]{2,1,0}, f32[4096,14336]{1,0},
                 f32[1024]{0}, f32[1]{0}, f32[1024]{0}, f32[1]{0},
                 f32[512,128]{1,0}, f32[512,32,128]{2,1,0}, f32[512,8,128]{2,1,0}, f32[512,14336]{1,0},
                 f32[4096,128]{1,0}, f32[4096,32,128]{2,1,0}, f32[4096,8,128]{2,1,0}, f32[4096,14336]{1,0},
                 f32[1024]{0}, f32[1]{0}, f32[1024]{0}, f32[1]{0},
                 f32[512,128]{1,0}, f32[512,32,128]{2,1,0}, f32[512,8,128]{2,1,0}, f32[512,14336]{1,0},
                 f32[4096,128]{1,0}, f32[4096,32,128]{2,1,0}, f32[4096,8,128]{2,1,0}, f32[4096,14336]{1,0},
                 f32[1024]{0}, f32[1]{0}, f32[1024]{0}, f32[1]{0},
                 f32[512,128]{1,0}, f32[512,32,128]{2,1,0}, f32[512,8,128]{2,1,0}, f32[512,14336]{1,0},
                 f32[4096,128]{1,0}, f32[4096,32,128]{2,1,0}, f32[4096,8,128]{2,1,0}, f32[4096,14336]{1,0})
                tuple(ag0, ag1, ag2, ag3, ar0, ar1, ar0, ar1,
                      p0, p1, p2, p3, ag0, ag1, ag2, ag3,
                      ar0, ar1, ar0, ar1, add0, add1, add2, p3,
                      ag0, ag1, ag2, ag3, ar0, ar1, ar0, ar1,
                      p0, p1, p2, p3, ag0, ag1, ag2, ag3,
                      ar0, ar1, ar0, ar1, add0, add1, add2, p3,
                      ag0, ag1, ag2, ag3)
 }
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kProgramWithCollectiveAndTuple, *client, options));

  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_stats, executable->GetExecutable()->GetCompiledMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 1764786624);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  // Difference in buffer aliasing causes a difference in peak memory usage
  if (client->platform_name() == xla::RocmName()) {
    EXPECT_EQ(memory_stats.peak_memory_in_bytes, 1845006788);
  } else {
    EXPECT_EQ(memory_stats.peak_memory_in_bytes, 1845010888);
  }
}

TEST(StreamExecutorGpuClientTest, GetCompiledMemoryStatsMixedTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  constexpr char const* kSimpleMixedTupleHlo = R"(
HloModule test

region_0 {
Arg_0 = f32[] parameter(0)
Arg_1 = f32[] parameter(1)
ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY main {
p0 = f32[2]{0} parameter(0)
// All-gather across 8 replicas to enlarge dim0.
ag = f32[16]{0} all-gather(p0), channel_id=1, replica_groups=[1,8]<=[8], dimensions={0}, use_global_device_ids=true
add0 = f32[2]{0} add(p0, p0)
ROOT tuple = (f32[16]{0}, f32[2]{0}, f32[2]{0}) tuple(ag, p0, add0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kSimpleMixedTupleHlo, *client, options));

  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_stats, executable->GetExecutable()->GetCompiledMemoryStats());

  EXPECT_EQ(memory_stats.output_size_in_bytes, 104);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.peak_memory_in_bytes, 120);
}

TEST(StreamExecutorGpuClientTest, GetCompiledMemoryStatsMixedTupleNotRoot) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  constexpr char const* kMixedTupleNotRootHlo = R"(
HloModule test

ENTRY main {
p0 = f32[2]{0} parameter(0)
ag = f32[16]{0} all-gather(p0), channel_id=1, replica_groups=[1,8]<=[8], dimensions={0}, use_global_device_ids=true
add0 = f32[2]{0} add(p0, p0)
t = (f32[16]{0}, f32[2]{0}, f32[2]{0}) tuple(ag, p0, add0)
ROOT gte0 = f32[16]{0} get-tuple-element(t), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kMixedTupleNotRootHlo, *client, options));

  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_stats, executable->GetExecutable()->GetCompiledMemoryStats());

  EXPECT_EQ(memory_stats.output_size_in_bytes, 64);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.peak_memory_in_bytes, 80);
}

TEST(StreamExecutorGpuClientTest, GetCompiledMemoryStatsCountTupleTable) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  constexpr char const* kManyTuplesHlo = R"(
HloModule test

ENTRY main {
p0 = f32[1]{0} parameter(0)
add0 = f32[1]{0} add(p0, p0)
ROOT t = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
         f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0})
 tuple(p0, add0, p0, add0,
       p0, add0, p0, add0,
       p0, add0, p0, add0,
       p0, add0, p0, add0,
       p0, add0, p0, add0,
       p0, add0, p0, add0,
       p0, add0, p0, add0,
       p0, add0, p0, add0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kManyTuplesHlo, *client));
  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_stats, executable->GetExecutable()->GetCompiledMemoryStats());

  EXPECT_EQ(memory_stats.output_size_in_bytes, 384);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.peak_memory_in_bytes, 388);
}

// Verify the output device memory kind with collective memory space shape
// when NCCL user buffer is enabled.
TEST(StreamExecutorGpuClientTest,
     ExecutableCollectiveMemoryOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kCollectiveMemorySpaceOutput, *client, options));
  std::vector<int32_t> data{1, 2, 3, 4};
  // Build the input shape with the correct memory space set.
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1, 4},
                                                    /*minor_to_major=*/{1, 0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);

  auto device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetExecutable()->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "device");

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, ExecuteOptions()));
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  TF_ASSERT_OK(result_buffers[0]->GetReadyFuture().Await());
  Shape result_shape = result_buffers[0]->on_device_shape();
  auto memory_space = result_shape.layout().memory_space();
  EXPECT_EQ(memory_space, 1);
}

TEST(StreamExecutorGpuClientTest, CollectiveMemorySpaceSmoke) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  xla::CompileOptions opts;
  opts.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto exe, CompileExecutable(kCollectiveMemorySpaceOutput, *client, opts));

  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1, 4}, {1, 0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);
  auto* device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          exe->Execute({{input.get()}}, ExecuteOptions()));
  auto& buf = results[0][0];
  TF_ASSERT_OK(buf->GetReadyFuture().Await());

  // Override default memory space to collective memory space.
  EXPECT_EQ(buf->on_device_shape().layout().memory_space(),
            (int)gpu::MemorySpaceColor::kCollective);
}

TEST(StreamExecutorGpuClientTest,
     ExecutablePinnedHostTupleOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

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
                          executable->GetExecutable()->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 2);
  EXPECT_EQ(memory_kinds[0][0], "device");
  EXPECT_EQ(memory_kinds[0][1], "pinned_host");
}

TEST(StreamExecutorGpuClientTest, MlirParameterHostMemorySpaceIsSetInHlo) {
  constexpr char kMlirH2D[] =
      R"(
    func.func public @main(%arg0: tensor<8x2xi32> {
            mhlo.layout_mode = "{1,0}",
            mhlo.memory_kind = "pinned_host",
            mhlo.sharding = "{devices=[2,2]<=[4]}"
        }) -> (tensor<8x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default",
            mhlo.memory_kind = "device",
            mhlo.sharding = "{devices=[2,2]<=[4]}"}) {
      %0 = stablehlo.custom_call @annotate_device_placement(%arg0) {
              has_side_effect = true,
              mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      return %0 : tensor<8x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseMlirModuleString(kMlirH2D, *context));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules,
                          executable->GetExecutable()->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kHostMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout.memory_space(), Layout::kDefaultMemorySpace);
}

TEST(StreamExecutorGpuClientTest, MlirResultHostMemorySpaceIsSetInHlo) {
  constexpr char kMlirD2H[] =
      R"(
    func.func public @main(%arg0: tensor<8x2xi32> {
            mhlo.layout_mode = "{1,0}",
            mhlo.memory_kind = "device",
            mhlo.sharding = "{devices=[2,2]<=[4]}"
        }) -> (tensor<8x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default",
            mhlo.memory_kind = "pinned_host",
            mhlo.sharding = "{devices=[2,2]<=[4]}"}) {
      %0 = stablehlo.custom_call @annotate_device_placement(%arg0) {
              has_side_effect = true,
              mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      return %0 : tensor<8x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseMlirModuleString(kMlirD2H, *context));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules,
                          executable->GetExecutable()->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kDefaultMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout.memory_space(), Layout::kHostMemorySpace);
}

TEST(StreamExecutorGpuClientTest, ProfileExecution) {
  static constexpr char const* kProgram = R"(
    HloModule profiled
      ENTRY main {
      c0 = f32[] constant(20)
      c1 = f32[] constant(21)
      ROOT res = f32[] add(c0, c1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));
  ExecutionProfile profile;
  ExecuteOptions opts;
  opts.execution_profile = &profile;
  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute(/*argument_handles=*/{{}}, opts));
  TF_ASSERT_OK(results[0][0]->GetReadyFuture().Await());
  EXPECT_GT(profile.compute_time_ns(), 0);
}

TEST(StreamExecutorGpuClientTest, MlirAutoResultLayoutIsSet) {
  constexpr char kMlirWithParameterLayout[] =
      R"(
    func.func public @main(%arg0: tensor<2x4x2xi32> {
            mhlo.layout_mode = "{2, 1, 0}"
        }) -> (tensor<2x2x4xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "auto"}) {
      %0 = stablehlo.transpose %arg0, dims = [0, 2, 1]
          : (tensor<2x4x2xi32>) -> tensor<2x2x4xi32>
      return %0 : tensor<2x2x4xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(auto module, xla::ParseMlirModuleString(
                                           kMlirWithParameterLayout, *context));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules,
                          executable->GetExecutable()->GetHloModules());

  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout, Layout({1, 2, 0}));
}

TEST(StreamExecutorGpuClientTest, MlirAutoParameterLayoutIsSet) {
  constexpr char kMlirWithParameterLayout[] =
      R"(
    func.func public @main(%arg0: tensor<2x4x2xi32> {
            mhlo.layout_mode = "auto"
        }) -> (tensor<2x2x4xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "{2, 1, 0}"}) {
      %0 = stablehlo.transpose %arg0, dims = [0, 2, 1]
          : (tensor<2x4x2xi32>) -> tensor<2x2x4xi32>
      return %0 : tensor<2x2x4xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(auto module, xla::ParseMlirModuleString(
                                           kMlirWithParameterLayout, *context));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules,
                          executable->GetExecutable()->GetHloModules());

  // Check that the executable can be serialized.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                          executable->GetExecutable()->SerializeExecutable());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({1, 2, 0}));
}

TEST(StreamExecutorGpuClientTest, ValidatesClientName) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

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
  auto gpu_exe = static_cast<PjRtStreamExecutorLoadedExecutable*>(
      std::move(executable).get());
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          gpu_exe->SerializeExecutable());

  ExecutableAndOptionsProto proto;
  ASSERT_OK(ReadSplitProto(
      std::make_unique<riegeli::StringReader<>>(serialized), proto));
  EXPECT_EQ(proto.pjrt_client_name(), "PjRtStreamExecutorClient");
  proto.set_pjrt_client_name("SomeGpuClient");
  std::string modified_serialized;
  ASSERT_OK(WriteSplitExecutableAndOptions(
      proto, std::make_unique<riegeli::StringWriter<>>(&modified_serialized)));

  EXPECT_THAT(client->DeserializeExecutable(modified_serialized, std::nullopt),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  HasSubstr("PjRt client type expected by the serialized "
                            "executable: SomeGpuClient")));
}

TEST(StreamExecutorGpuClientTest, MlirParameterLayoutIsSetInHlo) {
  constexpr char kMlirWithParameterLayout[] =
      R"(
    func.func public @main(%arg0: tensor<2x2x2xi32> {
            mhlo.layout_mode = "{0, 2, 1}"
        }) -> (tensor<2x2x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default"}) {
      return %arg0 : tensor<2x2x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(auto module, xla::ParseMlirModuleString(
                                           kMlirWithParameterLayout, *context));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules,
                          executable->GetExecutable()->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({0, 2, 1}));
}

TEST(StreamExecutorGpuClientTest, MlirParameterLayoutFromOptionsIsSetInHlo) {
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

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseMlirModuleString(kMlirCopy, *context));

  xla::CompileOptions options;
  options.argument_layouts = {
      {ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 2, 2}, {0, 2, 1})}};
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, client->Compile(MaybeOwningMlirModule(std::move(context),
                                                             std::move(module)),
                                       options));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({0, 2, 1}));
}

TEST(StreamExecutorGpuClientTest,
     MlirResultHostMemorySpaceIsSetInHloWithShardingPropagation) {
  constexpr absl::string_view mlir_mul_explicit_sharding_layout_and_memory =
      R"mlir(
  module @jit_f attributes {
      mhlo.num_partitions = 2 : i32,
      mhlo.num_replicas = 1 : i32
  } {
    func.func public @main(%arg0: tensor<8x2xi32> {
            mhlo.layout_mode = "{1,0}",
            mhlo.memory_kind = "device",
            mhlo.sharding = "{devices=[1,2]<=[2]}"
        }) -> (tensor<8x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "{0,1}",
            mhlo.memory_kind = "pinned_host"
        }) {
      %c = stablehlo.constant dense<2> : tensor<i32>
      %0 = stablehlo.broadcast_in_dim %c, dims = []
          : (tensor<i32>) -> tensor<8x2xi32>
      %1 = stablehlo.multiply %arg0, %0 : tensor<8x2xi32>
      %2 = stablehlo.custom_call @Sharding(%1) {
              mhlo.sharding = "{devices=[1,2]<=[2]}"
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      %3 = stablehlo.custom_call @annotate_device_placement(%2) {
              has_side_effect = true,
              mhlo.frontend_attributes = {
                  _xla_buffer_placement = "pinned_host"
              }
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      return %3 : tensor<8x2xi32>
    }
  })mlir";

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, xla::ParseMlirModuleString(
                       mlir_mul_explicit_sharding_layout_and_memory, *context));
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  xla::CompileOptions options;
  options.executable_build_options.set_num_partitions(2)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)),
          options));
  TF_ASSERT_OK_AND_ASSIGN(auto modules,
                          executable->GetExecutable()->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kDefaultMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout,
            Layout({0, 1}).set_memory_space(Layout::kHostMemorySpace));

  // Verify that the executable's layout callback is null.
  // This is necessary for the executable to be serializable.
  EXPECT_EQ(executable->GetExecutable()
                ->GetCompileOptions()
                .value()
                .executable_build_options.layout_canonicalization_callback(),
            nullptr);
}

TEST(StreamExecutorGpuClientTest, GetDefaultLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto shape = ShapeUtil::MakeShape(S4, {2, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      auto layout,
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));
  EXPECT_EQ(layout.element_size_in_bits(), 4);

  TF_ASSERT_OK_AND_ASSIGN(auto* const topology,
                          client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      layout,
      topology->GetDefaultLayout(shape.element_type(), shape.dimensions()));
  EXPECT_EQ(layout.element_size_in_bits(), 4);
}

TEST(StreamExecutorGpuClientTest, AutoLayoutIsSupported) {
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

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_pjrt_allow_auto_layout_in_hlo(true);
  XlaComputation computation = m->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->CompileAndLoad(computation, compile_options));
  TF_ASSERT_OK_AND_ASSIGN(auto layouts,
                          executable->GetExecutable()->GetParameterLayouts());
  // Check that the assigned layouts are not default.
  EXPECT_NE(layouts[0]->ToString(), "{2,1,0}");
  EXPECT_NE(layouts[1]->ToString(), "{2,1,0}");
}

// Same test as SendRecvChunked, but check non-zero GPU device time measurement.
TEST(StreamExecutorGpuClientTest, NonZeroGPUDeviceTimeMeasurementSingleGPU) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP()
        << "DeviceTimeMeasurement implementation isn't available in OSS.";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

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
        CHECK_OK(stream->AddChunk(std::move(chunk0)).Await());

        auto chunk1 = PjRtChunk::AllocateDefault(sizeof(float));
        *reinterpret_cast<float*>(chunk1.data()) = 6.0f;
        CHECK_OK(stream->AddChunk(std::move(chunk1)).Await());

        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  // Test non-zero GPU device time measurement.
  auto measurement0 = CreateDeviceTimeMeasurement();
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_EQ(sent_value[0], 2.0f);
  EXPECT_EQ(sent_value[1], 3.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<float>({5.0f, 6.0f}),
                                     *result_literal));

  // Check measurement after execution completes.
  EXPECT_GT(
      measurement0->GetTotalDuration(DeviceTimeMeasurement::DeviceType::kGpu),
      absl::ZeroDuration());
}

// Same test as MockNcclClientWithGpuTopologyExecuteTest, but check non-zero
// GPU device time measurement.
TEST(StreamExecutorGpuClientTest, NonZeroGPUDeviceTimeMeasurementMultiGPU) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP()
        << "DeviceTimeMeasurement implementation isn't available in OSS.";
  }
  GpuClientOptions client_options;
  client_options.enable_mock_nccl = true;
  client_options.num_nodes = 4;
  client_options.mock_gpu_topology = "2x2x2";
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(client_options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, xla::ParseMlirModuleString(kMlirDistributedSum, *context));

  xla::CompileOptions options;
  options.executable_build_options.set_num_partitions(8)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)),
          options));

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1}, {0});
  std::vector<std::unique_ptr<PjRtBuffer>> inputs;
  std::vector<std::vector<PjRtBuffer*>> input_ptrs;
  for (int i = 0; i < devices_per_host; i++) {
    auto device = client->addressable_devices()[i];
    std::vector<int32_t> data{i};
    TF_ASSERT_OK_AND_ASSIGN(
        auto input,
        client->BufferFromHostBuffer(
            data.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr,
            *device->default_memory_space(), /*device_layout=*/nullptr));
    input_ptrs.push_back({input.get()});
    inputs.push_back(std::move(input));
  }

  // Test non-zero GPU device time measurement.
  auto measurement0 = CreateDeviceTimeMeasurement();

  // Test that running the program does not crash/hang.
  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      executable->Execute(absl::MakeSpan(input_ptrs), ExecuteOptions()));
  TF_ASSERT_OK(res[0][0]->GetReadyFuture().Await());

  // Check measurement after execution completes.
  EXPECT_GT(
      measurement0->GetTotalDuration(DeviceTimeMeasurement::DeviceType::kGpu),
      absl::ZeroDuration());
}

TEST(StreamExecutorGpuClientTest, DmaMapUnmap) {
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto client = absl::down_cast<PjRtStreamExecutorClient*>(gpu_client.get());
  size_t dma_size = 1024;
  size_t alignment = 4096;
  auto host_dma_ptr = tsl::port::AlignedMalloc(
      dma_size, static_cast<std::align_val_t>(alignment));
  auto host_dma_ptr_cleanup =
      absl::Cleanup([host_dma_ptr, dma_size, alignment] {
        tsl::port::AlignedSizedFree(host_dma_ptr, dma_size,
                                    static_cast<std::align_val_t>(alignment));
      });
  TF_EXPECT_OK(client->DmaMap(host_dma_ptr, dma_size));
  EXPECT_TRUE(client->IsDmaMapped(host_dma_ptr, dma_size));
  EXPECT_FALSE(
      client->IsDmaMapped(reinterpret_cast<char*>(host_dma_ptr) + 5, dma_size));
  TF_EXPECT_OK(client->DmaUnmap(host_dma_ptr));
  EXPECT_FALSE(client->IsDmaMapped(host_dma_ptr, dma_size));
}

TEST(StreamExecutorGpuClientTest, MultipleDeviceShareDmaMapping) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  ASSERT_GE(client->devices().size(), 2);

  size_t test_length = 512 * 1024;
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
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          first_device->memory_spaces()[0], /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, first_buffer->GetOnDeviceSizeInBytes());

  size_t dma_size = 2 * 1024 * 1024;
  size_t alignment = 1024;
  auto host_dma_ptr = tsl::port::AlignedMalloc(
      dma_size, static_cast<std::align_val_t>(alignment));
  auto host_dma_ptr_cleanup =
      absl::Cleanup([host_dma_ptr, dma_size, alignment] {
        tsl::port::AlignedSizedFree(host_dma_ptr, dma_size,
                                    static_cast<std::align_val_t>(alignment));
      });
  TF_EXPECT_OK(client->DmaMap(host_dma_ptr, dma_size));

  auto result = first_buffer->CopyRawToHost(host_dma_ptr, 0, size);
  TF_EXPECT_OK(result.Await());

  PjRtDevice* const second_device = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              {shape}, second_device->memory_spaces()[0]));
  auto second_buffer = transfer_manager->RetrieveBuffer(0);

  TF_EXPECT_OK(transfer_manager->TransferRawDataToSubBuffer(
      0, host_dma_ptr, 0, size, true, []() {}));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, second_buffer->ToLiteral().Await());
  EXPECT_EQ(literal->element_count(), test_length);
  EXPECT_THAT(literal->data<int32_t>(), ElementsAreArray(data));

  TF_EXPECT_OK(client->DmaUnmap(host_dma_ptr));
}

TEST(StreamExecutorGpuClientTest, RawBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  std::vector<int32_t> data(256);
  absl::c_iota(data, 10);

  Shape shape = ShapeUtil::MakeShape(S32, {256});
  auto buffer =
      client
          ->BufferFromHostBuffer(
              data.data(), shape.element_type(), shape.dimensions(),
              /*byte_strides=*/std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr,
              *client->addressable_devices()[0]->default_memory_space(),
              /*device_layout=*/nullptr)
          .value();
  TF_ASSERT_OK(buffer->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(auto raw_buffer,
                          PjRtRawBuffer::CreateRawAliasOfBuffer(buffer.get()));
  ASSERT_EQ(raw_buffer->memory_space(), buffer->memory_space());
  size_t on_device_size = raw_buffer->GetOnDeviceSizeInBytes();
  ASSERT_EQ(on_device_size, 1024);

  std::vector<int32_t> data2(256);
  absl::c_iota(data2, 47);
  auto* dst1 =
      tsl::port::AlignedMalloc(1024, static_cast<std::align_val_t>(1024));
  auto* dst2 =
      tsl::port::AlignedMalloc(1024, static_cast<std::align_val_t>(1024));
  memcpy(dst1, data2.data(), sizeof(int32_t) * data2.size());
  TF_EXPECT_OK(raw_buffer->CopyRawHostToDevice(dst1, 0, 1024).Await());
  TF_EXPECT_OK(raw_buffer->CopyRawDeviceToHost(dst2, 0, 1024).Await());
  EXPECT_EQ(absl::MakeSpan(reinterpret_cast<int32_t*>(dst2), 256), data2);

  tsl::port::AlignedFree(dst1);
  tsl::port::AlignedFree(dst2);
}

TEST(StreamExecutorGpuClientTest, ComputeSynchronizedAllocatorRace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  PjRtDevice* const device = client->addressable_devices()[0];

  std::unique_ptr<xla::PjRtBuffer> w;
  {
    static constexpr char const* kInitMatrixProgram =
        R"(
HloModule jit_init_matrix, input_output_alias={}, entry_computation_layout={()->f32[4096,4096]{1,0}}

ENTRY main.5 {
  %a = f32[] constant(0)
  ROOT %b = f32[4096,4096]{1,0} broadcast(%a), dimensions={}
}
)";
    TF_ASSERT_OK_AND_ASSIGN(auto executable,
                            CompileExecutable(kInitMatrixProgram, *client));
    std::vector<std::vector<PjRtBuffer*>> input_ptrs = {{}};
    TF_ASSERT_OK_AND_ASSIGN(
        auto results,
        executable->Execute(absl::MakeSpan(input_ptrs), ExecuteOptions()));
    w = std::move(results[0][0]);
  }

  static constexpr char const* kSlowCheckProgram =
      R"(
HloModule jit_slow_verify, input_output_alias={}, entry_computation_layout={(f32[4096,4096]{1,0}, s32[4194304]{0})->(f32[4096,4096]{1,0}, s32[])}

ENTRY main.5 {
  %w.0 = f32[4096,4096]{1,0} parameter(0), sharding={replicated}
  %w.1 = f32[4096,4096]{1,0} dot(%w.0, %w.0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %w.2 = f32[4096,4096]{1,0} dot(%w.1, %w.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %checks.1 = s32[4194304]{0} parameter(1), sharding={replicated}
  %optimization_barrier.4 = (f32[4096,4096]{1,0}, s32[4194304]{0}) tuple(%w.2, %checks.1)
  %optimization_barrier.5 = (f32[4096,4096]{1,0}, s32[4194304]{0}) opt-barrier(%optimization_barrier.4)
  %optimization_barrier.6 = f32[4096,4096]{1,0} get-tuple-element(%optimization_barrier.5), index=0
  %optimization_barrier.7 = s32[4194304]{0} get-tuple-element(%optimization_barrier.5), index=1
  %slice.1 = s32[1]{0} slice(%optimization_barrier.7), slice={[0:1]}
  %squeeze.1 = s32[] reshape(%slice.1)
  ROOT %tuple.1 = (f32[4096,4096]{1,0}, s32[]) tuple(%optimization_barrier.6, %squeeze.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kSlowCheckProgram, *client));

  size_t dma_size = 4 * 1024;
  size_t alignment = 1024;
  auto host_dma_ptr = tsl::port::AlignedMalloc(
      dma_size, static_cast<std::align_val_t>(alignment));
  auto host_dma_ptr_deleter =
      absl::Cleanup([host_dma_ptr, dma_size, alignment] {
        tsl::port::AlignedSizedFree(host_dma_ptr, dma_size,
                                    static_cast<std::align_val_t>(alignment));
      });
  TF_EXPECT_OK(client->DmaMap(host_dma_ptr, dma_size));
  memset(host_dma_ptr, 0, dma_size);
  Shape shape =
      ShapeUtil::MakeShape(S32, {static_cast<int64_t>(dma_size * 1024)});

  void* last_opaque_ptr = nullptr;
  bool clobbered = false;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> res_lst;
  for (int32_t i = 0; i < 10; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                            client->CreateBuffersForAsyncHostToDevice(
                                {shape}, device->memory_spaces()[0]));
    auto buffer = transfer_manager->RetrieveBuffer(0);
    TF_ASSERT_OK_AND_ASSIGN(
        auto raw_buffer,
        xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buffer.get()));

    auto* opaque_ptr = absl::down_cast<CommonPjRtRawBuffer*>(raw_buffer.get())
                           ->OpaqueDeviceMemoryDataPointer();
    if (opaque_ptr == last_opaque_ptr) {
      clobbered = true;
    }
    last_opaque_ptr = opaque_ptr;

    memcpy(host_dma_ptr, &i, sizeof(int32_t));
    absl::Notification done;
    TF_EXPECT_OK(transfer_manager->TransferRawDataToSubBuffer(
        0, host_dma_ptr, 0, dma_size, true, [&done]() { done.Notify(); }));
    done.WaitForNotification();

    std::vector<std::vector<xla::PjRtBuffer*>> input_ptrs = {
        {w.get(), buffer.get()}};
    TF_ASSERT_OK_AND_ASSIGN(
        auto results,
        executable->Execute(absl::MakeSpan(input_ptrs), ExecuteOptions()));
    w = std::move(results[0][0]);
    res_lst.push_back(std::move(results[0][1]));
    if (i - 1 > 0) {
      TF_EXPECT_OK(res_lst[i - 1]->GetReadyFuture().Await());
    }
  }

  std::vector<int32_t> expected;
  std::vector<int32_t> actual;
  for (int32_t i = 0; i < static_cast<int32_t>(res_lst.size()); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto lit, res_lst[i]->ToLiteral().Await());
    expected.push_back(i);
    actual.push_back(lit->data<int32_t>()[0]);
  }

  EXPECT_EQ(expected, actual);

  EXPECT_TRUE(clobbered);

  TF_EXPECT_OK(client->DmaUnmap(host_dma_ptr));
}

TEST(StreamExecutorGpuClientTest, EventCaching) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto* async_work_runner =
      absl::down_cast<PjRtStreamExecutorClient*>(client.get())
          ->async_work_runner();
  const auto& device = client->addressable_devices()[0];
  LocalDeviceState* local_device_state =
      absl::down_cast<const PjRtStreamExecutorDevice*>(device)
          ->local_device_state();
  ASSERT_TRUE(local_device_state != nullptr);
  size_t sync_point0 = local_device_state->GetNextComputeStreamSyncPoint();
  TF_ASSERT_OK_AND_ASSIGN(auto event0,
                          local_device_state->GetEventForComputeStreamSyncPoint(
                              sync_point0, async_work_runner));
  TF_ASSERT_OK_AND_ASSIGN(auto event1,
                          local_device_state->GetEventForComputeStreamSyncPoint(
                              sync_point0, async_work_runner));
  size_t sync_point1 = local_device_state->GetNextComputeStreamSyncPoint();
  TF_ASSERT_OK_AND_ASSIGN(auto event2,
                          local_device_state->GetEventForComputeStreamSyncPoint(
                              sync_point1, async_work_runner));
  // Events are getting cached.
  EXPECT_EQ(&*event0, &*event1);
  // New events are getting assigned.
  EXPECT_NE(&*event0, &*event2);
  tsl::BlockUntilReady(event2);
  // sync_point1 is ready, so it is the most recent event.
  TF_ASSERT_OK_AND_ASSIGN(auto event3,
                          local_device_state->GetEventForComputeStreamSyncPoint(
                              sync_point0, async_work_runner));
  EXPECT_EQ(&*event3, &*event2);
}

TEST(StreamExecutorGpuClientTest, LinkedEventPromise) {
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                          GetStreamExecutorGpuClient(DefaultOptions()));
  auto* client = absl::down_cast<PjRtStreamExecutorClient*>(pjrt_client.get());
  auto* memory_space = client->memory_spaces()[0];
  auto literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      Shape device_shape,
      client->MakeDefaultShapeForMemorySpace(memory_space, literal.shape(),
                                             /*layout=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t on_device_bytes_count,
      client->GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      auto raw_buffer,
      client->AllocateRawBuffer(memory_space, on_device_bytes_count,
                                /*retry_on_oom=*/true,
                                /*allocate_after=*/{}));
  tsl::RCReference<PjRtDeviceEventPromise> promise;
  tsl::RCReference<PjRtDeviceEvent> event;
  TF_ASSERT_OK_AND_ASSIGN(std::tie(promise, event),
                          client->CreateLinkedEventPromise(memory_space, ""));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->DefineBuffer(device_shape, memory_space, raw_buffer,
                                        {std::move(event)}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto definition_event,
      client->LinearizeInto(
          literal, device_shape,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          raw_buffer));
  promise->Set(std::move(definition_event));

  TF_ASSERT_OK_AND_ASSIGN(auto new_literal, buffer->ToLiteral().Await());
  ASSERT_EQ(literal, *new_literal);
}

TEST(StreamExecutorGpuClientTest, FailedCrossHostSendArgsSizeMismatch) {
  // Create the client.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  // Create a buffer to try to send.
  std::vector<int32_t> data(256);
  absl::c_iota(data, 1);

  Shape shape = ShapeUtil::MakeShape(S32, {256});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *client->addressable_devices()[0]->default_memory_space(),
          /*device_layout=*/nullptr));

  // Try to send some data, giving an extra dst_global_device_id.
  EXPECT_THAT(
      client->CrossHostSendBuffers({buffer.get()},
                                   {GlobalDeviceId(1), GlobalDeviceId(2)},
                                   {CrossHostTransferKey(0)}),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq("CrossHostSendBuffers: buffers, "
                           "dst_global_device_ids, and transfer_keys "
                           "must have the same length, but got 1, 2, and 1.")));

  // Try to send some data, giving and extra transfer key.
  EXPECT_THAT(
      client->CrossHostSendBuffers(
          {buffer.get()}, {GlobalDeviceId(1)},
          {CrossHostTransferKey(0), CrossHostTransferKey(1)}),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq("CrossHostSendBuffers: buffers, "
                           "dst_global_device_ids, and transfer_keys "
                           "must have the same length, but got 1, 1, and 2.")));
}

TEST(StreamExecutorGpuClientTest, FailedCrossHostTransferSrcAndDstAddressable) {
  // Create the client.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  // Create a buffer to try to send.
  std::vector<int32_t> data(256);
  absl::c_iota(data, 1);

  Shape shape = ShapeUtil::MakeShape(S32, {256});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *client->addressable_devices()[0]->default_memory_space(),
          /*device_layout=*/nullptr));

  // Try to transfer some data between two addressable devices.
  EXPECT_THAT(
      client->CrossHostSendBuffers({buffer.get()}, {GlobalDeviceId(1)},
                                   {CrossHostTransferKey(0)}),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq(
              "CrossHostSendBuffers: destination device for buffer 0 is "
              "addressable (global device id 1), but cross-host transfers must "
              "be between an addressable and a non-addressable device.")));

  EXPECT_THAT(
      client->CrossHostReceiveBuffers(
          /*device=*/client->addressable_devices()[0],
          /*shapes=*/{shape},
          /*src_global_device_ids=*/{GlobalDeviceId(1)},
          /*transfer_keys=*/{CrossHostTransferKey(0)}),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq(
              "CrossHostReceiveBuffers: source device for buffer 0 is "
              "addressable (global device id 1), but cross-host transfers must "
              "be between an addressable and a non-addressable device.")));
}

TEST(StreamExecutorGpuClientTest, FailedCrossHostReceiveArgsSizeMismatch) {
  // Create the client.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(DefaultOptions()));

  // Create shapes to receive.
  std::vector<Shape> shapes = {ShapeUtil::MakeShape(S32, {256})};

  // Check InvalidArgument status when we don't give enough
  // src_global_device_ids.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
      mismatch_status_or_1 = client->CrossHostReceiveBuffers(
          /*device=*/client->addressable_devices()[0],
          /*shapes=*/shapes,
          /*src_global_device_ids=*/{},
          /*transfer_keys=*/{CrossHostTransferKey(0)});
  EXPECT_THAT(
      mismatch_status_or_1.status(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq(
              "CrossHostReceiveBuffers: shapes, src_global_device_ids, and "
              "transfer_keys must have the same length, but got 1, 0, and "
              "1.")));

  // Check InvalidArgument status when we give too many
  // transfer_keys.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
      mismatch_status_or_2 = client->CrossHostReceiveBuffers(
          /*device=*/client->addressable_devices()[0],
          /*shapes=*/shapes,
          /*src_global_device_ids=*/{GlobalDeviceId(0)},
          /*transfer_keys=*/{CrossHostTransferKey(0), CrossHostTransferKey(1)});
  EXPECT_THAT(
      mismatch_status_or_2.status(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq(
              "CrossHostReceiveBuffers: shapes, src_global_device_ids, and "
              "transfer_keys must have the same length, but got 1, 1, and "
              "2.")));
}

static std::string SuccessfulCrossHostTransferTestName(
    const ::testing::TestParamInfo<int>& info) {
  return absl::StrFormat("num_arrays_%d", info.param);
}

static const char* test_binary_name;

class SuccessfulCrossHostTransferTest : public ::testing::TestWithParam<int> {};

TEST_P(SuccessfulCrossHostTransferTest, SuccessfulCrossHostTransfer) {
  int num_arrays = GetParam();

  tsl::SubProcess sender;
  tsl::SubProcess receiver;

  std::vector<std::string> sender_argv;
  sender_argv.push_back(test_binary_name);
  sender_argv.push_back("successful_cross_host_transfer_test");
  sender_argv.push_back("--test_to_run=SuccessfulCrossHostTransferHelper");
  sender_argv.push_back("--cross_host_test_role=sender");
  sender_argv.push_back(absl::StrFormat("--num_arrays=%d", num_arrays));

  std::vector<std::string> receiver_argv;
  receiver_argv.push_back(test_binary_name);
  receiver_argv.push_back("successful_cross_host_transfer_test");
  receiver_argv.push_back("--test_to_run=SuccessfulCrossHostTransferHelper");
  receiver_argv.push_back("--cross_host_test_role=receiver");
  receiver_argv.push_back(absl::StrFormat("--num_arrays=%d", num_arrays));

  sender.SetProgram(test_binary_name, sender_argv);
  sender.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  sender.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  receiver.SetProgram(test_binary_name, receiver_argv);
  receiver.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  receiver.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  ASSERT_TRUE(sender.Start());
  ASSERT_TRUE(receiver.Start());

  std::string sender_stdout, sender_stderr;
  std::string receiver_stdout, receiver_stderr;

  int sender_status =
      sender.Communicate(nullptr, &sender_stdout, &sender_stderr);
  int receiver_status =
      receiver.Communicate(nullptr, &receiver_stdout, &receiver_stderr);

  EXPECT_EQ(sender_status, 0) << "sender stdout:\n"
                              << sender_stdout << "\nsender stderr:\n"
                              << sender_stderr;
  EXPECT_EQ(receiver_status, 0) << "receiver stdout:\n"
                                << receiver_stdout << "\nreceiver stderr:\n"
                                << receiver_stderr;
}

INSTANTIATE_TEST_SUITE_P(SuccessfulCrossHostTransfer,
                         SuccessfulCrossHostTransferTest,
                         ::testing::ValuesIn({1, 2, 3}),
                         SuccessfulCrossHostTransferTestName);

absl::Status SuccessfulCrossHostTransferTestBody(bool is_sender,
                                                 int num_arrays) {
  std::string log_prefix = is_sender ? "sender" : "receiver";

  // Sender creates a coordination service on so both processes can find each
  // other via the distributed runtime (port chosen arbitrarily).
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (is_sender) {
    LOG(INFO) << log_prefix << ": creating coordination service";
    TF_ASSIGN_OR_RETURN(
        service, xla::GetDistributedRuntimeService(
                     "127.0.0.1:12347",
                     xla::CoordinationServiceImpl::Options{/*num_nodes=*/2}));
    LOG(INFO) << log_prefix << ": created service";
  }

  // Connect to the coordination service.
  int32_t node_id = is_sender ? 0 : 1;
  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12347", distributed_options);

  LOG(INFO) << log_prefix << ": connecting distributed client";
  TF_QCHECK_OK(distributed_client->Connect());
  LOG(INFO) << log_prefix << ": distributed client connected";

  // Create the GPU client.
  GpuClientOptions options = DefaultOptions();
  options.node_id = node_id;
  options.num_nodes = 2;
  options.kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"cross:");
  options.allowed_devices = {node_id};

  LOG(INFO) << log_prefix << ": creating PjRtClient";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(options));
  LOG(INFO) << log_prefix << ": PjRtClient created";

  // Sender logic.
  if (is_sender) {
    LOG(INFO) << log_prefix << ": creating buffers";

    // Create the data to send.
    Shape shape = ShapeUtil::MakeShape(S32, {256});
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    for (int i = 0; i < num_arrays; ++i) {
      std::vector<int32_t> data(256);
      absl::c_iota(data, 1000 * i);

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> buffer,
          client->BufferFromHostBuffer(
              data.data(), shape.element_type(), shape.dimensions(),
              /*byte_strides=*/std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr,
              *client->addressable_devices()[0]->default_memory_space(),
              /*device_layout=*/nullptr));
      TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
      buffers.push_back(std::move(buffer));
    }

    // Send some data.
    LOG(INFO) << log_prefix << ": issuing CrossHostSendBuffers";

    std::vector<PjRtBuffer*> raw_buffers;
    std::vector<GlobalDeviceId> dst_device_ids;
    std::vector<CrossHostTransferKey> transfer_keys;
    for (int i = 0; i < buffers.size(); ++i) {
      raw_buffers.push_back(buffers[i].get());
      dst_device_ids.push_back(GlobalDeviceId(1));
      transfer_keys.push_back(CrossHostTransferKey(i));
    };

    TF_ASSIGN_OR_RETURN(
        std::vector<Future<>> send_futures,
        client->CrossHostSendBuffers(raw_buffers, dst_device_ids,
                                     std::move(transfer_keys)));

    EXPECT_EQ(send_futures.size(), num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      LOG(INFO) << log_prefix << ": waiting for send " << i << " to complete";
      TF_RETURN_IF_ERROR(send_futures[i].Await());
      LOG(INFO) << log_prefix << ": send " << i << " completed";
    }
  } else {
    // Receiver logic.
    std::vector<Shape> shapes;
    std::vector<GlobalDeviceId> src_device_ids;
    std::vector<CrossHostTransferKey> transfer_keys;
    for (int i = 0; i < num_arrays; ++i) {
      shapes.push_back(ShapeUtil::MakeShape(S32, {256}));
      src_device_ids.push_back(GlobalDeviceId(0));
      transfer_keys.push_back(CrossHostTransferKey(i));
    }

    LOG(INFO) << log_prefix << ": calling CrossHostReceiveBuffers";
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<PjRtBuffer>> receive_buffers,
        client->CrossHostReceiveBuffers(client->addressable_devices()[0],
                                        shapes, src_device_ids,
                                        std::move(transfer_keys)));
    LOG(INFO) << log_prefix
              << ": CrossHostReceiveBuffers returned, waiting for ready";

    // Verify we received the expected data.
    EXPECT_EQ(receive_buffers.size(), num_arrays);

    for (int i = 0; i < num_arrays; ++i) {
      std::vector<int32_t> expected_data(256);
      absl::c_iota(expected_data, 1000 * i);
      auto expected_literal = LiteralUtil::CreateR1<int32_t>(expected_data);

      LOG(INFO) << log_prefix << ": waiting for receive " << i
                << " to complete";
      TF_RETURN_IF_ERROR(receive_buffers[i]->GetReadyFuture().Await());
      LOG(INFO) << log_prefix << ": receive " << i << " completed";

      TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> recv_literal,
                          receive_buffers[i]->ToLiteral().Await());

      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *recv_literal));
      LOG(INFO) << log_prefix << ": verification of receive " << i
                << " complete";
    }
  }

  return absl::OkStatus();
}

struct ShardedAutotuningTestInfo {
  int num_active_nodes;
  int num_nodes_using_cache;

  static std::string Name(
      const ::testing::TestParamInfo<ShardedAutotuningTestInfo>& info) {
    return absl::StrFormat("active_%d_cache_%d", info.param.num_active_nodes,
                           info.param.num_nodes_using_cache);
  }
};

class ShardedAutotuningTest
    : public ::testing::TestWithParam<ShardedAutotuningTestInfo> {
 public:
  static constexpr int kNumNodes = 2;
};

TEST_P(ShardedAutotuningTest, ShardedAutotuningWorks) {
  ShardedAutotuningTestInfo param = GetParam();

  std::string cache_dir;
  CHECK(tsl::Env::Default()->LocalTempFilename(&cache_dir));

  if (tsl::kIsOpenSource) {
    // Test relies on VLOG(1) messages. Enable VLOG(1) in OSS.
    tsl::setenv("TF_CPP_VMODULE", "autotuner_pass=10,autotuner=10",
                /*overwrite=*/true);
  }

  // Compile twice to test both empty and non-empty disk cache.
  for (int iteration = 0; iteration < 2; ++iteration) {
    tsl::SubProcess child[kNumNodes];
    for (int node_id = 0; node_id < kNumNodes; ++node_id) {
      std::vector<std::string> argv;
      argv.reserve(7);
      argv.push_back(test_binary_name);
      argv.push_back("sharded_autotuning_test");
      argv.push_back("--test_to_run=ShardedAutotuningWorksHelper");
      argv.push_back(absl::StrFormat("--node_id=%d", node_id));
      argv.push_back(
          absl::StrFormat("--num_active_nodes=%d", param.num_active_nodes));
      argv.push_back(absl::StrFormat("--num_nodes_using_cache=%d",
                                     param.num_nodes_using_cache));
      argv.push_back(absl::StrFormat("--cache_dir=%s", cache_dir));
      // Test relies on VLOG(1) messages. Enable VLOG(1) in Non-OSS.
      if (!tsl::kIsOpenSource) {
        argv.push_back("--vmodule=autotuner_pass=10,autotuner=10");
        argv.push_back("--logtostderr");
      }
      child[node_id].SetProgram(test_binary_name, argv);
      child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
      child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
      ASSERT_TRUE(child[node_id].Start()) << "node " << node_id;
    }
    for (int node_id = 0; node_id < kNumNodes; ++node_id) {
      std::string stdout_str;
      std::string stderr_str;
      int child_status =
          child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);
      if (WIFEXITED(child_status) &&
          WEXITSTATUS(child_status) ==
              static_cast<int>(absl::StatusCode::kFailedPrecondition)) {
        GTEST_SKIP() << "Requires Ampere+ GPU.";
      }
      EXPECT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                                 << stdout_str << "\nstderr:\n"
                                 << stderr_str;
      if (node_id < param.num_active_nodes) {
        int num_fusions_to_autotune = (node_id == 0) ? 1 : 0;
        if (iteration > 0 && node_id < param.num_nodes_using_cache) {
          num_fusions_to_autotune = 0;
        }
        LOG(INFO) << "stderr_str: " << stderr_str;
        if (num_fusions_to_autotune > 0) {
          EXPECT_THAT(
              stderr_str,
              HasSubstr(absl::StrFormat(
                  "Shard %d/%d: finding configs for %d/1 unique instructions",
                  node_id, kNumNodes, num_fusions_to_autotune)));
        } else {
          EXPECT_THAT(stderr_str, HasSubstr("No instructions to autotune."));
        }
      } else {
        stderr_str = absl::StrReplaceAll(
            stderr_str, {{"sharded_autotuning_test", "sharded_test"}});
        EXPECT_THAT(stderr_str, Not(HasSubstr("autotuning")));
      }
    }
  }
}

absl::Status ShardedAutotuningWorksTestBody(const int node_id,
                                            const int num_active_nodes,
                                            const int num_nodes_using_cache,
                                            absl::string_view cache_dir) {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(
        service,
        xla::GetDistributedRuntimeService(
            "[::]:12345", xla::CoordinationServiceImpl::Options{
                              /*num_nodes=*/ShardedAutotuningTest::kNumNodes}));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  GpuClientOptions options = DefaultOptions();
  options.node_id = node_id;
  options.allowed_devices = {node_id};
  options.num_nodes = ShardedAutotuningTest::kNumNodes;
  options.kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"gpu:");
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(options));
  TF_RET_CHECK(client->platform_name() == xla::CudaName() ||
               client->platform_name() == xla::RocmName());
  if (client->platform_name() == xla::CudaName()) {
    TF_ASSIGN_OR_RETURN(
        se::CudaComputeCapability cc,
        se::CudaComputeCapability::FromString(
            std::get<std::string>(client->addressable_devices()
                                      .front()
                                      ->description()
                                      .Attributes()
                                      .at("compute_capability"))));
    if (!cc.IsAtLeastAmpere()) {
      return absl::FailedPreconditionError("Ampere+ GPU required");
    }
  }
  TF_RET_CHECK(client->addressable_device_count() == 1);
  TF_RET_CHECK(client->device_count() == ShardedAutotuningTest::kNumNodes);

  if (node_id >= num_active_nodes) {
    // Inactive nodes connect to the coordination service but don't compile.
    return absl::OkStatus();
  }

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(num_active_nodes);
  DebugOptions& debug_options =
      *compile_options.executable_build_options.mutable_debug_options();
  debug_options.set_xla_gpu_shard_autotuning(true);
  debug_options.set_xla_gpu_cublas_fallback(false);

  if (node_id < num_nodes_using_cache) {
    debug_options.set_xla_gpu_experimental_autotune_cache_mode(
        DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE);
    debug_options.set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);
  }

  const char* kHlo = R"(
    HloModule main
    ENTRY main {
      %p0 = f16[2,32,32] parameter(0)
      ROOT %dot = f16[2,32,32] dot(%p0, %p0), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )";

  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(kHlo, {}));
  xla::XlaComputation computation(hlo_module->ToProto());

  std::unique_ptr<PjRtLoadedExecutable> executable;
  TF_ASSIGN_OR_RETURN(executable,
                      client->CompileAndLoad(computation, compile_options));

  const std::string optimized_hlo =
      executable->GetExecutable()->GetHloModules()->front()->ToString();
  TF_RET_CHECK(absl::StrContains(optimized_hlo, "triton_gemm") ||
               absl::StrContains(optimized_hlo, "__triton_nested_gemm_fusion"))
      << optimized_hlo;

  return absl::OkStatus();
}

INSTANTIATE_TEST_SUITE_P(
    ShardedAutotuningTest, ShardedAutotuningTest,
    ::testing::ValuesIn(std::vector<ShardedAutotuningTestInfo>{
        {2, 0}, {2, 1}, {2, 2}}),
    ShardedAutotuningTestInfo::Name);

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  // Populated by a command line flag. Will be either
  // 'ShardedAutotuningWorksHelper', 'SuccessfulCrossHostTransferHelper', or
  // empty. If empty, all tests are run. Otherwise, the test body for
  // 'ShardedAutotuningWorks' or 'SuccessfulCrossHostTransfer' will be run.
  std::string test_to_run;
  xla::test_binary_name = argv[0];

  // Variables used by ShardedAutotuningWorks.
  int node_id = -1;
  int num_active_nodes = -1;
  int num_nodes_using_cache = -1;
  std::string cache_dir;

  // Variables used by SuccessfulCrossHostTransfer.
  std::string cross_host_test_role;
  int num_arrays = -1;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_to_run", &test_to_run,
                "Which test(s) to execute. Allowed values: '' (runs "
                "all tests), 'ShardedAutotuningWorksHelper' or "
                "'SuccessfulCrossHostTransferHelper'."),

      // Flags for ShardedAutotuningWorks.
      tsl::Flag("node_id", &node_id,
                "Node ID for ShardedAutotuningWorks test."),
      tsl::Flag("num_active_nodes", &num_active_nodes,
                "Test parameter for ShardedAutotuningWorks."),
      tsl::Flag("num_nodes_using_cache", &num_nodes_using_cache,
                "Test parameter for ShardedAutotuningWorks."),
      tsl::Flag("cache_dir", &cache_dir,
                "Test parameter for ShardedAutotuningWorks."),

      // Flags for SuccessfulCrossHostTransfer.
      tsl::Flag("cross_host_test_role", &cross_host_test_role,
                "Test parameter for SuccessfulCrossHostTransfer; either "
                "'sender' or 'receiver'."),
      tsl::Flag("num_arrays", &num_arrays,
                "Test parameter for SuccessfulCrossHostTransfer; number of "
                "arrays to transfer.")};

  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);

  testing::InitGoogleTest(&argc, argv);
  if (test_to_run.empty()) {
    return RUN_ALL_TESTS();
  }

  if (test_to_run == "ShardedAutotuningWorksHelper") {
    absl::Status result = xla::ShardedAutotuningWorksTestBody(
        node_id, num_active_nodes, num_nodes_using_cache, cache_dir);
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
    return result.raw_code();
  }
  if (test_to_run == "SuccessfulCrossHostTransferHelper") {
    absl::Status s;
    if (cross_host_test_role == "sender") {
      s = xla::SuccessfulCrossHostTransferTestBody(/*is_sender=*/true,
                                                   num_arrays);
    } else if (cross_host_test_role == "receiver") {
      s = xla::SuccessfulCrossHostTransferTestBody(/*is_sender=*/false,
                                                   num_arrays);
    } else {
      LOG(ERROR) << "cross_host_test_role must be 'sender' or 'receiver'.";
      return 1;
    }
    if (!s.ok()) {
      LOG(ERROR) << s;
    }
    return s.raw_code();
  }
  LOG(ERROR) << "Unrecognized multiprocess test name " << test_to_run << ".";
  return 1;
}
