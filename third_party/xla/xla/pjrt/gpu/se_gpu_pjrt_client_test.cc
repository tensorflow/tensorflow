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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/log_severity.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "google/protobuf/text_format.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/ffi/ffi.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/pjrt/profiling/test_util/mock_device_time_measurement.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#if GOOGLE_CUDA
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"
#endif  // GOOGLE_CUDA
#include "xla/pjrt/gpu/se_gpu_pjrt_client_test_helper.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
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

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::SizeIs;

TEST(StreamExecutorGpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    const auto it = device->Attributes().find("numa_node");
    ASSERT_NE(it, device->Attributes().end());

    const int64_t* value = std::get_if<int64_t>(&it->second);
    ASSERT_NE(value, nullptr);
    EXPECT_NE(*value, tsl::port::kNUMANoAffinity);
  }
}

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM) || \
    defined(TENSORFLOW_USE_SYCL)
TEST(StreamExecutorGpuClientTest, DonateExternalMem) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
#endif  // defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM) ||
        // defined(TENSORFLOW_USE_SYCL)

TEST(StreamExecutorGpuClientTest, CreateErrorBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest, CreateErrorBufferToken) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

  xla::Shape shape = ShapeUtil::MakeTokenShape();
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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
    EXPECT_THAT(b->GetReadyFuture().Await(),
                StatusIs(input_error.code(), HasSubstr(input_error.message())));
  }
}

// TODO(b/372735047): Fix and reenable.
TEST(StreamExecutorGpuClientTest, DISABLED_DonateWithControlDependency) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  EXPECT_THAT(another_buffer->GetReadyFuture().Await(),
              StatusIs(input_error.code(), HasSubstr(input_error.message())));
}

TEST(StreamExecutorGpuClientTest, SendRecvChunked) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest, ForwardUserDataToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

TEST(StreamExecutorGpuClientTest, PassAttrToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
          custom_call_target="MemsetFromAttr",
          api_version=API_VERSION_TYPED_FFI,
          backend_config={"custom_call_backend_config": {"attributes": "{attr = 3.0 : f32}"}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

TEST(StreamExecutorGpuClientTest, AsyncTransferToken) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  xla::Shape shape = ShapeUtil::MakeTokenShape();
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {shape}, client->addressable_devices()[0]->memory_spaces()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);
  auto ready_future = buffer->GetReadyFuture();
  EXPECT_FALSE(ready_future.IsReady());

  TF_ASSERT_OK(transfer_manager->TransferRawDataToBuffer(0, absl::string_view(),
                                                         []() {}));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  EXPECT_TRUE(literal->shape().IsToken());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncBeforeBufferReady) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient> client,
      GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "This test is skipped in OSS CI environments.";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  std::vector<float> data{1, 3, 5, 7, 11, 13, 17, 19};
  Shape shape = ShapeUtil::MakeShape(F32, {static_cast<int64_t>(data.size())});
  // On ROCm, 10k buffers can hit vm.max_map_count, causing
  // pthread_create to fail.
  const int num_buffers =
      (client->platform_name() == xla::RocmName()) ? 2000 : 10000;
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

TEST(StreamExecutorGpuClientTest, CreateMixOfErrorBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<const GpuTopology> gpu_topology,
                       GpuTopology::FromProto(msg));
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

TEST(StreamExecutorGpuClientTest, GpuDeviceDescriptionTest) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  for (int device_index = 0; device_index < client->device_count();
       device_index++) {
    auto device =
        static_cast<PjRtStreamExecutorDevice*>(client->devices()[device_index]);
    auto coords = device->description().coords();
    // All devices are in the same partition & process.
    EXPECT_THAT(coords, ElementsAre(0, 0, device->local_device_id().value()));
  }
}

TEST(StreamExecutorGpuClientTest, GpuDeviceMemoryLimit) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  for (const auto& device : client->devices()) {
    const auto& attributes = device->description().Attributes();
    const auto it = attributes.find("device_memory_bytes_limit");
    ASSERT_NE(it, attributes.end());
    ASSERT_TRUE(std::holds_alternative<int64_t>(it->second));
    EXPECT_GT(std::get<int64_t>(it->second), 1 << 30);
  }
}

TEST(StreamExecutorGpuClientTest, GpuDeviceSharedMemoryInfo) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

TEST(PjRtCpuClientTest, CopyToMemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  xla::Shape shape = xla::ShapeUtil::MakeShape(S32, {128, 256});
  TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
  for (auto* memory_space : client->memory_spaces()) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer, client->BufferFromHostLiteral(literal, memory_space));
    TF_ASSERT_OK_AND_ASSIGN(buffer,
                            buffer->CopyToMemorySpace(buffer->memory_space()));
    TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteral().Await());
    EXPECT_EQ(*received_literal, literal);
  }
}

TEST(PjRtCpuClientTest, CopyToMemorySpaceWithCustomLayout) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  xla::Shape shape = xla::ShapeUtil::MakeShape(S32, {128, 256});
  TF_ASSERT_OK_AND_ASSIGN(auto literal, xla::MakeFakeLiteral(shape));
  Layout device_layout = LayoutUtil::MakeAscendingLayout(2);
  for (auto* memory_space : client->memory_spaces()) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        client->BufferFromHostLiteral(literal, memory_space, &device_layout));
    TF_ASSERT_OK_AND_ASSIGN(buffer,
                            buffer->CopyToMemorySpace(buffer->memory_space()));
    EXPECT_EQ(buffer->layout()->xla_layout(), device_layout);
    TF_ASSERT_OK_AND_ASSIGN(auto received_literal, buffer->ToLiteral().Await());
    EXPECT_EQ(*received_literal, literal);
  }
}

TEST(StreamExecutorGpuClientTest, ShouldStageHostToDeviceTransfersSetToTrue) {
  GpuClientOptions options_staging = GetTestGpuClientOptions();
  options_staging.should_stage_host_to_device_transfers = true;
  TF_ASSERT_OK_AND_ASSIGN(auto client_staging,
                          GetStreamExecutorGpuClient(options_staging));

  std::vector<float> data(1024, 1.0f);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});

  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated("remove after absl upgrade")]] auto* staging_client =
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
  GpuClientOptions options_no_staging = GetTestGpuClientOptions();
  options_no_staging.should_stage_host_to_device_transfers = false;
  TF_ASSERT_OK_AND_ASSIGN(auto client_no_staging,
                          GetStreamExecutorGpuClient(options_no_staging));

  std::vector<float> data(1024, 1.0f);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});

  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated("remove after absl upgrade")]] auto* no_staging_client =
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

TEST(StreamExecutorGpuClientTest, BufferFromHostBufferPinnedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient> client,
      GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  EXPECT_GT(memory_stats.peak_memory_in_bytes, 0);
  EXPECT_GT(memory_stats.total_allocation_bytes, 0);
  EXPECT_GT(memory_stats.indefinite_allocations, 0);
}

TEST(StreamExecutorGpuClientTest, ExecutePinnedHostOutputTupleTest) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

TEST(StreamExecutorGpuClientTest, ExecutableDeviceParameterMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(
      auto memory_kinds,
      executable->GetExecutable()->GetParameterMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "device");
}

TEST(StreamExecutorGpuClientTest, ExecutablePinnedHostOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetExecutable()->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "pinned_host");
}

TEST(StreamExecutorGpuClientTest, GetCompiledMemoryStatsCountTupleTable) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest,
     ExecutablePinnedHostTupleOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest, ProfileExecution) {
  static constexpr char const* kProgram = R"(
    HloModule profiled
      ENTRY main {
      c0 = f32[] constant(20)
      c1 = f32[] constant(21)
      ROOT res = f32[] add(c0, c1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest, GetDefaultLayout) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

TEST(StreamExecutorGpuClientTest, DmaMapUnmap) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto gpu_client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated("remove after absl upgrade")]] auto client =
      absl::down_cast<PjRtStreamExecutorClient*>(gpu_client.get());
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

TEST(StreamExecutorGpuClientTest, RawBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
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

    auto* opaque_ptr = raw_buffer->OpaqueDeviceMemoryDataPointer();
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated("remove after absl upgrade")]] auto* async_work_runner =
      absl::down_cast<PjRtStreamExecutorClient*>(client.get())
          ->async_work_runner();
  const auto& device = client->addressable_devices()[0];
  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated(
      "remove after absl upgrade")]] LocalDeviceState* local_device_state =
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
  // Wait for the background cleanup callback to prune the completed event0,
  // which is happening asynchronously.
  // If pruning has occurred, querying with nullptr_if_past = true will return a
  // null event.
  bool pruned = false;
  absl::Time deadline = absl::Now() + absl::Seconds(1);
  while (absl::Now() < deadline) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto event,
        local_device_state->GetEventForComputeStreamSyncPoint(
            sync_point0, async_work_runner, /*nullptr_if_past=*/true));
    if (!event) {
      pruned = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(pruned) << "Timeout waiting for completed event0 to be pruned.";

  TF_ASSERT_OK_AND_ASSIGN(auto event3,
                          local_device_state->GetEventForComputeStreamSyncPoint(
                              sync_point0, async_work_runner));
  EXPECT_EQ(&*event3, &*event2);
}

TEST(StreamExecutorGpuClientTest, LinkedEventPromise) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client, GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated("remove after absl upgrade")]] auto* client =
      absl::down_cast<PjRtStreamExecutorClient*>(pjrt_client.get());
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
  PjRtDeviceEventPromiseRef promise;
  PjRtDeviceEventRef event;
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
  promise.Set(std::move(definition_event));

  TF_ASSERT_OK_AND_ASSIGN(auto new_literal, buffer->ToLiteral().Await());
  ASSERT_EQ(literal, *new_literal);
}

TEST(StreamExecutorGpuClientTest, GetAbiVersion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions()));

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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                       CompileExecutable(kAddProgram, *client));

  LOG(ERROR) << typeid(*executable).name();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutableAbiVersion> executable_abi_version,
      executable->GetAbiVersion());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtRuntimeAbiVersion> runtime_abi_version,
      client->RuntimeAbiVersion());
  EXPECT_OK(runtime_abi_version->IsCompatibleWith(*executable_abi_version));
}

TEST(StreamExecutorGpuClientTest,
     TopologyDescriptionHasTargetConfigAndHostTargetMachineOptions) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                       client->GetTopologyDescription());
  EXPECT_THAT(topology->Attributes(), Contains(Pair("target_config", _)));
  EXPECT_THAT(topology->Attributes(),
              Contains(Pair("host_target_machine_options", _)));

  auto se_topology =
      dynamic_cast<const StreamExecutorGpuTopologyDescription*>(topology);
  ASSERT_NE(se_topology, nullptr);
  EXPECT_TRUE(se_topology->gpu_topology().has_gpu_target_config());
  EXPECT_TRUE(
      se_topology->gpu_topology().host_target_machine_options().has_value());
}

// The "address" allocator must give a dedicated synchronous passthrough
// StreamExecutorAddressAllocator at the PJRT level and bypass the BFC allocator
// (MultiDeviceAdapter) entirely.
TEST(StreamExecutorGpuClientTest, AddressAllocatorIsSynchronousPassthrough) {
  GpuClientOptions options;
  options.allocator_config.kind = GpuAllocatorConfig::Kind::kAddress;
  options.allowed_devices = {0};

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto* pjrt_se_client =
      absl::down_cast<PjRtStreamExecutorClient*>(client.get());
  EXPECT_NE(dynamic_cast<se::StreamExecutorAddressAllocator*>(
                pjrt_se_client->allocator()),
            nullptr);
  EXPECT_EQ(dynamic_cast<se::MultiDeviceAdapter*>(pjrt_se_client->allocator()),
            nullptr);
}

#if GOOGLE_CUDA
class VmmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ::testing::Test::SetUp();

    auto platform_or = xla::PlatformUtil::GetPlatform("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available.";
    }
    auto* platform = platform_or.value();

    auto executor_or = platform->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "No CUDA device available.";
    }
    auto* executor = executor_or.value();

    const auto& dev_desc = executor->GetDeviceDescription();
    if (!dev_desc.cuda_compute_capability().IsAtLeastHopper()) {
      GTEST_SKIP() << "This test requires at least a Hopper GPU (SM 9.0).";
    }
  }
};

TEST_F(VmmTest, VmmAllocatorCanBeSet) {
  GpuClientOptions options;
  options.allocator_config.kind = GpuAllocatorConfig::Kind::kVmm;
  options.allowed_devices = {0};

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto* pjrt_se_client =
      absl::down_cast<PjRtStreamExecutorClient*>(client.get());
  EXPECT_NE(dynamic_cast<se::gpu::CudaDeviceAddressVmmAllocator*>(
                pjrt_se_client->allocator()),
            nullptr);
}

TEST_F(VmmTest, VmmAllocatorE2ETest) {
  GpuClientOptions options;
  options.allocator_config.kind = GpuAllocatorConfig::Kind::kVmm;
  options.allowed_devices = {0};

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  static constexpr char kAddProgram[] = R"(
HloModule Add, entry_computation_layout={(f32[], f32[])->f32[]}
ENTRY main (a: f32[], b: f32[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(
      auto* memory_space,
      client->addressable_devices()[0]->default_memory_space());
  Literal literal_a = LiteralUtil::CreateR0<float>(1.0f);
  Literal literal_b = LiteralUtil::CreateR0<float>(2.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto a, client->BufferFromHostLiteral(literal_a, memory_space));
  TF_ASSERT_OK_AND_ASSIGN(
      auto b, client->BufferFromHostLiteral(literal_b, memory_space));

  TF_ASSERT_OK_AND_ASSIGN(
      auto results, executable->Execute({{a.get(), b.get()}}, /*options=*/{}));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                          results[0][0]->ToLiteral().Await());
  EXPECT_EQ(literal->Get<float>({}), 3.0f);
}

GpuClientOptions VmmClientOptions() {
  GpuClientOptions options;
  options.allowed_devices = {0};
  options.allocator_config.kind = GpuAllocatorConfig::Kind::kVmm;
  return options;
}

// Creates CompileOptions enabling command buffer VA remapping and all command
// buffer types. Sets xla_gpu_graph_min_graph_size=1 so even small computations
// are wrapped in command buffers.
CompileOptions CmdBufVaRemappingOptions() {
  CompileOptions opts;
  auto* dbg = opts.executable_build_options.mutable_debug_options();
  dbg->set_xla_gpu_command_buffer_update_mode(DebugOptions::NEVER_UPDATE);
  dbg->set_xla_gpu_graph_min_graph_size(1);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::CUBLASLT);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::CONDITIONAL);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::DYNAMIC_SLICE_FUSION);
  return opts;
}

// Tests that element-wise fusion operations (FUSION command type) produce
// correct results under command buffer VA remapping across multiple runs,
// reusing one VA reservation.
TEST_F(VmmTest, CommandBufferVaRemappingFusionOps) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  static constexpr char kHlo[] = R"(
    HloModule fusion_va_remapping_test
    ENTRY main {
      x = f32[8] parameter(0)
      y = f32[8] parameter(1)
      ROOT add = f32[8] add(x, y)
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kHlo, *client, CmdBufVaRemappingOptions()));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  // 3 runs reuse the same VA reservation with different physical allocations.
  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();
  for (int run = 0; run < 3; ++run) {
    float base = static_cast<float>(run * 10);
    auto x_lit =
        LiteralUtil::CreateR1<float>({base + 1, base + 2, base + 3, base + 4,
                                      base + 5, base + 6, base + 7, base + 8});
    auto y_lit = LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5, 6, 7, 8});

    TF_ASSERT_OK_AND_ASSIGN(auto x_buf,
                            client->BufferFromHostLiteral(x_lit, mem));
    TF_ASSERT_OK_AND_ASSIGN(auto y_buf,
                            client->BufferFromHostLiteral(y_lit, mem));

    auto result = executable->Execute({{x_buf.get(), y_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>({base + 2, base + 4, base + 6, base + 8,
                                      base + 10, base + 12, base + 14,
                                      base + 16}),
        *result_lit))
        << "Mismatch on run " << run;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests that GEMM operations (CUBLAS/CUBLASLT command type) produce correct
// results under command buffer VA remapping.
TEST_F(VmmTest, CommandBufferVaRemappingGemmOps) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  static constexpr char kHlo[] = R"(
    HloModule gemm_va_remapping_test
    ENTRY main {
      lhs = f32[4,4] parameter(0)
      rhs = f32[4,4] parameter(1)
      ROOT dot = f32[4,4] dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  CompileOptions opts = CmdBufVaRemappingOptions();
  // Force CUBLAS routing even for small matrices.
  opts.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_gemm_rewrite_size_threshold(0);

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kHlo, *client, opts));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  // rhs = identity matrix → lhs * identity == lhs.
  auto identity = LiteralUtil::CreateR2<float>(
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});

  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();
  for (int run = 0; run < 3; ++run) {
    float s = static_cast<float>(run + 1);
    // lhs = s * identity.
    auto lhs = LiteralUtil::CreateR2<float>(
        {{s, 0, 0, 0}, {0, s, 0, 0}, {0, 0, s, 0}, {0, 0, 0, s}});

    TF_ASSERT_OK_AND_ASSIGN(auto lhs_buf,
                            client->BufferFromHostLiteral(lhs, mem));
    TF_ASSERT_OK_AND_ASSIGN(auto rhs_buf,
                            client->BufferFromHostLiteral(identity, mem));

    auto result = executable->Execute({{lhs_buf.get(), rhs_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Near(lhs, *result_lit, ErrorSpec{1e-5}))
        << "Mismatch on run " << run;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests that conditional operations (CONDITIONAL command type) produce correct
// results under command buffer VA remapping.
TEST_F(VmmTest, CommandBufferVaRemappingConditional) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  static constexpr char kHlo[] = R"(
    HloModule conditional_va_remapping_test
    true_branch {
      p = f32[] parameter(0)
      c = f32[] constant(10.0)
      ROOT r = f32[] add(p, c)
    }
    false_branch {
      p = f32[] parameter(0)
      c = f32[] constant(20.0)
      ROOT r = f32[] add(p, c)
    }
    ENTRY main {
      cond = pred[] parameter(0)
      val = f32[] parameter(1)
      ROOT result = f32[] conditional(cond, val, val),
        true_computation=true_branch, false_computation=false_branch
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kHlo, *client, CmdBufVaRemappingOptions()));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  // Alternate true/false to exercise repeated VA remapping.
  struct RunConfig {
    bool cond;
    float val;
    float expected;
  };
  std::vector<RunConfig> runs = {
      {true, 5.0f, 15.0f}, {false, 5.0f, 25.0f}, {true, 7.0f, 17.0f}};

  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();
  for (const auto& cfg : runs) {
    auto cond_lit = LiteralUtil::CreateR0<bool>(cfg.cond);
    auto val_lit = LiteralUtil::CreateR0<float>(cfg.val);

    TF_ASSERT_OK_AND_ASSIGN(auto cond_buf,
                            client->BufferFromHostLiteral(cond_lit, mem));
    TF_ASSERT_OK_AND_ASSIGN(auto val_buf,
                            client->BufferFromHostLiteral(val_lit, mem));

    auto result = executable->Execute({{cond_buf.get(), val_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR0<float>(cfg.expected), *result_lit));
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests that while-loop operations (WHILE command type) produce correct results
// under command buffer VA remapping.
TEST_F(VmmTest, CommandBufferVaRemappingWhileLoop) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  // Loop runs 4 iterations, adding 1.0 each time: result = init_val + 4.
  static constexpr char kHlo[] = R"(
    HloModule while_va_remapping_test
    cond {
      state = (s32[], f32[]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }
    body {
      state = (s32[], f32[]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      val = f32[] get-tuple-element(state), index=1
      one_i = s32[] constant(1)
      one_f = f32[] constant(1.0)
      i1 = s32[] add(i, one_i)
      val1 = f32[] add(val, one_f)
      ROOT next = (s32[], f32[]) tuple(i1, val1)
    }
    ENTRY main {
      init_val = f32[] parameter(0)
      init_i = s32[] constant(0)
      init = (s32[], f32[]) tuple(init_i, init_val)
      loop = (s32[], f32[]) while(init), condition=cond, body=body
      ROOT result = f32[] get-tuple-element(loop), index=1
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kHlo, *client, CmdBufVaRemappingOptions()));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();
  for (int run = 0; run < 3; ++run) {
    float init_val = static_cast<float>(run);
    auto init_lit = LiteralUtil::CreateR0<float>(init_val);
    TF_ASSERT_OK_AND_ASSIGN(auto init_buf,
                            client->BufferFromHostLiteral(init_lit, mem));

    auto result = executable->Execute({{init_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR0<float>(init_val + 4.0f), *result_lit))
        << "Mismatch on run " << run;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests that dynamic-slice fusion operations (DYNAMIC_SLICE_FUSION command
// type) produce correct results under command buffer VA remapping.
// Pattern: dynamic-slice → element-wise op → dynamic-update-slice.
TEST_F(VmmTest, CommandBufferVaRemappingDynamicSliceFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  static constexpr char kHlo[] = R"(
    HloModule ds_fusion_va_remapping_test
    ENTRY main {
      src = f32[8] parameter(0)
      offset = s32[] parameter(1)
      slice = f32[4] dynamic-slice(src, offset), dynamic_slice_sizes={4}
      doubled = f32[4] add(slice, slice)
      ROOT result = f32[8] dynamic-update-slice(src, doubled, offset)
    })";

  CompileOptions opts = CmdBufVaRemappingOptions();
  opts.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_dynamic_slice_fusion(true);

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kHlo, *client, opts));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  struct RunConfig {
    int32_t offset;
    std::vector<float> expected;
  };
  // For each run: src={1,2,3,4,5,6,7,8}, slice src[offset:offset+4], double
  // it, write back. Expected differs by offset.
  std::vector<RunConfig> runs = {
      {0, {2, 4, 6, 8, 5, 6, 7, 8}},
      {2, {1, 2, 6, 8, 10, 12, 7, 8}},
      {4, {1, 2, 3, 4, 10, 12, 14, 16}},
  };

  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();
  for (const auto& cfg : runs) {
    auto src_lit = LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5, 6, 7, 8});
    auto offset_lit = LiteralUtil::CreateR0<int32_t>(cfg.offset);

    TF_ASSERT_OK_AND_ASSIGN(auto src_buf,
                            client->BufferFromHostLiteral(src_lit, mem));
    TF_ASSERT_OK_AND_ASSIGN(auto off_buf,
                            client->BufferFromHostLiteral(offset_lit, mem));

    auto result = executable->Execute({{src_buf.get(), off_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>(cfg.expected), *result_lit))
        << "Mismatch at offset " << cfg.offset;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests repeated reuse of the single command-buffer VA range across multiple
// executions. Verifies no memory corruption from remapping new physical
// allocations into the same reserved VA addresses.
TEST_F(VmmTest, CommandBufferVaRemappingSingleRangeReuse) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  // add-constant: expected = input + {1,2,3,4}.
  static constexpr char kHlo[] = R"(
    HloModule single_range_va_remapping_test
    ENTRY main {
      x = f32[4] parameter(0)
      c = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      ROOT add = f32[4] add(x, c)
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kHlo, *client, CmdBufVaRemappingOptions()));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  // Reuse the same VA range across multiple remaps.
  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();
  for (int run = 0; run < 6; ++run) {
    float base = static_cast<float>(run * 10);
    auto x_lit = LiteralUtil::CreateR1<float>({base, base, base, base});
    TF_ASSERT_OK_AND_ASSIGN(auto x_buf,
                            client->BufferFromHostLiteral(x_lit, mem));

    auto result = executable->Execute({{x_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>({base + 1, base + 2, base + 3, base + 4}),
        *result_lit))
        << "Mismatch on run " << run;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests that CAPTURE_CMD_NEVER_UPDATE mode produces correct results across
// multiple runs. The GEMM is routed through cuBLAS (GemmCmd/CublasLtCmd), which
// are traced commands. In CAPTURE_CMD_NEVER_UPDATE mode only traced commands
// populate command_buffer_allocation_indexes_, activating VA remapping so that
// traced commands skip command buffer updates across single-range remaps.
TEST_F(VmmTest, CommandBufferVaRemappingCustomLibraryUpdateFree) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  // Pure GEMM: lhs * rhs. With Triton disabled the dot is lowered to a
  // GemmCmd/CublasLtCmd (TracedCommandBufferCmd subclass), so its allocations
  // populate command_buffer_allocation_indexes_ under
  // CAPTURE_CMD_NEVER_UPDATE.
  static constexpr char kHlo[] = R"(
    HloModule custom_lib_update_free_test
    ENTRY main {
      lhs = f32[4,4] parameter(0)
      rhs = f32[4,4] parameter(1)
      ROOT dot = f32[4,4] dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  CompileOptions opts;
  auto* dbg = opts.executable_build_options.mutable_debug_options();
  dbg->set_xla_gpu_command_buffer_update_mode(
      DebugOptions::CAPTURE_CMD_NEVER_UPDATE);
  dbg->set_xla_gpu_graph_min_graph_size(1);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions::CUBLASLT);
  // Force CUBLAS routing even for small matrices.
  dbg->set_xla_gpu_gemm_rewrite_size_threshold(0);
  // Disable Triton GEMM fusion so the dot is lowered to a cuBLAS GemmCmd
  // (a TracedCommandBufferCmd subclass) rather than a non-traced KernelCmd.
  dbg->set_xla_gpu_enable_triton_gemm(false);

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kHlo, *client, opts));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  // rhs = identity → lhs * identity = lhs.
  auto identity = LiteralUtil::CreateR2<float>(
      {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});

  // Verify VA remapping is active: traced GEMM allocations are in
  // command_buffer_allocation_indexes_, so ExecuteThunksWithVaRemapping fires.
  int old_vlog = absl::SetVLogLevel("gpu_executable", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            ::testing::HasSubstr("VA remapping: Mapped")))
      .Times(::testing::AtLeast(1));
  mock_log.StartCapturingLogs();

  // 3 runs reuse the same VA reservation with different physical allocations.
  for (int run = 0; run < 3; ++run) {
    float s = static_cast<float>(run + 1);
    // lhs = s * identity → s * identity * identity = s * identity.
    auto lhs = LiteralUtil::CreateR2<float>(
        {{s, 0, 0, 0}, {0, s, 0, 0}, {0, 0, s, 0}, {0, 0, 0, s}});

    ASSERT_OK_AND_ASSIGN(auto lhs_buf, client->BufferFromHostLiteral(lhs, mem));
    ASSERT_OK_AND_ASSIGN(auto rhs_buf,
                         client->BufferFromHostLiteral(identity, mem));

    auto result = executable->Execute({{lhs_buf.get(), rhs_buf.get()}}, {});
    ASSERT_OK_AND_ASSIGN(auto result_lit, ExtractSingleResult(result));

    EXPECT_TRUE(LiteralTestUtil::Near(lhs, *result_lit, ErrorSpec{1e-5}))
        << "Mismatch on run " << run;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog);
}

// Tests that two different executables using NEVER_UPDATE can coexist and
// interleave executions without interfering with each other's VA range.
// Each executable maintains its own per-executor VA reservation, so remapping
// in one does not corrupt the other.
TEST_F(VmmTest, CommandBufferVaRemappingTwoExecutables) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(VmmClientOptions()));

  // exec1: element-wise add.
  static constexpr char kHlo1[] = R"(
    HloModule exec1_va_remapping
    ENTRY main {
      x = f32[8] parameter(0)
      y = f32[8] parameter(1)
      ROOT add = f32[8] add(x, y)
    })";

  // exec2: element-wise multiply.
  static constexpr char kHlo2[] = R"(
    HloModule exec2_va_remapping
    ENTRY main {
      a = f32[8] parameter(0)
      b = f32[8] parameter(1)
      ROOT mul = f32[8] multiply(a, b)
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto exec1,
      CompileExecutable(kHlo1, *client, CmdBufVaRemappingOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto exec2,
      CompileExecutable(kHlo2, *client, CmdBufVaRemappingOptions()));

  auto* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto* mem, device->default_memory_space());

  auto ones = LiteralUtil::CreateR1<float>({1, 1, 1, 1, 1, 1, 1, 1});
  auto twos = LiteralUtil::CreateR1<float>({2, 2, 2, 2, 2, 2, 2, 2});

  // Each executable owns one VA range and one command buffer per executor.
  // Repeated executions remap new physical allocations into the same VA range
  // and reuse the same command buffer after warmup.
  int old_vlog_exec = absl::SetVLogLevel("gpu_executable", 3);
  int old_vlog_cbt = absl::SetVLogLevel("command_buffer_thunk", 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);

  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            AllOf(HasSubstr("exec1_va_remapping"),
                                  HasSubstr("VA remapping: module"))))
      .Times(::testing::AtLeast(1));

  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            AllOf(HasSubstr("exec2_va_remapping"),
                                  HasSubstr("VA remapping: module"))))
      .Times(::testing::AtLeast(1));

  // Initialization fires once per executable after warmup.
  EXPECT_CALL(mock_log, Log(absl::LogSeverity::kInfo, ::testing::_,
                            HasSubstr("Initialize command buffer on device")))
      .Times(2);

  mock_log.StartCapturingLogs();

  // Interleaving stresses that the two executables' VA ranges do not alias or
  // corrupt each other.
  for (int run = 0; run < 5; ++run) {
    float base = static_cast<float>(run + 1);
    auto x = LiteralUtil::CreateR1<float>(
        {base, base, base, base, base, base, base, base});

    TF_ASSERT_OK_AND_ASSIGN(auto x_buf, client->BufferFromHostLiteral(x, mem));
    TF_ASSERT_OK_AND_ASSIGN(auto ones_buf,
                            client->BufferFromHostLiteral(ones, mem));
    TF_ASSERT_OK_AND_ASSIGN(auto twos_buf,
                            client->BufferFromHostLiteral(twos, mem));

    // exec1: x + ones = base + 1.
    auto res1 = exec1->Execute({{x_buf.get(), ones_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto lit1, ExtractSingleResult(res1));
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>({base + 1, base + 1, base + 1, base + 1,
                                      base + 1, base + 1, base + 1, base + 1}),
        *lit1))
        << "exec1 mismatch on run " << run;

    // exec2: x * twos = base * 2.
    auto res2 = exec2->Execute({{x_buf.get(), twos_buf.get()}}, {});
    TF_ASSERT_OK_AND_ASSIGN(auto lit2, ExtractSingleResult(res2));
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR1<float>({base * 2, base * 2, base * 2, base * 2,
                                      base * 2, base * 2, base * 2, base * 2}),
        *lit2))
        << "exec2 mismatch on run " << run;
  }
  mock_log.StopCapturingLogs();
  absl::SetVLogLevel("gpu_executable", old_vlog_exec);
  absl::SetVLogLevel("command_buffer_thunk", old_vlog_cbt);
}

#endif  // GOOGLE_CUDA

}  // namespace
}  // namespace xla
