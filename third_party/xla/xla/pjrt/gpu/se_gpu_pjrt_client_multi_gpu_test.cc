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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/ffi.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
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
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#if GOOGLE_CUDA
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#endif  // GOOGLE_CUDA
#include "xla/pjrt/gpu/se_gpu_pjrt_client_test_helper.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::TestWithParam;

TEST(StreamExecutorGpuClientTest, AsyncCopyToDevice) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  ASSERT_GE(client->addressable_devices().size(), 2);

  // d0 is the device we will perform local/remote sends from.
  auto* d0 = client->addressable_devices()[0];
  // d1 is the device we will perform local/remote recvs, where the recv
  // sync flag may be contended.
  auto* d1 = client->addressable_devices()[1];

  auto src_literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  ASSERT_OK_AND_ASSIGN(auto* d0_memory_space, d0->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto* d1_memory_space, d1->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {src_literal.shape()}, d0_memory_space));
  auto src_buffer = transfer_manager->RetrieveBuffer(0);
  // CopyToMemorySpace won't be enqueued until src_buffer is available.
  ASSERT_OK_AND_ASSIGN(auto local_recv_buffer,
                       src_buffer->CopyToMemorySpace(d1_memory_space));

  ASSERT_OK(transfer_manager->TransferLiteralToBuffer(0, src_literal, []() {}));

  auto literal = std::make_shared<Literal>(src_literal.shape());

  auto local_recv_literal = local_recv_buffer->ToLiteral(literal.get());
  EXPECT_OK(local_recv_literal.Await());

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<float>(),
            literal->Relayout(src_literal.shape().layout()).data<float>());
}

TEST(StreamExecutorGpuClientTest, CopyErrorBufferToDevice) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

  auto* src_device = client->addressable_devices()[0];
  auto* dst_device = client->addressable_devices()[1];

  ASSERT_OK_AND_ASSIGN(auto* src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto* dst_memory_space,
                       dst_device->default_memory_space());

  ASSERT_OK_AND_ASSIGN(auto send_buffer, client->CreateErrorBuffer(
                                             Internal("some error"),
                                             ShapeUtil::MakeShape(U32, {3, 2}),
                                             src_memory_space));

  ASSERT_OK_AND_ASSIGN(auto recv_buffer,
                       send_buffer->CopyToMemorySpace(dst_memory_space));

  EXPECT_THAT(
      recv_buffer->ToLiteral().Await(),
      absl_testing::StatusIs(tsl::error::INTERNAL, HasSubstr("some error")));
}

TEST(StreamExecutorGpuClientTest, CopyTokenToDevice) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  ASSERT_GE(client->addressable_devices().size(), 2);

  auto* d0 = client->addressable_devices()[0];
  auto* d1 = client->addressable_devices()[1];

  xla::Literal literal = xla::LiteralUtil::CreateToken();
  ASSERT_OK_AND_ASSIGN(auto* d0_memory_space, d0->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto* d1_memory_space, d1->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto src_buffer,
                       client->BufferFromHostLiteral(literal, d0_memory_space));

  ASSERT_OK_AND_ASSIGN(auto dst_buffer,
                       src_buffer->CopyToMemorySpace(d1_memory_space));

  xla::Literal received_literal = xla::LiteralUtil::CreateToken();
  ASSERT_OK(dst_buffer->ToLiteral(&received_literal).Await());
  EXPECT_TRUE(received_literal.shape().IsToken());
}

TEST(StreamExecutorGpuClientTest, CopyErrorTokenToDevice) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  ASSERT_GE(client->addressable_devices().size(), 2);

  auto* d0 = client->addressable_devices()[0];
  auto* d1 = client->addressable_devices()[1];

  xla::Shape shape = ShapeUtil::MakeTokenShape();
  ASSERT_OK_AND_ASSIGN(auto* d0_memory_space, d0->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto* d1_memory_space, d1->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto src_buffer, client->CreateErrorBuffer(
                                            absl::InternalError("token error"),
                                            shape, d0_memory_space));

  ASSERT_OK_AND_ASSIGN(auto dst_buffer,
                       src_buffer->CopyToMemorySpace(d1_memory_space));

  EXPECT_THAT(dst_buffer->ToLiteral().Await(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("token error")));
}

TEST(StreamExecutorGpuClientTest, CopyDelayedErrorBufferToDevice) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

  auto* src_device = client->addressable_devices()[0];
  auto* dst_device = client->addressable_devices()[1];

  ASSERT_OK_AND_ASSIGN(auto* src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto* dst_memory_space,
                       dst_device->default_memory_space());

  xla::Shape shape = ShapeUtil::MakeShape(U32, {3, 2});

  ASSERT_OK_AND_ASSIGN(auto alias_pair,
                       client->CreateAliasBuffer(shape, src_memory_space));
  auto& send_buffer = alias_pair.first;
  auto& fulfill_cb = alias_pair.second;

  ASSERT_OK_AND_ASSIGN(auto recv_buffer,
                       send_buffer->CopyToMemorySpace(dst_memory_space));

  absl::SleepFor(absl::Seconds(3));

  absl::Status error = fulfill_cb(absl::InternalError("delayed error"));

  EXPECT_THAT(recv_buffer->ToLiteral().Await(), error);
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
      ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
      EXPECT_TRUE(client->platform_name() == xla::CudaName() ||
                  client->platform_name() == xla::RocmName() ||
                  client->platform_name() == xla::OneapiName());
      EXPECT_EQ(client->addressable_device_count(), 2);
      EXPECT_EQ(client->device_count(), 4);
    });
  }
}

TEST(StreamExecutorGpuClientTest, GetAllocatorStatsTest) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  ASSERT_GE(client->addressable_devices().size(), 2);

  for (auto device : client->addressable_devices()) {
    const xla::Literal literal = xla::LiteralUtil::CreateR0<int32_t>(0);
    ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> buffer,
                         client->BufferFromHostLiteral(literal, memory_space));
    ASSERT_OK(buffer->GetReadyFuture().Await());

    auto stats = device->GetAllocatorStats();
    ASSERT_OK(stats.status());
    ASSERT_GT(stats.value().peak_bytes_in_use, 0);
  }
}

TEST(StreamExecutorGpuClientTest, GetTopologyDescriptionWithGlobalDevicesTest) {
  const int num_nodes = 4;
  GpuClientOptions options;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  options.mock_gpu_topology = "2x2x2";

  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  int devices_per_host = client->addressable_device_count();

  ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                       client->GetTopologyDescription());

  std::vector<std::unique_ptr<const PjRtDeviceDescription>>
      device_descriptions = topology->DeviceDescriptions();
  EXPECT_EQ(client->device_count(), device_descriptions.size());

  for (const auto& device_description : device_descriptions) {
    EXPECT_EQ(device_description->process_index(),
              device_description->id() / devices_per_host);
  }
}

TEST(StreamExecutorGpuClientTest, MockNcclClientTest) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  const int num_nodes = 4;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

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

TEST(StreamExecutorGpuClientTest, MockNcclClientWithGpuTopologyTest) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  options.enable_mock_nccl = true;
  options.num_nodes = 8;
  options.mock_gpu_topology = "2x4x2";
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  ASSERT_OK_AND_ASSIGN(const xla::PjRtTopologyDescription* topology,
                       client->GetTopologyDescription());
  // TODO(b/b/482307468) Switch to absl::down_cast after upgrade.
  [[deprecated(
      "remove after absl upgrade")]] const StreamExecutorGpuTopologyDescription&
      gpu_topology =
          tensorflow::down_cast<const StreamExecutorGpuTopologyDescription&>(
              *topology);

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
  GpuClientOptions client_options = GetTestGpuClientOptions(2);
  client_options.enable_mock_nccl = true;
  client_options.num_nodes = 4;
  client_options.mock_gpu_topology = "2x2x2";
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(client_options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  auto context = std::make_unique<mlir::MLIRContext>();
  ASSERT_OK_AND_ASSIGN(
      auto module, xla::ParseMlirModuleString(kMlirDistributedSum, *context));

  xla::CompileOptions options;
  options.executable_build_options.set_num_partitions(8)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});
  ASSERT_OK_AND_ASSIGN(
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
    ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    ASSERT_OK_AND_ASSIGN(
        auto input,
        client->BufferFromHostBuffer(
            data.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));
    input_ptrs.push_back({input.get()});
    inputs.push_back(std::move(input));
  }

  // Test that running the program does not crash/hang.
  ASSERT_OK(executable->Execute(absl::MakeSpan(input_ptrs), ExecuteOptions()));
}

TEST(StreamExecutorGpuClientTest, MockNcclClientWithGpuTopologyMismatchTest) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  options.enable_mock_nccl = true;
  options.num_nodes = 16;
  options.mock_gpu_topology = "2x4";
  EXPECT_FALSE(GetStreamExecutorGpuClient(options).ok());
}

TEST(StreamExecutorGpuClientTest,
     GetCompiledMemoryStatsWithTupleAndNcclUserBuffers) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

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

  ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kProgramWithCollectiveAndTuple, *client, options));

  ASSERT_OK_AND_ASSIGN(auto memory_stats,
                       executable->GetExecutable()->GetCompiledMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 1764786624);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  // Difference in buffer aliasing causes a difference in peak memory usage
  if (client->platform_name() == xla::RocmName()) {
    EXPECT_EQ(memory_stats.peak_memory_in_bytes, 1845006788);
  } else {
    EXPECT_EQ(memory_stats.peak_memory_in_bytes, 2165875144);
  }
}

TEST(StreamExecutorGpuClientTest, GetCompiledMemoryStatsMixedTuple) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

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

  ASSERT_OK_AND_ASSIGN(auto executable, CompileExecutable(kSimpleMixedTupleHlo,
                                                          *client, options));

  ASSERT_OK_AND_ASSIGN(auto memory_stats,
                       executable->GetExecutable()->GetCompiledMemoryStats());

  EXPECT_EQ(memory_stats.output_size_in_bytes, 104);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.peak_memory_in_bytes, 184);
}

TEST(StreamExecutorGpuClientTest, GetCompiledMemoryStatsMixedTupleNotRoot) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

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

  ASSERT_OK_AND_ASSIGN(auto executable, CompileExecutable(kMixedTupleNotRootHlo,
                                                          *client, options));

  ASSERT_OK_AND_ASSIGN(auto memory_stats,
                       executable->GetExecutable()->GetCompiledMemoryStats());

  EXPECT_EQ(memory_stats.output_size_in_bytes, 64);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.peak_memory_in_bytes, 144);
}

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

// Verify the output device memory kind with collective memory space shape
// when NCCL user buffer is enabled.
TEST(StreamExecutorGpuClientTest,
     ExecutableCollectiveMemoryOutputMemoryKindTest) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kCollectiveMemorySpaceOutput, *client, options));
  std::vector<int32_t> data{1, 2, 3, 4};
  // Build the input shape with the correct memory space set.
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1, 4},
                                                    /*minor_to_major=*/{1, 0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);

  auto device = client->addressable_devices()[0];
  ASSERT_OK_AND_ASSIGN(auto* default_memory_space,
                       device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(
      auto input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, default_memory_space,
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");

  ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                       executable->GetExecutable()->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "device");

  ASSERT_OK_AND_ASSIGN(auto result,
                       executable->Execute({{input.get()}}, ExecuteOptions()));
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  ASSERT_OK(result_buffers[0]->GetReadyFuture().Await());
  Shape result_shape = result_buffers[0]->on_device_shape();
  auto memory_space = result_shape.layout().memory_space();
  // Entry results should be copied from S1 to S0 memory space.
  EXPECT_EQ(memory_space, 0);
}

TEST(StreamExecutorGpuClientTest, CollectiveMemorySpaceSmoke) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GpuClientOptions()));
  xla::CompileOptions opts;
  opts.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  ASSERT_OK_AND_ASSIGN(
      auto exe, CompileExecutable(kCollectiveMemorySpaceOutput, *client, opts));

  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1, 4}, {1, 0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);
  auto* device = client->addressable_devices()[0];
  ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(
      auto input, client->BufferFromHostBuffer(
                      data.data(), shape.element_type(), shape.dimensions(),
                      /*byte_strides=*/std::nullopt,
                      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr, memory_space,
                      /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");

  ASSERT_OK_AND_ASSIGN(auto results,
                       exe->Execute({{input.get()}}, ExecuteOptions()));
  auto& buf = results[0][0];
  ASSERT_OK(buf->GetReadyFuture().Await());

  // Entry results should be copied from S1 to S0 memory space.
  EXPECT_EQ(buf->on_device_shape().layout().memory_space(),
            (int)gpu::MemorySpaceColor::kDefault);
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

  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

  auto context = std::make_unique<mlir::MLIRContext>();
  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseMlirModuleString(kMlirH2D, *context));

  ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  ASSERT_OK_AND_ASSIGN(auto modules,
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

  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

  auto context = std::make_unique<mlir::MLIRContext>();
  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseMlirModuleString(kMlirD2H, *context));

  ASSERT_OK_AND_ASSIGN(
      auto executable,
      client->CompileAndLoad(
          MaybeOwningMlirModule(std::move(context), std::move(module)), {}));
  ASSERT_OK_AND_ASSIGN(auto modules,
                       executable->GetExecutable()->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kDefaultMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout.memory_space(), Layout::kHostMemorySpace);
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
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(client_options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  auto context = std::make_unique<mlir::MLIRContext>();
  ASSERT_OK_AND_ASSIGN(
      auto module, xla::ParseMlirModuleString(kMlirDistributedSum, *context));

  xla::CompileOptions options;
  options.executable_build_options.set_num_partitions(8)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});
  ASSERT_OK_AND_ASSIGN(
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
    ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    ASSERT_OK_AND_ASSIGN(
        auto input,
        client->BufferFromHostBuffer(
            data.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));
    input_ptrs.push_back({input.get()});
    inputs.push_back(std::move(input));
  }

  // Test non-zero GPU device time measurement.
  auto measurement0 = CreateDeviceTimeMeasurement();

  // Test that running the program does not crash/hang.
  ASSERT_OK_AND_ASSIGN(auto res, executable->Execute(absl::MakeSpan(input_ptrs),
                                                     ExecuteOptions()));
  ASSERT_OK(res[0][0]->GetReadyFuture().Await());

  // Check measurement after execution completes.
  EXPECT_GT(
      measurement0->GetTotalDuration(DeviceTimeMeasurement::DeviceType::kGpu),
      absl::ZeroDuration());
}

TEST(StreamExecutorGpuClientTest, MultipleDeviceShareDmaMapping) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  ASSERT_GE(client->devices().size(), 2);

  size_t test_length = 512 * 1024;
  std::vector<int32_t> data(test_length);
  for (int32_t i = 0; i < test_length; ++i) {
    data[i] = i;
  }
  Shape shape = ShapeUtil::MakeShape(S32, {static_cast<int64_t>(data.size())});
  PjRtDevice* const first_device = client->addressable_devices()[0];

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> first_buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          first_device->memory_spaces()[0], /*device_layout=*/nullptr));

  ASSERT_OK_AND_ASSIGN(int64_t size, first_buffer->GetOnDeviceSizeInBytes());

  size_t dma_size = 2 * 1024 * 1024;
  size_t alignment = 1024;
  auto host_dma_ptr = tsl::port::AlignedMalloc(
      dma_size, static_cast<std::align_val_t>(alignment));
  auto host_dma_ptr_cleanup =
      absl::Cleanup([host_dma_ptr, dma_size, alignment] {
        tsl::port::AlignedSizedFree(host_dma_ptr, dma_size,
                                    static_cast<std::align_val_t>(alignment));
      });
  EXPECT_OK(client->DmaMap(host_dma_ptr, dma_size));

  auto result = first_buffer->CopyRawToHost(host_dma_ptr, 0, size);
  EXPECT_OK(result.Await());

  PjRtDevice* const second_device = client->addressable_devices()[1];

  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {shape}, second_device->memory_spaces()[0]));
  auto second_buffer = transfer_manager->RetrieveBuffer(0);

  EXPECT_OK(transfer_manager->TransferRawDataToSubBuffer(0, host_dma_ptr, 0,
                                                         size, true, []() {}));
  ASSERT_OK_AND_ASSIGN(auto literal, second_buffer->ToLiteral().Await());
  EXPECT_EQ(literal->element_count(), test_length);
  EXPECT_THAT(literal->data<int32_t>(), ElementsAreArray(data));

  EXPECT_OK(client->DmaUnmap(host_dma_ptr));
}

TEST(StreamExecutorGpuClientTest, FailedCrossHostSendArgsSizeMismatch) {
  // Create the client.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

  // Create a buffer to try to send.
  std::vector<int32_t> data(256);
  absl::c_iota(data, 1);

  Shape shape = ShapeUtil::MakeShape(S32, {256});

  ASSERT_OK_AND_ASSIGN(
      auto* memory_space,
      client->addressable_devices()[0]->default_memory_space());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          /*memory_space=*/memory_space,
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

TEST(StreamExecutorGpuClientTest, FailedCrossHostReceiveArgsSizeMismatch) {
  // Create the client.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

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

TEST(StreamExecutorGpuClientTest,
     FailedCrossHostSendReceiveSrcAndDstAddressable) {
  // Create the client.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));

  // Create a buffer to try to send.
  std::vector<int32_t> data(256);
  absl::c_iota(data, 1);

  Shape shape = ShapeUtil::MakeShape(S32, {256});

  ASSERT_OK_AND_ASSIGN(
      auto* memory_space,
      client->addressable_devices()[0]->default_memory_space());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          /*memory_space=*/memory_space,
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

static std::string SuccessfulCrossHostSendReceiveTestName(
    const ::testing::TestParamInfo<int>& info) {
  return absl::StrFormat("num_arrays_%d", info.param);
}

static const char* test_binary_name;

class SuccessfulCrossHostSendReceiveTest
    : public ::testing::TestWithParam<int> {};

TEST_P(SuccessfulCrossHostSendReceiveTest, SuccessfulCrossHostSendReceive) {
  int num_arrays = GetParam();

  tsl::SubProcess sender;
  tsl::SubProcess receiver;

  std::vector<std::string> sender_argv;
  sender_argv.push_back(test_binary_name);
  sender_argv.push_back("successful_cross_host_send_receive_test");
  sender_argv.push_back("--test_to_run=SuccessfulCrossHostSendReceiveHelper");
  sender_argv.push_back("--cross_host_send_receive_test_role=sender");
  sender_argv.push_back(absl::StrFormat("--num_arrays=%d", num_arrays));

  std::vector<std::string> receiver_argv;
  receiver_argv.push_back(test_binary_name);
  receiver_argv.push_back("successful_cross_host_send_receive_test");
  receiver_argv.push_back("--test_to_run=SuccessfulCrossHostSendReceiveHelper");
  receiver_argv.push_back("--cross_host_send_receive_test_role=receiver");
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

INSTANTIATE_TEST_SUITE_P(SuccessfulCrossHostSendReceive,
                         SuccessfulCrossHostSendReceiveTest,
                         ::testing::ValuesIn({1, 2, 3}),
                         SuccessfulCrossHostSendReceiveTestName);

struct PreparedCrossHostTransferTest {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  std::unique_ptr<PjRtClient> client;
};

absl::StatusOr<PreparedCrossHostTransferTest> PrepareCrossHostTransferTest(
    int rank_id, absl::string_view log_prefix) {
  PreparedCrossHostTransferTest prepared_test;

  // Rank 0 creates a coordination service on so both processes can find each
  // other via the distributed runtime (port chosen arbitrarily).
  if (rank_id == 0) {
    LOG(INFO) << log_prefix << ": creating coordination service";
    ASSIGN_OR_RETURN(
        prepared_test.service,
        xla::GetDistributedRuntimeService(
            "127.0.0.1:12347",
            xla::CoordinationServiceImpl::Options{/*num_nodes=*/2}));
    LOG(INFO) << log_prefix << ": created service";
  }

  // Connect to the coordination service.
  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = rank_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12347", distributed_options);

  LOG(INFO) << log_prefix << ": connecting distributed client";
  CHECK_OK(distributed_client->Connect());
  LOG(INFO) << log_prefix << ": distributed client connected";

  // Create the GPU client.
  GpuClientOptions options = GetTestGpuClientOptions(2);
  options.node_id = rank_id;
  options.num_nodes = 2;
  options.kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"cross:");
  options.allowed_devices = {rank_id};

  LOG(INFO) << log_prefix << ": creating PjRtClient";
  ASSIGN_OR_RETURN(prepared_test.client, GetStreamExecutorGpuClient(options));
  LOG(INFO) << log_prefix << ": PjRtClient created";

  return prepared_test;
}

absl::Status SuccessfulCrossHostSendReceiveTestBody(bool is_sender,
                                                    int num_arrays) {
  std::string log_prefix = is_sender ? "sender" : "receiver";

  ASSIGN_OR_RETURN(PreparedCrossHostTransferTest prepared_test,
                   PrepareCrossHostTransferTest(is_sender ? 0 : 1, log_prefix));

  std::unique_ptr<PjRtClient> client = std::move(prepared_test.client);

  // Sender logic.
  if (is_sender) {
    LOG(INFO) << log_prefix << ": creating buffers";

    // Create the data to send.
    Shape shape = ShapeUtil::MakeShape(S32, {256});
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    for (int i = 0; i < num_arrays; ++i) {
      std::vector<int32_t> data(256);
      absl::c_iota(data, 1000 * i);

      ASSIGN_OR_RETURN(
          auto* memory_space,
          client->addressable_devices()[0]->default_memory_space());
      ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> buffer,
          client->BufferFromHostBuffer(
              data.data(), shape.element_type(), shape.dimensions(),
              /*byte_strides=*/std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr, memory_space,
              /*device_layout=*/nullptr));
      RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
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

    ASSIGN_OR_RETURN(std::vector<Future<>> send_futures,
                     client->CrossHostSendBuffers(raw_buffers, dst_device_ids,
                                                  std::move(transfer_keys)));

    EXPECT_EQ(send_futures.size(), num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      LOG(INFO) << log_prefix << ": waiting for send " << i << " to complete";
      RETURN_IF_ERROR(send_futures[i].Await());
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
    ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> receive_buffers,
                     client->CrossHostReceiveBuffers(
                         client->addressable_devices()[0], shapes,
                         src_device_ids, std::move(transfer_keys)));
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
      RETURN_IF_ERROR(receive_buffers[i]->GetReadyFuture().Await());
      LOG(INFO) << log_prefix << ": receive " << i << " completed";

      ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> recv_literal,
                       receive_buffers[i]->ToLiteral().Await());

      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *recv_literal));
      LOG(INFO) << log_prefix << ": verification of receive " << i
                << " complete";
    }
  }

  return absl::OkStatus();
}

TEST(StreamExecutorGpuClientTest, FailedCrossHostTransferSrcAndDstAddressable) {
  ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  auto* client =
      tensorflow::down_cast<PjRtStreamExecutorClient*>(pjrt_client.get());
  auto* memory_space = client->memory_spaces()[0];
  auto literal = LiteralUtil::CreateR1<float>({41.0f, 42.0f, 43.0f, 44.0f});
  ASSERT_OK_AND_ASSIGN(
      Shape device_shape,
      client->MakeDefaultShapeForMemorySpace(memory_space, literal.shape(),
                                             /*layout=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      int64_t on_device_bytes_count,
      client->GetOnDeviceBytesCount(memory_space, device_shape));
  ASSERT_OK_AND_ASSIGN(auto raw_buffer, client->AllocateRawBuffer(
                                            memory_space, on_device_bytes_count,
                                            /*retry_on_oom=*/true,
                                            /*allocate_after=*/{}));

  EXPECT_THAT(
      client
          ->CrossHostTransferBuffers(
              /*transfer_dependencies=*/{},
              /*transfer_specs=*/{CommonPjRtClient::CrossHostTransferSpec{
                  GlobalDeviceId(0), GlobalDeviceId(1), std::move(raw_buffer)}})
          .status(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::StrEq(
              "CrossHostTransferBuffers: remote device for buffer 0 is "
              "addressable (global device id 1), but cross-host transfers must "
              "be between an addressable and a non-addressable device.")));
}

struct SuccessfulCrossHostTransferTestParam {
  int num_rank_0_to_rank_1;
  int num_rank_1_to_rank_0;
};

static std::string SuccessfulCrossHostTransferTestName(
    const ::testing::TestParamInfo<SuccessfulCrossHostTransferTestParam>&
        info) {
  return absl::StrFormat("num_rank0_to_rank1_%d_num_rank1_to_rank0_%d",
                         info.param.num_rank_0_to_rank_1,
                         info.param.num_rank_1_to_rank_0);
}

class SuccessfulCrossHostTransferTest
    : public ::testing::TestWithParam<SuccessfulCrossHostTransferTestParam> {};

TEST_P(SuccessfulCrossHostTransferTest, SuccessfulCrossHostTransfer) {
  SuccessfulCrossHostTransferTestParam param = GetParam();

  tsl::SubProcess rank_0;
  tsl::SubProcess rank_1;

  std::vector<std::string> rank_0_argv;
  rank_0_argv.push_back(test_binary_name);
  rank_0_argv.push_back("successful_cross_host_transfer_test");
  rank_0_argv.push_back("--test_to_run=SuccessfulCrossHostTransferHelper");
  rank_0_argv.push_back("--cross_host_transfer_test_rank=0");
  rank_0_argv.push_back(
      absl::StrFormat("--num_rank_0_to_rank_1=%d", param.num_rank_0_to_rank_1));
  rank_0_argv.push_back(
      absl::StrFormat("--num_rank_1_to_rank_0=%d", param.num_rank_1_to_rank_0));

  std::vector<std::string> rank_1_argv;
  rank_1_argv.push_back(test_binary_name);
  rank_1_argv.push_back("successful_cross_host_transfer_test");
  rank_1_argv.push_back("--test_to_run=SuccessfulCrossHostTransferHelper");
  rank_1_argv.push_back("--cross_host_transfer_test_rank=1");
  rank_1_argv.push_back(
      absl::StrFormat("--num_rank_0_to_rank_1=%d", param.num_rank_0_to_rank_1));
  rank_1_argv.push_back(
      absl::StrFormat("--num_rank_1_to_rank_0=%d", param.num_rank_1_to_rank_0));

  rank_0.SetProgram(test_binary_name, rank_0_argv);
  rank_0.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  rank_0.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  rank_1.SetProgram(test_binary_name, rank_1_argv);
  rank_1.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  rank_1.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  ASSERT_TRUE(rank_0.Start());
  ASSERT_TRUE(rank_1.Start());

  std::string rank_0_stdout, rank_0_stderr;
  std::string rank_1_stdout, rank_1_stderr;

  int rank_0_status =
      rank_0.Communicate(nullptr, &rank_0_stdout, &rank_0_stderr);
  int rank_1_status =
      rank_1.Communicate(nullptr, &rank_1_stdout, &rank_1_stderr);

  EXPECT_EQ(rank_0_status, 0) << "rank_0 stdout:\n"
                              << rank_0_stdout << "\nrank_0 stderr:\n"
                              << rank_0_stderr;
  EXPECT_EQ(rank_1_status, 0) << "rank_1 stdout:\n"
                              << rank_1_stdout << "\nrank_1 stderr:\n"
                              << rank_1_stderr;
}

INSTANTIATE_TEST_SUITE_P(
    SuccessfulCrossHostTransfer, SuccessfulCrossHostTransferTest,
    ::testing::ValuesIn(std::vector<SuccessfulCrossHostTransferTestParam>{
        {1, 0}, {1, 1}, {2, 1}}),
    SuccessfulCrossHostTransferTestName);

absl::Status SuccessfulCrossHostTransferTestBody(int rank_id,
                                                 int num_rank_0_to_rank_1,
                                                 int num_rank_1_to_rank_0) {
  std::string log_prefix = rank_id == 0 ? "rank_0" : "rank_1";
  const int num_transfers = num_rank_0_to_rank_1 + num_rank_1_to_rank_0;

  ASSIGN_OR_RETURN(PreparedCrossHostTransferTest prepared_test,
                   PrepareCrossHostTransferTest(rank_id, log_prefix));
  std::unique_ptr<PjRtClient> client = std::move(prepared_test.client);

  // Prepare the data sent for each transfer.
  // rank_id 0 sends buffers with data:
  //  [0, ..., 255]
  //  [1000, ..., 1255]
  //  [2000, ..., 2255]
  //  ...
  // rank_id 1 sends buffers with data:
  //  [10_000, ..., 10_255]
  //  [11_000, ..., 11_255]
  //  [12_000, ..., 12_255]
  //  ...
  std::vector<std::vector<int32_t>> transferred_data;
  transferred_data.reserve(num_transfers);
  for (int i = 0; i < num_rank_0_to_rank_1; ++i) {
    std::vector<int32_t> curr_data(256);
    absl::c_iota(curr_data, 1000 * i);
    transferred_data.push_back(std::move(curr_data));
  }
  for (int i = 0; i < num_rank_1_to_rank_0; ++i) {
    std::vector<int32_t> curr_data(256);
    absl::c_iota(curr_data, 10000 + 1000 * i);
    transferred_data.push_back(std::move(curr_data));
  }
  Shape shape = ShapeUtil::MakeShape(S32, {256});
  ASSIGN_OR_RETURN(PjRtMemorySpace * default_memory_space,
                   client->addressable_devices()[0]->default_memory_space());

  // Initial values that will be populated in receive buffers (all zeros).
  std::vector<int32_t> initial_zero_values(256, 0);

  // The send / receive PjRtBuffers this rank allocates.
  std::vector<std::unique_ptr<PjRtBuffer>> owned_buffers;
  owned_buffers.reserve(num_transfers);

  // Usage event promises that set the usage events on owned_buffers
  // corresponding to the data transfers.
  std::vector<PjRtDeviceEventPromiseRef> usage_event_promises;
  usage_event_promises.reserve(num_transfers);

  // Passed as input to CrossHostTransferBuffers; contains raw buffers wrapped
  // by owned_buffers.
  std::vector<CommonPjRtClient::CrossHostTransferSpec> transfer_specs;
  transfer_specs.reserve(num_transfers);

  // Holds definition events of owned_buffers.
  PjRtDeviceEventRefVector transfer_dependencies;

  LOG(INFO) << log_prefix << ": preparing transfers.";
  for (int i = 0; i < num_transfers; ++i) {
    int src_global_device_id = i < num_rank_0_to_rank_1 ? 0 : 1;
    int dst_global_device_id = i < num_rank_0_to_rank_1 ? 1 : 0;
    bool is_sender = rank_id == src_global_device_id;

    // Initialize a send / receive buffer.
    ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> buffer,
        client->BufferFromHostBuffer(
            /*data=*/is_sender ? transferred_data[i].data()
                               : initial_zero_values.data(),
            shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
            default_memory_space, /*device_layout=*/nullptr));

    // Create a usage event for the transfer of this buffer.
    PjRtDeviceEventPromiseRef usage_event_promise;
    PjRtDeviceEventRef usage_event;
    ASSIGN_OR_RETURN(
        std::tie(usage_event_promise, usage_event),
        tensorflow::down_cast<CommonPjRtClient*>(client.get())
            ->CreateLinkedEventPromise(default_memory_space,
                                       absl::StrFormat("buffer %i", i)));
    usage_event_promises.push_back(std::move(usage_event_promise));

    // Get a raw buffer.
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
    RETURN_IF_ERROR(
        tensorflow::down_cast<CommonPjRtBufferImpl*>(buffer.get())
            ->AcquireScopedRawBuffer(
                [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
                    PjRtDeviceEventRefVector buf_definition_events) mutable
                    -> absl::StatusOr<PjRtDeviceEventRef> {
                  raw_buffer = std::move(buf_raw_buffer);
                  ConsumeEvents(
                      std::move(buf_definition_events),
                      [&](PjRtDeviceEventRef&& ev) {
                        transfer_dependencies.push_back(std::move(ev));
                      });
                  return PjRtDeviceEventRef(usage_event);
                },
                "SuccessfulCrossHostTransferTestBody"));

    // Form the transfer spec.
    transfer_specs.push_back(CommonPjRtClient::CrossHostTransferSpec{
        GlobalDeviceId(src_global_device_id),
        GlobalDeviceId(dst_global_device_id), std::move(raw_buffer)});

    owned_buffers.push_back(std::move(buffer));

    LOG(INFO) << log_prefix << ": finished preparing transfer " << i;
  }

  // Perform transfers.
  LOG(INFO) << log_prefix << ": enqueuing transfers";
  ASSIGN_OR_RETURN(
      PjRtDeviceEventRefVector usage_events,
      tensorflow::down_cast<CommonPjRtClient*>(client.get())
          ->CrossHostTransferBuffers(std::move(transfer_dependencies),
                                     std::move(transfer_specs)));
  EXPECT_EQ(usage_events.size(), num_transfers);

  // Populate usage events.
  LOG(INFO) << log_prefix << ": setting usage events";
  for (int i = 0; i < usage_events.size(); ++i) {
    usage_event_promises[i].Set(usage_events[i].CopyRef());
  }

  // Wait until the transfers are complete.
  LOG(INFO) << log_prefix << ": waiting for transfers to complete";
  for (int i = 0; i < usage_events.size(); ++i) {
    BlockUntilReady(usage_events[i].down_cast<BufferSequencingEvent>());
  }

  // Verify we received the correct data, and that the data we sent is
  // uncorrupted.
  for (int i = 0; i < num_transfers; ++i) {
    ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> buffer_literal,
                     owned_buffers[i]->ToLiteral().Await());
    auto expected_literal = LiteralUtil::CreateR1<int32_t>(transferred_data[i]);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *buffer_literal));
    LOG(INFO) << log_prefix << ": finished verification of transfer " << i;
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
    tsl::setenv("TF_CPP_VMODULE", "autotuner_pass=10,config_assigner=10",
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
        argv.push_back("--vmodule=autotuner_pass=10,config_assigner=10");
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
          EXPECT_THAT(stderr_str, HasSubstr("Found cached config for HLO"));
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
    ASSIGN_OR_RETURN(
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
  CHECK_OK(distributed_client->Connect());
  GpuClientOptions options = GetTestGpuClientOptions(2);
  options.node_id = node_id;
  options.allowed_devices = {node_id};
  options.num_nodes = ShardedAutotuningTest::kNumNodes;
  options.kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"gpu:");
  ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                   GetStreamExecutorGpuClient(options));
  TF_RET_CHECK(client->platform_name() == xla::CudaName() ||
               client->platform_name() == xla::RocmName() ||
               client->platform_name() == xla::OneapiName());
  if (client->platform_name() == xla::CudaName()) {
#if GOOGLE_CUDA
    ASSIGN_OR_RETURN(se::CudaComputeCapability cc,
                     se::CudaComputeCapability::FromString(
                         std::get<std::string>(client->addressable_devices()
                                                   .front()
                                                   ->description()
                                                   .Attributes()
                                                   .at("compute_capability"))));
    if (!cc.IsAtLeastAmpere()) {
      return absl::FailedPreconditionError("Ampere+ GPU required");
    }
#endif  // GOOGLE_CUDA
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

  ASSIGN_OR_RETURN(auto hlo_module, ParseAndReturnUnverifiedModule(kHlo, {}));
  xla::XlaComputation computation(hlo_module->ToProto());

  std::unique_ptr<PjRtLoadedExecutable> executable;
  ASSIGN_OR_RETURN(executable,
                   client->CompileAndLoad(computation, compile_options));

  ASSIGN_OR_RETURN(auto hlo_modules,
                   executable->GetExecutable()->GetHloModules());
  const std::string optimized_hlo = hlo_modules.front()->ToString();
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
  // 'ShardedAutotuningWorksHelper', 'SuccessfulCrossHostSendReceiveHelper',
  // 'SuccessfulCrossHostTransferHelper', or empty. If empty, all tests are run.
  // Otherwise, the test body for the selected helper will be run.
  std::string test_to_run;
  xla::test_binary_name = argv[0];

  // Variables used by ShardedAutotuningWorks.
  int node_id = -1;
  int num_active_nodes = -1;
  int num_nodes_using_cache = -1;
  std::string cache_dir;

  // Variables used by cross host transfer tests.
  int num_arrays = -1;
  // Used by SuccessfulCrossHostSendReceiveTest.
  std::string cross_host_send_receive_test_role;
  // Used by SuccessfulCrossHostTransferTest.
  int cross_host_transfer_test_rank = -1;
  int num_rank_0_to_rank_1 = -1;
  int num_rank_1_to_rank_0 = -1;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_to_run", &test_to_run,
                "Which test(s) to execute. Allowed values: '' (runs "
                "all tests), 'ShardedAutotuningWorksHelper', "
                "'SuccessfulCrossHostSendReceiveHelper', or "
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

      // Flags for cross host transfer tests.
      tsl::Flag("cross_host_send_receive_test_role",
                &cross_host_send_receive_test_role,
                "Test parameter for SuccessfulCrossHostSendReceive; either "
                "'sender' or 'receiver'."),
      tsl::Flag("num_arrays", &num_arrays,
                "Test parameter for SuccessfulCrossHostSendReceive; number of "
                "arrays to transfer."),
      tsl::Flag("num_rank_0_to_rank_1", &num_rank_0_to_rank_1,
                "Test parameter for SuccessfulCrossHostTransfer; number of "
                "arrays sent from rank 0 to rank 1."),
      tsl::Flag("num_rank_1_to_rank_0", &num_rank_1_to_rank_0,
                "Test parameter for SuccessfulCrossHostTransfer; number of "
                "arrays sent from rank 1 to rank 0."),
      tsl::Flag(
          "cross_host_transfer_test_rank", &cross_host_transfer_test_rank,
          "Test parameter for SuccessfulCrossHostTransfer; either 0 or 1.")};

  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);

  testing::InitGoogleTest(&argc, argv);
  if (test_to_run.empty()) {
    return RUN_ALL_TESTS();
  }

  absl::Status result = absl::OkStatus();

  if (test_to_run == "ShardedAutotuningWorksHelper") {
    result = xla::ShardedAutotuningWorksTestBody(
        node_id, num_active_nodes, num_nodes_using_cache, cache_dir);
  } else if (test_to_run == "SuccessfulCrossHostSendReceiveHelper") {
    if (cross_host_send_receive_test_role == "sender") {
      result = xla::SuccessfulCrossHostSendReceiveTestBody(
          /*is_sender=*/true, num_arrays);
    } else if (cross_host_send_receive_test_role == "receiver") {
      result = xla::SuccessfulCrossHostSendReceiveTestBody(
          /*is_sender=*/false, num_arrays);
    } else {
      result = absl::InvalidArgumentError(
          "cross_host_send_receive_test_role must be 'sender' or "
          "'receiver'.");
    }
  } else if (test_to_run == "SuccessfulCrossHostTransferHelper") {
    if (cross_host_transfer_test_rank != 0 &&
        cross_host_transfer_test_rank != 1) {
      result = absl::InvalidArgumentError(
          "cross_host_transfer_test_rank must be 0 or 1.");
    } else if (num_rank_0_to_rank_1 < 0 || num_rank_1_to_rank_0 < 0) {
      result = absl::InvalidArgumentError(
          "num_rank_0_to_rank_1 and num_rank_1_to_rank_0 must be set.");
    } else {
      result = xla::SuccessfulCrossHostTransferTestBody(
          cross_host_transfer_test_rank, num_rank_0_to_rank_1,
          num_rank_1_to_rank_0);
    }
  } else {
    result = absl::InvalidArgumentError(absl::StrFormat(
        "Unrecognized multiprocess test name %s.", test_to_run));
  }

  if (!result.ok()) {
    LOG(ERROR) << result;
  }
  return result.raw_code();
}
