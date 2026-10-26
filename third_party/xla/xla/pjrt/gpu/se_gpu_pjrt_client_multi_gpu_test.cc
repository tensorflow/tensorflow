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
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
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
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/pjrt/se/pjrt_stream_executor_client.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client_test_helper.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"
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

TEST(StreamExecutorGpuClientTest,
     PropagateAsyncHostToDeviceDelayedErrorCollective) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  ASSERT_GE(client->addressable_devices().size(), 2);

  PjRtDevice* d0 = client->addressable_devices()[0];
  PjRtDevice* d1 = client->addressable_devices()[1];
  ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * d0_memory_space,
                       d0->default_memory_space());
  ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * d1_memory_space,
                       d1->default_memory_space());

  static constexpr absl::string_view kAllReduceProgram = R"(
HloModule AllReduce

sum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  p1 = f32[4]{0} parameter(1)
  sum_inputs = f32[4]{0} add(p0, p1)
  ROOT ar = f32[4]{0} all-reduce(sum_inputs), replica_groups={{0,1}}, to_apply=sum
}
)";

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(2);
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_executable_terminate_timeout_seconds(5);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      CompileExecutable(kAllReduceProgram, *client, compile_options));

  Shape shape = ShapeUtil::MakeShape(F32, {4});
  ASSERT_OK_AND_ASSIGN(auto txm_d0_0, client->CreateBuffersForAsyncHostToDevice(
                                          {shape}, d0_memory_space));
  ASSERT_OK_AND_ASSIGN(auto txm_d0_1, client->CreateBuffersForAsyncHostToDevice(
                                          {shape}, d0_memory_space));
  ASSERT_OK_AND_ASSIGN(auto txm_d1_0, client->CreateBuffersForAsyncHostToDevice(
                                          {shape}, d1_memory_space));

  std::unique_ptr<PjRtBuffer> p0_d0 = txm_d0_0->RetrieveBuffer(0);
  std::unique_ptr<PjRtBuffer> p1_d0 = txm_d0_1->RetrieveBuffer(0);
  std::unique_ptr<PjRtBuffer> p0_d1 = txm_d1_0->RetrieveBuffer(0);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> p1_d1,
                       p1_d0->CopyToMemorySpace(d1_memory_space));

  absl::Status input_error =
      absl::UnavailableError("ReadHostBuffer connection timeout");
  std::unique_ptr<tsl::Thread> error_thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "set_buffer_error", [&]() {
        // Wait for both devices' launch_on_device() callbacks to block in
        // p0's BufferSequencingEvent::WaitForEventOnStream().
        absl::SleepFor(absl::Milliseconds(100));
        // Poison p0 on d0 first, then p1 (which also poisons p1_d1 via
        // CopyToMemorySpace), and then p0 on d1. If IsPredeterminedError() is
        // checked before WaitForEventOnStream(), d0 misses both errors and
        // launches the collective while d1 sees p1_d1's error and skips
        // RunAsync(), deadlocking the collective.
        txm_d0_0->SetBufferError(0, input_error);
        absl::SleepFor(absl::Milliseconds(50));
        txm_d0_1->SetBufferError(0, input_error);
        absl::SleepFor(absl::Milliseconds(50));
        txm_d1_0->SetBufferError(0, input_error);
      }));

  std::optional<std::vector<Future<>>> returned_futures =
      std::vector<Future<>>();
  ASSERT_OK_AND_ASSIGN(auto results,
                       executable->Execute({{p0_d0.get(), p1_d0.get()},
                                            {p0_d1.get(), p1_d1.get()}},
                                           ExecuteOptions(), returned_futures));

  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(returned_futures->size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT((*returned_futures)[i].Await(),
                StatusIs(input_error.code(), HasSubstr(input_error.message())));
    EXPECT_THAT(results[i][0]->GetReadyFuture().Await(),
                StatusIs(input_error.code(), HasSubstr(input_error.message())));
  }
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

TEST(StreamExecutorGpuClientTest,
     DistributedInitWithAbortCollectivesOnFailure) {
  const int num_nodes = 2;
  const char* kServiceAddress = "127.0.0.1:12351";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::DistributedRuntimeService> service,
      xla::GetDistributedRuntimeService(
          kServiceAddress, xla::CoordinationServiceImpl::Options{num_nodes}));

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(
        tsl::Env::Default(), "DistributedInitAbortCollectives", num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([i, num_nodes, kServiceAddress, &statuses]() {
        DistributedRuntimeClient::Options distributed_options;
        distributed_options.node_id = i;
        distributed_options.init_timeout = absl::Seconds(120);
        auto distributed_client =
            GetDistributedRuntimeClient(kServiceAddress, distributed_options);
        statuses[i] = distributed_client->Connect();
        if (!statuses[i].ok()) {
          return;
        }

        GpuClientOptions options = GetTestGpuClientOptions(2);
        options.node_id = i;
        options.num_nodes = num_nodes;
        options.enable_mock_nccl = true;
        options.abort_collectives_on_failure = true;
        options.abort_collectives_timeout = absl::Seconds(30);
        options.distributed_client = distributed_client;
        options.kv_store =
            GetDistributedKeyValueStore(distributed_client, "abort:");

        absl::StatusOr<std::unique_ptr<PjRtClient>> client_status =
            GetStreamExecutorGpuClient(options);
        if (!client_status.ok()) {
          statuses[i] = client_status.status();
          return;
        }
        std::unique_ptr<PjRtClient>& client = *client_status;
        const gpu::GpuExecutableRunOptions* run_options =
            absl::down_cast<PjRtStreamExecutorRawClient*>(
                absl::down_cast<CommonPjRtClient*>(client.get())->raw_client())
                ->gpu_run_options();
        if (run_options == nullptr ||
            run_options->execution_timeout_handlers().size() != 2) {
          statuses[i] = absl::InternalError(
              "execution timeout handler not configured when "
              "abort_collectives_timeout is set");
          return;
        }

        EXPECT_TRUE(client->platform_name() == xla::CudaName() ||
                    client->platform_name() == xla::RocmName() ||
                    client->platform_name() == xla::OneapiName());
        EXPECT_EQ(client->addressable_device_count(), 2);
        EXPECT_EQ(client->device_count(), 4);
        statuses[i] = absl::OkStatus();
      });
    }
  }  // Join all worker threads before reading statuses.

  for (const absl::Status& status : statuses) {
    EXPECT_OK(status);
  }
}

TEST(StreamExecutorGpuClientTest, AbortCollectivesTimeout) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  options.abort_collectives_timeout = absl::Seconds(30);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  const gpu::GpuExecutableRunOptions* run_options =
      absl::down_cast<PjRtStreamExecutorRawClient*>(
          absl::down_cast<CommonPjRtClient*>(client.get())->raw_client())
          ->gpu_run_options();
  ASSERT_NE(run_options, nullptr);

  std::vector<gpu::ExecutionTimeoutHandler> handlers =
      run_options->execution_timeout_handlers();
  ASSERT_EQ(handlers.size(), 2);
  EXPECT_EQ(handlers[0].scope, gpu::ExecutionTimeoutHandler::Scope::kHost);
  EXPECT_EQ(handlers[1].scope, gpu::ExecutionTimeoutHandler::Scope::kDevice);

  for (gpu::ExecutionTimeoutHandler& handler : handlers) {
    EXPECT_EQ(handler.timeout, absl::Seconds(30));
    std::move(handler.callback)("test execution timeout", absl::Seconds(30));
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
          absl::down_cast<const StreamExecutorGpuTopologyDescription&>(
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

TEST(StreamExecutorGpuClientTest, FailedCrossHostTransferSrcAndDstAddressable) {
  ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions(2)));
  auto* client = absl::down_cast<CommonPjRtClientImpl*>(pjrt_client.get());
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

TEST(StreamExecutorGpuClientTest, DeviceErrorCallbackPropagatesToPeerNodes) {
  constexpr int kNumNodes = 2;
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  ASSERT_OK(kv_store->Set("gpu_fatal_error", "stale_error_from_previous_run"));

  absl::StatusOr<std::unique_ptr<PjRtClient>> client_0_or;
  absl::StatusOr<std::unique_ptr<PjRtClient>> client_1_or;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "create_clients",
                                 kNumNodes);
    pool.Schedule([&]() {
      GpuClientOptions options = GetTestGpuClientOptions(2);
      options.node_id = 0;
      options.num_nodes = kNumNodes;
      options.allowed_devices = {0};
      options.kv_store = kv_store;
      client_0_or = GetStreamExecutorGpuClient(options);
    });
    pool.Schedule([&]() {
      GpuClientOptions options = GetTestGpuClientOptions(2);
      options.node_id = 1;
      options.num_nodes = kNumNodes;
      options.allowed_devices = {1};
      options.kv_store = kv_store;
      client_1_or = GetStreamExecutorGpuClient(options);
    });
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client_0,
                       std::move(client_0_or));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client_1,
                       std::move(client_1_or));
  EXPECT_THAT(kv_store->TryGet("gpu_fatal_error"),
              StatusIs(absl::StatusCode::kNotFound));

  auto* se_client_0 = absl::down_cast<CommonPjRtClientImpl*>(client_0.get());
  auto* raw_client_0 =
      absl::down_cast<PjRtStreamExecutorRawClient*>(se_client_0->raw_client());
  ASSERT_OK_AND_ASSIGN(LocalDeviceState * device_state_0,
                       raw_client_0->GetLocalDeviceState(LocalDeviceId(0)));
  ASSERT_TRUE(device_state_0->device_error_callback());

  device_state_0->device_error_callback()(
      absl::InternalError("CUDA_ERROR_ILLEGAL_ADDRESS"));

  ASSERT_OK_AND_ASSIGN(std::string kv_error,
                       kv_store->Get("gpu_fatal_error", absl::Seconds(5)));
  EXPECT_THAT(kv_error, HasSubstr("Fatal GPU device error on node 0 device 0"));
  EXPECT_THAT(kv_error, HasSubstr("CUDA_ERROR_ILLEGAL_ADDRESS"));
}

}  // namespace
}  // namespace xla
