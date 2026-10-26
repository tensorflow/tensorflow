/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/in_memory_store.h"
#include "xla/backends/gpu/tests/multi_process_gpu_test_base.h"
#include "xla/client/executable_build_options.h"
#include "xla/debug_options_flags.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/pjrt/se/pjrt_stream_executor_client.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/autotuning/autotuner_cache.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

std::string SuccessfulCrossHostSendReceiveTestName(
    const ::testing::TestParamInfo<int>& info) {
  return absl::StrFormat("num_arrays_%d", info.param);
}

using SuccessfulCrossHostSendReceiveTest = MultiProcessGpuTestWithParam<int>;

TEST_P(SuccessfulCrossHostSendReceiveTest, SuccessfulCrossHostSendReceive) {
  const int num_arrays = GetParam();
  const bool is_sender = (rank() == 0);
  const std::string log_prefix = is_sender ? "sender" : "receiver";

  if (is_sender) {
    LOG(INFO) << log_prefix << ": creating buffers";

    Shape shape = ShapeUtil::MakeShape(S32, {256});
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    for (int i = 0; i < num_arrays; ++i) {
      std::vector<int32_t> data(256);
      absl::c_iota(data, 1000 * i);

      ASSERT_OK_AND_ASSIGN(
          PjRtMemorySpace * memory_space,
          client()->addressable_devices()[0]->default_memory_space());
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<PjRtBuffer> buffer,
          client()->BufferFromHostBuffer(
              data.data(), shape.element_type(), shape.dimensions(),
              /*byte_strides=*/std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr, memory_space,
              /*device_layout=*/nullptr));
      ASSERT_OK(buffer->GetReadyFuture().Await());
      buffers.push_back(std::move(buffer));
    }

    LOG(INFO) << log_prefix << ": issuing CrossHostSendBuffers";

    std::vector<PjRtBuffer*> raw_buffers;
    std::vector<GlobalDeviceId> dst_device_ids;
    std::vector<CrossHostTransferKey> transfer_keys;
    for (int i = 0; i < buffers.size(); ++i) {
      raw_buffers.push_back(buffers[i].get());
      dst_device_ids.push_back(GlobalDeviceId(1));
      transfer_keys.push_back(CrossHostTransferKey(i));
    }

    ASSERT_OK_AND_ASSIGN(
        std::vector<Future<>> send_futures,
        client()->CrossHostSendBuffers(raw_buffers, dst_device_ids,
                                       std::move(transfer_keys)));

    ASSERT_EQ(send_futures.size(), num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      LOG(INFO) << log_prefix << ": waiting for send " << i << " to complete";
      ASSERT_OK(send_futures[i].Await());
      LOG(INFO) << log_prefix << ": send " << i << " completed";
    }
  } else {
    std::vector<Shape> shapes;
    std::vector<GlobalDeviceId> src_device_ids;
    std::vector<CrossHostTransferKey> transfer_keys;
    for (int i = 0; i < num_arrays; ++i) {
      shapes.push_back(ShapeUtil::MakeShape(S32, {256}));
      src_device_ids.push_back(GlobalDeviceId(0));
      transfer_keys.push_back(CrossHostTransferKey(i));
    }

    LOG(INFO) << log_prefix << ": calling CrossHostReceiveBuffers";
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<PjRtBuffer>> receive_buffers,
        client()->CrossHostReceiveBuffers(client()->addressable_devices()[0],
                                          shapes, src_device_ids,
                                          std::move(transfer_keys)));
    LOG(INFO) << log_prefix
              << ": CrossHostReceiveBuffers returned, waiting for ready";

    ASSERT_EQ(receive_buffers.size(), num_arrays);

    for (int i = 0; i < num_arrays; ++i) {
      std::vector<int32_t> expected_data(256);
      absl::c_iota(expected_data, 1000 * i);
      Literal expected_literal = LiteralUtil::CreateR1<int32_t>(expected_data);

      LOG(INFO) << log_prefix << ": waiting for receive " << i
                << " to complete";
      ASSERT_OK(receive_buffers[i]->GetReadyFuture().Await());
      LOG(INFO) << log_prefix << ": receive " << i << " completed";

      ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> recv_literal,
                           receive_buffers[i]->ToLiteral().Await());

      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *recv_literal));
      LOG(INFO) << log_prefix << ": verification of receive " << i
                << " complete";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SuccessfulCrossHostSendReceive,
                         SuccessfulCrossHostSendReceiveTest,
                         ::testing::ValuesIn({1, 2, 3}),
                         SuccessfulCrossHostSendReceiveTestName);

struct SuccessfulCrossHostTransferTestParam {
  int num_rank_0_to_rank_1;
  int num_rank_1_to_rank_0;
};

std::string SuccessfulCrossHostTransferTestName(
    const ::testing::TestParamInfo<SuccessfulCrossHostTransferTestParam>&
        info) {
  return absl::StrFormat("num_rank0_to_rank1_%d_num_rank1_to_rank0_%d",
                         info.param.num_rank_0_to_rank_1,
                         info.param.num_rank_1_to_rank_0);
}

using SuccessfulCrossHostTransferTest =
    MultiProcessGpuTestWithParam<SuccessfulCrossHostTransferTestParam>;

TEST_P(SuccessfulCrossHostTransferTest, SuccessfulCrossHostTransfer) {
  const SuccessfulCrossHostTransferTestParam param = GetParam();
  const int rank_id = rank();
  const std::string log_prefix = rank_id == 0 ? "rank_0" : "rank_1";
  const int num_transfers =
      param.num_rank_0_to_rank_1 + param.num_rank_1_to_rank_0;

  std::vector<std::vector<int32_t>> transferred_data;
  transferred_data.reserve(num_transfers);
  for (int i = 0; i < param.num_rank_0_to_rank_1; ++i) {
    std::vector<int32_t> curr_data(256);
    absl::c_iota(curr_data, 1000 * i);
    transferred_data.push_back(std::move(curr_data));
  }
  for (int i = 0; i < param.num_rank_1_to_rank_0; ++i) {
    std::vector<int32_t> curr_data(256);
    absl::c_iota(curr_data, 10000 + 1000 * i);
    transferred_data.push_back(std::move(curr_data));
  }
  Shape shape = ShapeUtil::MakeShape(S32, {256});
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * default_memory_space,
      client()->addressable_devices()[0]->default_memory_space());

  std::vector<int32_t> initial_zero_values(256, 0);
  std::vector<std::unique_ptr<PjRtBuffer>> owned_buffers;
  owned_buffers.reserve(num_transfers);
  std::vector<PjRtDeviceEventPromiseRef> usage_event_promises;
  usage_event_promises.reserve(num_transfers);
  std::vector<CommonPjRtClient::CrossHostTransferSpec> transfer_specs;
  transfer_specs.reserve(num_transfers);
  PjRtDeviceEventRefVector transfer_dependencies;

  LOG(INFO) << log_prefix << ": preparing transfers.";
  for (int i = 0; i < num_transfers; ++i) {
    int src_global_device_id = i < param.num_rank_0_to_rank_1 ? 0 : 1;
    int dst_global_device_id = i < param.num_rank_0_to_rank_1 ? 1 : 0;
    bool is_sender = rank_id == src_global_device_id;

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> buffer,
        client()->BufferFromHostBuffer(
            /*data=*/is_sender ? transferred_data[i].data()
                               : initial_zero_values.data(),
            shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
            default_memory_space, /*device_layout=*/nullptr));

    PjRtDeviceEventPromiseRef usage_event_promise;
    PjRtDeviceEventRef usage_event;
    ASSERT_OK_AND_ASSIGN(
        std::tie(usage_event_promise, usage_event),
        absl::down_cast<CommonPjRtClient*>(client())->CreateLinkedEventPromise(
            default_memory_space, absl::StrFormat("buffer %i", i)));
    usage_event_promises.push_back(std::move(usage_event_promise));

    PjRtRawBufferRef raw_buffer;
    ASSERT_OK(
        absl::down_cast<CommonPjRtBufferImpl*>(buffer.get())
            ->AcquireScopedRawBuffer(
                [&](PjRtRawBufferRef buf_raw_buffer,
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

    transfer_specs.push_back(CommonPjRtClient::CrossHostTransferSpec{
        GlobalDeviceId(src_global_device_id),
        GlobalDeviceId(dst_global_device_id), std::move(raw_buffer)});

    owned_buffers.push_back(std::move(buffer));
    LOG(INFO) << log_prefix << ": finished preparing transfer " << i;
  }

  LOG(INFO) << log_prefix << ": enqueuing transfers";
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRefVector usage_events,
      absl::down_cast<CommonPjRtClient*>(client())->CrossHostTransferBuffers(
          std::move(transfer_dependencies), std::move(transfer_specs)));
  ASSERT_EQ(usage_events.size(), num_transfers);

  LOG(INFO) << log_prefix << ": setting usage events";
  for (int i = 0; i < usage_events.size(); ++i) {
    usage_event_promises[i].Set(usage_events[i].CopyRef());
  }

  LOG(INFO) << log_prefix << ": waiting for transfers to complete";
  for (int i = 0; i < usage_events.size(); ++i) {
    BlockUntilReady(usage_events[i].down_cast<BufferSequencingEvent>());
  }

  for (int i = 0; i < num_transfers; ++i) {
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> buffer_literal,
                         owned_buffers[i]->ToLiteral().Await());
    Literal expected_literal =
        LiteralUtil::CreateR1<int32_t>(transferred_data[i]);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *buffer_literal));
    LOG(INFO) << log_prefix << ": finished verification of transfer " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SuccessfulCrossHostTransfer, SuccessfulCrossHostTransferTest,
    ::testing::ValuesIn(std::vector<SuccessfulCrossHostTransferTestParam>{
        {1, 0}, {1, 1}, {2, 1}}),
    SuccessfulCrossHostTransferTestName);

using InterProcessCollectiveInitTest =
    MultiProcessGpuTestWithParam<std::string>;

TEST_P(InterProcessCollectiveInitTest, EagerInit) {
  const std::string& collectives_impl = GetParam();

  // The MORI backend is ROCm-only; skip on any other platform.
  if (collectives_impl == "mori") {
    ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                         PlatformUtil::GetPlatform("gpu"));
    if (platform->Name() != "ROCM") {
      GTEST_SKIP() << "MORI backend is only available on the ROCM platform";
    }
  }

  const char* prev_xla_flags_ptr = std::getenv("XLA_FLAGS");
  std::optional<std::string> prev_xla_flags =
      prev_xla_flags_ptr != nullptr
          ? std::optional<std::string>(prev_xla_flags_ptr)
          : std::nullopt;
  absl::Cleanup restore_flags = [&]() {
    if (prev_xla_flags.has_value()) {
      tsl::setenv("XLA_FLAGS", prev_xla_flags->c_str(), /*overwrite=*/true);
    } else {
      tsl::setenv("XLA_FLAGS", "--xla_flags_reset=true", /*overwrite=*/true);
    }
    xla::ResetFlagValues();
    if (!prev_xla_flags.has_value()) {
      tsl::unsetenv("XLA_FLAGS");
    }
  };
  if (collectives_impl != "default") {
    std::string new_flags = absl::StrFormat(
        "%s --xla_flags_reset=true --xla_gpu_collectives_implementation=%s",
        prev_xla_flags.value_or(""), collectives_impl);
    tsl::setenv("XLA_FLAGS", new_flags.c_str(), /*overwrite=*/true);
    xla::ResetFlagValues();
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> custom_client,
                       CreateCustomPjRtClient());

  ASSERT_EQ(custom_client->addressable_device_count(), 1);
  ASSERT_EQ(custom_client->device_count(), num_ranks());

  auto* se_client = absl::down_cast<CommonPjRtClientImpl*>(custom_client.get());
  ASSERT_NE(se_client, nullptr);
  auto* raw_client =
      absl::down_cast<PjRtStreamExecutorRawClient*>(se_client->raw_client());
  ASSERT_OK_AND_ASSIGN(
      LocalDeviceState * local_device_state,
      raw_client->GetLocalDeviceState(
          custom_client->addressable_devices()[0]->local_device_id()));
  se::StreamExecutor* executor = local_device_state->executor();

  constexpr uint64_t kCollectiveBytes = 1024;
  se::DeviceAddressBase mem = executor->Allocate(
      kCollectiveBytes, static_cast<int64_t>(se::MemorySpace::kCollective));
  EXPECT_NE(mem.opaque(), nullptr);
  executor->Deallocate(&mem);
}

INSTANTIATE_TEST_SUITE_P(InterProcessCollectiveInit,
                         InterProcessCollectiveInitTest,
                         ::testing::Values(std::string("default"),
                                           std::string("mori")),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

class CapturingLogSink final : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(mu_);
    absl::StrAppend(&logs_, entry.text_message_with_newline());
  }

  std::string GetLogs() const {
    absl::MutexLock lock(mu_);
    return logs_;
  }

 private:
  mutable absl::Mutex mu_;
  std::string logs_ ABSL_GUARDED_BY(mu_);
};

struct ShardedAutotuningTestInfo {
  int num_active_nodes;
  int num_nodes_using_cache;

  static std::string Name(
      const ::testing::TestParamInfo<ShardedAutotuningTestInfo>& info) {
    return absl::StrFormat("active_%d_cache_%d", info.param.num_active_nodes,
                           info.param.num_nodes_using_cache);
  }
};

using ShardedAutotuningTest =
    MultiProcessGpuTestWithParam<ShardedAutotuningTestInfo>;

TEST_P(ShardedAutotuningTest, ShardedAutotuningWorks) {
  const ShardedAutotuningTestInfo param = GetParam();

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetPlatform("gpu"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(rank()));
  const se::DeviceDescription& desc = executor->GetDeviceDescription();
  if (desc.gpu_compute_capability().IsCuda() && !desc.gpu_compute_capability()
                                                     .cuda_compute_capability()
                                                     ->IsAtLeastAmpere()) {
    GTEST_SKIP() << "Requires Ampere+ GPU.";
  }

  ASSERT_EQ(client()->addressable_device_count(), 1);
  ASSERT_EQ(client()->device_count(), num_ranks());

  std::string cache_dir;
  const std::string cache_dir_key = absl::StrCat(CurrentTestId(), "_cache_dir");
  if (rank() == 0) {
    ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&cache_dir));
    ASSERT_OK(distributed_client()->KeyValueSet(cache_dir_key, cache_dir));
  } else {
    ASSERT_OK_AND_ASSIGN(cache_dir, distributed_client()->BlockingKeyValueGet(
                                        cache_dir_key, absl::Seconds(30)));
  }

  const int prev_autotuner_vlog = absl::SetVLogLevel("autotuner_pass", 10);
  const int prev_assigner_vlog = absl::SetVLogLevel("config_assigner", 10);
  absl::Cleanup restore_vlog = [&]() {
    absl::SetVLogLevel("autotuner_pass", prev_autotuner_vlog);
    absl::SetVLogLevel("config_assigner", prev_assigner_vlog);
  };

  constexpr absl::string_view kHlo = R"hlo(
    HloModule main
    ENTRY main {
      %p0 = f16[2,32,32] parameter(0)
      ROOT %dot = f16[2,32,32] dot(%p0, %p0), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )hlo";

  // Compile twice to test both empty and non-empty disk cache.
  for (int iteration = 0; iteration < 2; ++iteration) {
    InMemoryStore::Clear();
    gpu::AutotunerCache::ClearAutotuneResults();

    ASSERT_OK(distributed_client()->WaitAtBarrier(
        absl::StrFormat("%s_iter_%d_start", CurrentTestId(), iteration),
        absl::Seconds(120), std::nullopt));

    if (rank() >= param.num_active_nodes) {
      continue;
    }

    CapturingLogSink log_sink;
    absl::AddLogSink(&log_sink);
    absl::Cleanup remove_sink = [&]() { absl::RemoveLogSink(&log_sink); };

    CompileOptions compile_options;
    compile_options.executable_build_options.set_num_replicas(
        param.num_active_nodes);
    compile_options.executable_build_options.set_key_value_store(
        CreateScopedKeyValueStore(absl::StrFormat("iter_%d", iteration)));
    DebugOptions& debug_options =
        *compile_options.executable_build_options.mutable_debug_options();
    debug_options.set_xla_gpu_shard_autotuning(true);
    debug_options.set_xla_gpu_cublas_fallback(false);

    if (rank() < param.num_nodes_using_cache) {
      debug_options.set_xla_gpu_experimental_autotune_cache_mode(
          DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE);
      debug_options.set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);
    }

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                         ParseAndReturnUnverifiedModule(kHlo));
    XlaComputation computation(hlo_module->ToProto());

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtLoadedExecutable> executable,
        client()->CompileAndLoad(computation, compile_options));

    ASSERT_OK_AND_ASSIGN(auto hlo_modules,
                         executable->GetExecutable()->GetHloModules());
    const std::string optimized_hlo = hlo_modules.front()->ToString();
    EXPECT_TRUE(absl::StrContains(optimized_hlo, "triton_gemm") ||
                absl::StrContains(optimized_hlo, "__triton_nested_gemm_fusion"))
        << optimized_hlo;

    int num_fusions_to_autotune = (rank() == 0) ? 1 : 0;
    if (iteration > 0 && rank() < param.num_nodes_using_cache) {
      num_fusions_to_autotune = 0;
    }
    const std::string logs = log_sink.GetLogs();
    if (num_fusions_to_autotune > 0) {
      EXPECT_THAT(
          logs, HasSubstr(absl::StrFormat(
                    "Shard %d/%d: finding configs for %d/1 unique instructions",
                    rank(), num_ranks(), num_fusions_to_autotune)));
    } else {
      EXPECT_THAT(logs, HasSubstr("Found cached config for HLO"));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ShardedAutotuningTest, ShardedAutotuningTest,
    ::testing::ValuesIn(std::vector<ShardedAutotuningTestInfo>{
        {2, 0}, {2, 1}, {2, 2}}),
    ShardedAutotuningTestInfo::Name);

}  // namespace
}  // namespace xla
