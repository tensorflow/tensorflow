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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/utils.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/transfer_manager.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

TfrtGpuDevice::TfrtGpuDevice(Options&& options)
    : id_(options.id),
      local_device_id_(options.local_device_id),
      local_hardware_id_(options.local_hardware_id),
      executor_(options.executor),
      stream_(MaybeCreateStream(options.executor)),
      d2h_stream_(MaybeCreateStream(options.executor)),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<GpuEvent>()),
      description_(options.id, local_device_id_.value(), options.process_index,
                   options.process_index_in_partition, options.partition_index,
                   options.platform_version),
      max_inflight_computations_semaphore_(
          /*capacity=*/options.max_inflight_computations) {
  std::vector<int64_t> v_coords(description_.coords().begin(),
                                description_.coords().end());

  description_.SetAttributes({
      {"coords", xla::PjRtDeviceAttribute(v_coords)},
      {"device_vendor", options.device_vendor},
      // TODO - b/435521225: `slice_index` is deprecated. Use `partition_index`,
      // which better aligns with NVIDIA's terminology.
      {"slice_index", static_cast<int64_t>(options.partition_index)},
      {"partition_index", static_cast<int64_t>(options.partition_index)},
      {"compute_capability",
       xla::PjRtDeviceAttribute(options.compute_capability)},
      {"core_count", static_cast<int64_t>(options.core_count)},
  });

  description_.SetDebugString(absl::StrCat("TFRT_GPU_", id_));
  description_.SetToString(absl::StrCat("GpuDevice(id=", id_, ")"));
}

TfrtGpuDevice::~TfrtGpuDevice() {
  // Bounce through a thread to avoid calling CUDA inline.
  absl::WrapUnique(
      tsl::Env::Default()->StartThread({}, "TfrtGpuDeviceDestructor", [&]() {
        // Block the host until all pending work on the stream is done. This is
        // to avoid user-after-free errors in host callbacks.
        if (stream() != nullptr) {
          absl::Status status = BlockHostUntilDoneWithHostCallback(stream());
          if (!status.ok()) {
            LOG(ERROR) << "Failed to wait for stream to finish: " << status;
          }
        }
        if (d2h_stream() != nullptr) {
          absl::Status status =
              BlockHostUntilDoneWithHostCallback(d2h_stream());
          if (!status.ok()) {
            LOG(ERROR) << "Failed to wait for d2h stream to finish: " << status;
          }
        }
      }));
}

PjRtClient* TfrtGpuDevice::client() const { return client_; }

void TfrtGpuDevice::SetClient(TfrtGpuClient* client) {
  CHECK(client_ == nullptr);
  client_ = client;

  // We have to define debug_string_ and to_string_ here, because
  // platform_name() requires client_ to be set.
  CHECK(!client_->platform_name().empty());
  std::string device_name =
      absl::StrCat(MakeAsciiTitlecase(client_->platform_name()), "Device");
  description_.SetDebugString(
      absl::StrCat(client_->platform_name(), ":", id()));
  description_.SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
}

absl::StatusOr<TransferManager*> TfrtGpuDevice::GetTransferManager() {
  // Downcast Base class to TfrtGpuClient.
  if (client_ == nullptr) {
    return absl::InternalError("Client is null");
  }
  return client_->xla_client()->backend().transfer_manager();
}

absl::Status TfrtGpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager, GetTransferManager());
  return RunOnAsyncWorkRunner(client_->blocking_thread_pool(), [&]() {
    return transfer_manager->TransferLiteralToInfeed(executor(), literal);
  });
}

absl::Status TfrtGpuDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager, GetTransferManager());
  return RunOnAsyncWorkRunner(client_->blocking_thread_pool(), [&]() {
    return transfer_manager->TransferLiteralFromOutfeed(executor(), literal);
  });
}

int TfrtGpuDevice::GetNewPrngSeed() {
  absl::MutexLock lock(mu_);
  int x = 0;
  do {
    x = prng_seed_distribution_(prng_seed_generator_);
  } while (x == 0);
  return x;
}

void TfrtGpuDevice::AttachMemorySpace(PjRtMemorySpace* memory_space,
                                      bool is_default) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a TfrtGpuExecutable to a PjRtMemorySpace owned "
      "by a different client, the device's client: %s, the memory space's "
      "client: %s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_kind_id_.emplace(memory_space->kind_id(), memory_space);
  if (is_default) {
    CHECK(default_memory_space_ == nullptr)
        << "Default memory space already set to "
        << default_memory_space_->DebugString() << ".";
    default_memory_space_ = memory_space;
  }
}

absl::Span<PjRtMemorySpace* const> TfrtGpuDevice::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind_id(
    int id) const {
  auto it = memory_spaces_by_kind_id_.find(id);
  if (it == memory_spaces_by_kind_id_.end()) {
    return absl::InternalError(
        absl::StrCat("No memory space found (kind_id: ", id, ")"));
  }
  return it->second;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind(
    absl::string_view kind) const {
  auto it = absl::c_find_if(memory_spaces_, [kind](PjRtMemorySpace* ms) {
    return ms->kind() == kind;
  });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", kind, ")"));
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::default_memory_space() const {
  if (default_memory_space_ == nullptr) {
    return absl::InternalError(
        "No default memory space is set for this device.");
  }
  return default_memory_space_;
}

absl::StatusOr<tsl::AllocatorStats> TfrtGpuDevice::GetAllocatorStats() const {
  if (!IsAddressable()) {
    return FailedPrecondition(
        "GetAllocatorStats() is allowed only for addressable devices");
  }

  auto* allocator_adapter =
      dynamic_cast<se::MultiDeviceAdapter*>(client_->allocator());
  if (!allocator_adapter) {
    return Unimplemented(
        "GetAllocatorStats() is only implemented with MultiDeviceAdapter "
        "allocator");
  }

  TF_ASSIGN_OR_RETURN(auto allocator, allocator_adapter->GetAllocator(
                                          local_device_id().value()));

  auto stats = allocator->GetStats();
  TF_RET_CHECK(stats.has_value());
  return stats.value();
}

tsl::AsyncValueRef<GpuEvent> TfrtGpuDevice::SetLastCollectiveLaunchEvent(
    tsl::AsyncValueRef<GpuEvent> event) {
  absl::MutexLock lock(mu_);
  VLOG(3) << "SetLastCollectiveLaunchEvent: IsAvailable: "
          << event.IsAvailable() << "; pointer: " << event.GetAsyncValue()
          << "Old Event: IsAvailable: "
          << last_collective_launch_event_.IsAvailable()
          << "; pointer: " << last_collective_launch_event_.GetAsyncValue();
  std::swap(last_collective_launch_event_, event);
  return event;
}

}  // namespace xla
