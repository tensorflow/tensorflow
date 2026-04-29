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

#include "xla/python/transfer/pjrt_transfer_server.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/socket-server.h"
#include "xla/python/transfer/socket_bulk_transport.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/streaming_ifrt.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace ifrt {
namespace {

absl::StatusOr<xla::PjRtMemorySpace*> GetMemorySpace(
    std::optional<MemoryKind> memory_kind, xla::ifrt::Device* device) {
  if (memory_kind.has_value()) {
    xla::ifrt::MemoryKind canonical_memory_kind =
        CanonicalizeMemoryKind(*memory_kind, device);
    xla::ifrt::Memory* memory = nullptr;
    for (xla::ifrt::Memory* ms : device->Memories()) {
      if (ms->Kind() == canonical_memory_kind) {
        memory = ms;
        break;
      }
    }
    if (memory == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid memory kind: %s; available memory kinds: %s",
          *canonical_memory_kind.memory_kind(),
          absl::StrJoin(device->Memories(), ", ",
                        [](std::string* out, xla::ifrt::Memory* ms) {
                          absl::StrAppend(out, *ms->Kind().memory_kind());
                        })));
    }
    return absl::down_cast<PjRtMemory*>(memory)->pjrt_memory();
  }
  return absl::down_cast<PjRtDevice*>(device)
      ->pjrt_device()
      ->default_memory_space();
}

const char kKeyPrefixSocketAddress[] = "ifrt_cross_host_socket_address_";

}  // namespace

PjRtTransferServer::PjRtTransferServer(
    std::shared_ptr<xla::PjRtClient> pjrt_client,
    size_t max_num_parallel_copies, size_t transfer_size,
    absl::Duration cross_host_transfer_timeout,
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
    aux::SocketAddress socket_address,
    std::vector<aux::SocketAddress> transport_addresses)
    : pjrt_client_(pjrt_client),
      max_num_parallel_copies_(max_num_parallel_copies),
      transfer_size_(transfer_size),
      cross_host_transfer_timeout_(cross_host_transfer_timeout),
      kv_store_(kv_store),
      socket_address_(socket_address),
      transport_addresses_(transport_addresses) {}

PjRtTransferServer::~PjRtTransferServer() {
  connections_.clear();
  if (socket_server_.has_value()) {
    (*socket_server_)->WaitForQuiesce();
  }
  socket_server_ = std::nullopt;
}

absl::StatusOr<PjRtTransferServer::PjRtTransferServerFactory>
PjRtTransferServer::MakePjRtTransferServerFactory(
    size_t transfer_size, absl::Duration cross_host_transfer_timeout,
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
    const std::string& socket_address,
    const std::vector<std::string>& transport_addresses) {
  TF_ASSIGN_OR_RETURN(aux::SocketAddress address,
                      aux::SocketAddress::Parse(socket_address));
  std::vector<aux::SocketAddress> transport_socket_addresses;
  if (transport_addresses.empty()) {
    TF_ASSIGN_OR_RETURN(aux::SocketAddress transport_address,
                        aux::SocketAddress::Parse("0.0.0.0:0"));
    // TODO(emilyaf, parkers): Remove this once defaults are set per device
    // platform.
    transport_socket_addresses.reserve(4);
    for (int i = 0; i < 4; ++i) {
      transport_socket_addresses.push_back(transport_address);
    }
  } else {
    transport_socket_addresses.reserve(transport_addresses.size());
    for (const std::string& transport_address : transport_addresses) {
      TF_ASSIGN_OR_RETURN(aux::SocketAddress socket_address,
                          aux::SocketAddress::Parse(transport_address));
      transport_socket_addresses.push_back(socket_address);
    }
  }
  PjRtTransferServer::PjRtTransferServerFactory factory =
      [transfer_size, cross_host_transfer_timeout, kv_store, address,
       transport_socket_addresses](std::shared_ptr<xla::PjRtClient> client)
      -> absl::StatusOr<std::unique_ptr<PjRtTransferServer>> {
    auto transfer_server = std::make_unique<PjRtTransferServer>(
        client, client->addressable_device_count() * 2, transfer_size,
        cross_host_transfer_timeout, kv_store, address,
        transport_socket_addresses);
    TF_RETURN_IF_ERROR(transfer_server->StartTransferServer());
    return transfer_server;
  };
  return factory;
}

absl::Status PjRtTransferServer::StartTransferServer() {
  // Populate the KV store with this process's socket address.
  TF_RETURN_IF_ERROR(kv_store_->Set(
      absl::StrCat(kKeyPrefixSocketAddress, pjrt_client_->process_index()),
      socket_address_.ToString()));

  size_t total_size = transfer_size_ * max_num_parallel_copies_;
  TF_ASSIGN_OR_RETURN(auto tmp, aux::AllocateAlignedMemory(total_size));
  TF_ASSIGN_OR_RETURN(auto map, aux::MapPjrtMemory(pjrt_client_, tmp->data(),
                                                   tmp->size(), tmp));
  aux::SlabAllocator uallocator(map, transfer_size_);
  TF_ASSIGN_OR_RETURN(auto factory, aux::CreateSocketBulkTransportFactory(
                                        transport_addresses_, std::nullopt,
                                        std::move(uallocator)));

  socket_server_ = std::make_shared<aux::SocketServer>();
  TF_ASSIGN_OR_RETURN(
      auto mem, aux::AllocateAndMapPjrtMemory(pjrt_client_, total_size * 2));
  premapped_copier_ = std::make_shared<aux::PremappedCopierState>(
      mem, max_num_parallel_copies_, transfer_size_);
  return (*socket_server_)->Start(socket_address_, factory);
}

absl::Status PjRtTransferServer::CrossHostAwaitPull(
    int64_t uuid, absl::Span<xla::ifrt::ArrayRef> arrays,
    const std::vector<int>& buffer_idxs) {
  if (!socket_server_.has_value()) {
    return absl::InternalError("Socket server is not initialized.");
  }
  std::vector<aux::PjRtBufferEntry::BufferRef> refs;
  refs.reserve(buffer_idxs.size() * arrays.size());
  for (xla::ifrt::ArrayRef& arr : arrays) {
    auto* pjrt_arr = llvm::dyn_cast_or_null<xla::ifrt::PjRtArray>(arr.get());
    if (pjrt_arr == nullptr) {
      return absl::InvalidArgumentError(
          "Cannot remote transfer non-pjrt arrays.");
    }
    if (pjrt_arr->pjrt_buffers().size() != buffer_idxs.size()) {
      return absl::InvalidArgumentError(
          "PjRtArray has different number of buffers than buffer_idxs.");
    }
    if (pjrt_arr->pjrt_buffers().empty()) {
      return absl::InvalidArgumentError("PjRtArray has no buffers.");
    }
    TF_ASSIGN_OR_RETURN(size_t buf_size,
                        pjrt_arr->pjrt_buffers()[0]->GetOnDeviceSizeInBytes());
    for (int j : buffer_idxs) {
      auto& pjrt_buf = pjrt_arr->pjrt_buffers()[j];
      refs.push_back({pjrt_buf, buf_size});
    }
  }
  auto state = tsl::MakeRef<aux::PjRtBufferEntry>(
      std::move(refs), *premapped_copier_, transfer_size_);
  (*socket_server_)->AwaitPull(uuid, state);
  return absl::OkStatus();
}

absl::StatusOr<tsl::RCReference<aux::SocketServer::Connection>>
PjRtTransferServer::GetConnection(int remote_pid) {
  if (!connections_.contains(remote_pid)) {
    TF_ASSIGN_OR_RETURN(
        std::string address,
        kv_store_->Get(absl::StrCat(kKeyPrefixSocketAddress, remote_pid),
                       cross_host_transfer_timeout_));
    TF_ASSIGN_OR_RETURN(auto addr, aux::SocketAddress::Parse(address));
    connections_[remote_pid] = (*socket_server_)->Connect(addr);
  }
  return connections_[remote_pid];
}

absl::Status PjRtTransferServer::CrossHostPull(
    int64_t uuid, absl::Span<xla::ifrt::ArrayRef> arrays,
    std::vector<int>& dst_device_idxs, xla::ifrt::DeviceListRef dst_devices,
    std::optional<MemoryKind> memory_kind, int remote_pid,
    absl::btree_map<int, PjRtArray::PjRtBuffers>& buffer_list) {
  if (!socket_server_.has_value()) {
    return absl::InternalError("Socket server is not initialized.");
  }
  tsl::RCReference<aux::SocketServer::Connection> connection;
  {
    absl::MutexLock lock(connections_mu_);
    TF_ASSIGN_OR_RETURN(connection, GetConnection(remote_pid));
  }

  std::vector<xla::PjRtClient::ShapeSpec> shape_specs;
  std::vector<std::optional<xla::Layout>> layouts;
  shape_specs.reserve(arrays.size());
  layouts.reserve(arrays.size());
  for (int i = 0; i < arrays.size(); ++i) {
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType prim_type,
                        xla::ifrt::ToPrimitiveType(arrays[i]->dtype()));
    TF_ASSIGN_OR_RETURN(
        Shape shape, arrays[i]->sharding().GetShardShape(arrays[i]->shape()));
    xla::PjRtClient::ShapeSpec shape_spec = {
        prim_type,
        xla::DimensionVector(shape.dims().begin(), shape.dims().end())};
    shape_specs.push_back(shape_spec);

    absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> pjrt_layout =
        arrays[i]->pjrt_layout();
    std::optional<xla::Layout> layout;
    if (pjrt_layout.ok() && *pjrt_layout == nullptr) {
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Shape shard_shape,
          arrays[i]->sharding().GetShardShape(arrays[i]->shape()));
      TF_ASSIGN_OR_RETURN(
          pjrt_layout, arrays[i]->client()->GetDefaultPjRtLayout(
                           arrays[i]->dtype(), shard_shape.dims(),
                           arrays[i]->sharding().devices()->devices().front(),
                           arrays[i]->sharding().memory_kind()));
    }
    if (pjrt_layout.ok()) {
      layout = (*pjrt_layout)->xla_layout();
    }
    layouts.push_back(layout);
  }

  for (int j = 0; j < dst_device_idxs.size(); ++j) {
    int device_index = dst_device_idxs[j];
    TF_ASSIGN_OR_RETURN(
        xla::PjRtMemorySpace * mem_space,
        GetMemorySpace(memory_kind, dst_devices->devices()[device_index]));
    // TODO(emilyaf, parkers): Pass `layouts` instead of `std::nullopt` once
    // ASAN failure is debugged.
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
        pjrt_client_->CreateBuffersForAsyncHostToDevice(
            shape_specs, std::nullopt, mem_space));
    buffer_list[device_index].reserve(arrays.size());
    for (int i = 0; i < arrays.size(); ++i) {
      const int buffer_id = dst_device_idxs.size() * i + j;
      auto chunk_dest = aux::MakeDmaDestination(atm, i, atm->buffer_size(i));
      connection->Pull(uuid, buffer_id, std::move(chunk_dest));
      buffer_list[device_index].push_back(atm->RetrieveBuffer(i));
    }
  }
  return absl::OkStatus();
}

int64_t PjRtTransferServer::CreateNewTransferKey() {
  return next_transfer_key_++;
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
PjRtTransferServer::CopyArraysForCrossHost(
    xla::ifrt::PjRtClient* client, absl::Span<ArrayRef> arrays,
    DeviceListRef src_devices, DeviceListRef dst_devices,
    std::optional<MemoryKind> memory_kind) {
  // Maps dst pid to src pids.
  absl::btree_map<int, absl::btree_set<int>> process_index_map;
  // Maps src pid to addressable destination device inds.
  absl::btree_map<int, std::vector<int>> pull_to_device_idxs;
  // Maps dst pid to source buffer inds.
  absl::btree_map<int, std::vector<int>> await_pull_buffer_idxs;
  int j = 0;
  for (int i = 0; i < src_devices->devices().size(); ++i) {
    if (dst_devices->devices()[i]->IsAddressable()) {
      if (src_devices->devices()[i]->IsAddressable()) {
        // TODO(emilyaf): Support host-local transfers alongside cross-host
        // transfers.
        return absl::UnimplementedError(absl::StrFormat(
            "Cross-host transfers are currently supported only if every "
            "shard requires a cross-host transfer. A host-local transfer "
            "was requested from device %s to device %s.",
            src_devices->devices()[i]->DebugString(),
            dst_devices->devices()[i]->DebugString()));
      }
      pull_to_device_idxs[src_devices->devices()[i]->ProcessIndex()].push_back(
          i);
    }
    if (src_devices->devices()[i]->IsAddressable()) {
      await_pull_buffer_idxs[dst_devices->devices()[i]->ProcessIndex()]
          .push_back(j++);
    }
    process_index_map[dst_devices->devices()[i]->ProcessIndex()].emplace(
        src_devices->devices()[i]->ProcessIndex());
  }

  absl::btree_map<int, PjRtArray::PjRtBuffers> buffers_by_device;
  for (auto& [dst_process_index, src_process_indices] : process_index_map) {
    for (auto& src_process_index : src_process_indices) {
      // The transfer key must be incremented in all processes, regardless of
      // whether they participate in the transfer.
      int64_t uuid = CreateNewTransferKey();
      if ((src_process_index != pjrt_client_->process_index()) &&
          (dst_process_index != pjrt_client_->process_index())) {
        continue;
      }
      if (src_process_index == pjrt_client_->process_index()) {
        TF_RETURN_IF_ERROR(CrossHostAwaitPull(
            uuid, arrays, await_pull_buffer_idxs[dst_process_index]));
      } else {
        TF_RETURN_IF_ERROR(CrossHostPull(
            uuid, arrays, pull_to_device_idxs[src_process_index], dst_devices,
            memory_kind, src_process_index, buffers_by_device));
      }
    }
  }

  std::vector<xla::ifrt::ArrayRef> new_arrays;
  new_arrays.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto new_sharding,
                        arrays[i]->shared_ptr_sharding()->WithDeviceAssignment(
                            dst_devices, memory_kind));
    TF_ASSIGN_OR_RETURN(auto new_layout, arrays[i]->pjrt_layout());
    if (new_layout == nullptr) {
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Shape shard_shape,
          arrays[i]->sharding().GetShardShape(arrays[i]->shape()));
      TF_ASSIGN_OR_RETURN(
          new_layout, arrays[i]->client()->GetDefaultPjRtLayout(
                          arrays[i]->dtype(), shard_shape.dims(),
                          arrays[i]->sharding().devices()->devices().front(),
                          arrays[i]->sharding().memory_kind()));
    }
    PjRtArray::PjRtBuffers array_buffers;
    array_buffers.reserve(buffers_by_device.size());
    for (auto& [_, bufs] : buffers_by_device) {
      array_buffers.push_back(std::move(bufs[i]));
    }
    TF_ASSIGN_OR_RETURN(
        auto arr,
        PjRtArray::Create(client, arrays[i]->dtype(), arrays[i]->shape(),
                          std::move(new_sharding), std::move(array_buffers),
                          std::move(new_layout)));
    new_arrays.push_back(std::move(arr));
  }
  return new_arrays;
}

}  // namespace ifrt
}  // namespace xla
