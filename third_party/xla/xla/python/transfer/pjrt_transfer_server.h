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

#ifndef XLA_PYTHON_TRANSFER_PJRT_TRANSFER_SERVER_H_
#define XLA_PYTHON_TRANSFER_PJRT_TRANSFER_SERVER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/transfer_server_interface.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/socket-server.h"
#include "xla/python/transfer/streaming_ifrt.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class PjRtTransferServer : public TransferServerInterface {
 public:
  // Factory for creating a PjRtTransferServer. Must be called with a mutex
  // held.
  using PjRtTransferServerFactory =
      std::function<absl::StatusOr<std::unique_ptr<PjRtTransferServer>>(
          std::shared_ptr<xla::PjRtClient>)>;

  ~PjRtTransferServer() override;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> CopyArraysForCrossHost(
      xla::ifrt::PjRtClient* client, absl::Span<ArrayRef> arrays,
      DeviceListRef src_devices, DeviceListRef dst_devices,
      std::optional<MemoryKind> memory_kind) override;

  // Awaits a pull from a remote process.
  absl::Status CrossHostAwaitPull(int64_t uuid,
                                  absl::Span<xla::ifrt::ArrayRef> arrays,
                                  const std::vector<int>& buffer_idxs) override;

  // Pulls buffers from a remote process.
  absl::Status CrossHostPull(
      int64_t uuid, absl::Span<xla::ifrt::ArrayRef> arrays,
      std::vector<int>& dst_device_idxs, xla::ifrt::DeviceListRef dst_devices,
      std::optional<MemoryKind> memory_kind, int remote_pid,
      absl::btree_map<int, PjRtArray::PjRtBuffers>& buffer_list) override;

  static absl::StatusOr<PjRtTransferServerFactory>
  MakePjRtTransferServerFactory(
      size_t transfer_size, absl::Duration cross_host_transfer_timeout,
      std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
      const std::string& socket_address,
      const std::vector<std::string>& transport_addresses);

  PjRtTransferServer(std::shared_ptr<xla::PjRtClient> pjrt_client,
                     size_t max_num_parallel_copies, size_t transfer_size,
                     absl::Duration cross_host_transfer_timeout,
                     std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
                     aux::SocketAddress socket_address,
                     std::vector<aux::SocketAddress> transport_addresses);

 private:
  // Starts the DCN SocketServer.
  absl::Status StartTransferServer();

  int64_t CreateNewTransferKey();
  absl::StatusOr<tsl::RCReference<aux::SocketServer::Connection>> GetConnection(
      int remote_pid) ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mu_);

  std::shared_ptr<xla::PjRtClient> pjrt_client_;
  size_t max_num_parallel_copies_;
  size_t transfer_size_;
  absl::Duration cross_host_transfer_timeout_;
  std::shared_ptr<xla::KeyValueStoreInterface> kv_store_;
  aux::SocketAddress socket_address_;
  std::vector<aux::SocketAddress> transport_addresses_;
  std::optional<std::shared_ptr<aux::SocketServer>> socket_server_;
  std::optional<std::shared_ptr<aux::PremappedCopierState>> premapped_copier_;
  std::atomic<int64_t> next_transfer_key_ = 0;
  absl::flat_hash_map<int, tsl::RCReference<aux::SocketServer::Connection>>
      connections_;
  absl::Mutex connections_mu_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_TRANSFER_PJRT_TRANSFER_SERVER_H_
