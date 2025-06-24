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
#ifndef XLA_PYTHON_TRANSFER_SOCKET_SERVER_H_
#define XLA_PYTHON_TRANSFER_SOCKET_SERVER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux {

// Basic server for publishing buffers and connecting to fetch buffers from
// other servers.
class SocketServer {
 public:
  SocketServer() = default;

  // Address that this server is listening on.
  const SocketAddress& addr() { return listener_->addr(); }

  // Starts listening for connections on addr. Bulk transports happen
  // over a transport constructed from the factory.
  absl::Status Start(
      const SocketAddress& addr,
      std::shared_ptr<BulkTransportFactory> bulk_transport_factory);

  // Registers an entry for a particular uuid which is a list of buffers.
  void AwaitPull(uint64_t uuid, tsl::RCReference<PullTable::Entry> handler) {
    pull_table_->AwaitPull(uuid, std::move(handler));
  }

  class SocketNetworkState;

  // Connection state.
  class Connection : public tsl::ReferenceCounted<Connection> {
   public:
    explicit Connection(SocketNetworkState* local) : local_(local) {}
    ~Connection();

    // Fetch a particular buffer from a remote server.
    void Pull(uint64_t uuid, int buffer_id,
              tsl::RCReference<ChunkDestination> dest);

    // Fetch a list of buffers from a remote server.
    void Pull(uint64_t uuid, absl::Span<const int> buffer_ids,
              std::vector<tsl::RCReference<ChunkDestination>> dests);

    void InjectFailure();

   private:
    SocketNetworkState* local_;
  };

  // Connect to a remote server at an address.
  tsl::RCReference<Connection> Connect(const SocketAddress& other_addr);

 private:
  std::unique_ptr<SocketListener> listener_;
  std::shared_ptr<BulkTransportFactory> bulk_transport_factory_;
  std::shared_ptr<PullTable> pull_table_ = std::make_shared<PullTable>();
};

}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_SOCKET_SERVER_H_
