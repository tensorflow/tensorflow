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
#ifndef XLA_PYTHON_TRANSFER_STREAMING_H_
#define XLA_PYTHON_TRANSFER_STREAMING_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/python/transfer/transfer_socket.pb.h"

namespace aux {

// Destination for data-chunks.
class ChunkDestination : public tsl::ReferenceCounted<ChunkDestination> {
 public:
  virtual ~ChunkDestination() = default;

  // Must call on_done when done copying out of data.
  virtual absl::Status Put(const void* data, int64_t offset, size_t size,
                           absl::AnyInvocable<void() &&> on_done) = 0;

  virtual void Poison(absl::Status s) = 0;

  // For testing.
  static std::pair<xla::PjRtFuture<std::string>,
                   tsl::RCReference<ChunkDestination>>
  MakeStringDest();
};

// Suballocates from data allocations of size max_allocation_size.
class SlabAllocator {
 public:
  SlabAllocator(std::shared_ptr<absl::Span<uint8_t>> data,
                size_t max_allocation_size);
  ~SlabAllocator();

  // A single subrange allocation.
  struct Allocation {
    void* data;
    size_t size;
    // Must be called when data is no longer needed.
    absl::AnyInvocable<void() &&> on_done;
  };

  // Blocking allocation.
  Allocation Allocate(size_t size);

  // The max size of the suballocations.
  size_t max_allocation_size() const;

 private:
  struct State : public tsl::ReferenceCounted<State> {
    size_t max_allocation_size;
    std::shared_ptr<absl::Span<uint8_t>> data;
    absl::Mutex mu;
    std::vector<void*> slots;
    static bool HasSlots(State* state) { return !state->slots.empty(); }
  };
  tsl::RCReference<State> state_;
};

// Allocates some memory which is read-only and pinned into
// the network stack for zero copy receives.
absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>>
AllocateNetworkPinnedMemory(size_t size);

// Allocates some memory which is aligned up to page alignment.
absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> AllocateAlignedMemory(
    size_t size);

// A bulk transport implementation directly sends frames over the network.
class BulkTransportInterface {
 public:
  virtual ~BulkTransportInterface() = default;

  struct Message {
    void* data;
    size_t size;
    // Called when the consumer is done with the message.
    absl::AnyInvocable<void() &&> on_done;
  };

  struct SendMessage : public Message {
    // There may be some delay between Send() and when the message
    // is actually sent. on_send gets called when the message actually
    // gets sent.
    absl::AnyInvocable<void(int bond_id, size_t size) &&> on_send;
  };

  // Schedules a send over a BulkTransportInterface connection.
  virtual void Send(SendMessage msg) = 0;

  // Receives a fixed size message (size and bond_id) must match those
  // sent by send_message (in the same order).
  virtual void Recv(
      size_t size, int bond_id,
      absl::AnyInvocable<void(absl::StatusOr<Message> msg) &&> on_recv) = 0;

  // Creates an example bulk-transport which runs locally.
  static std::pair<std::unique_ptr<BulkTransportInterface>,
                   std::unique_ptr<BulkTransportInterface>>
  MakeLocalBulkTransportPair();

  // For testing.
  static SendMessage MakeMessage(
      std::string message,
      absl::AnyInvocable<void(int bond_id, size_t size) &&> on_send);
};

// A BulkTransportFactory allows establishing a BulkTransportInterface
// connection between two processes through
// the SocketTransferEstablishBulkTransport protocol.
//
// The API for this is:
//  auto a = transport_factory->InitBulkTransport();
//  auto b = transport_factory->RecvBulkTransport(a.request);
//  std::move(a.start_bulk_transport)(b.request);
class BulkTransportFactory {
 public:
  virtual ~BulkTransportFactory() = default;

  struct BulkTransportInitResult {
    // request to send to peer.
    SocketTransferEstablishBulkTransport request;
    // Actual interface to send and receive on.
    std::unique_ptr<BulkTransportInterface> bulk_transport;
    // Must be called to finalize the setup once the peer has replied.
    absl::AnyInvocable<void(const SocketTransferEstablishBulkTransport&
                                remote_bulk_transport_info) &&>
        start_bulk_transport;
  };
  // Creates 1/2 of the bulk transport.
  virtual BulkTransportInitResult InitBulkTransport() = 0;

  struct BulkTransportRecvResult {
    // Reply from the peer.
    SocketTransferEstablishBulkTransport request;
    // Actual interface to send and receive on.
    std::unique_ptr<BulkTransportInterface> bulk_transport;
  };
  // Receives the bulk transport request.
  virtual BulkTransportRecvResult RecvBulkTransport(
      const SocketTransferEstablishBulkTransport&
          remote_bulk_transport_info) = 0;

  // Creates a factory (mostly for testing) which runs entirely
  // locally.
  static std::shared_ptr<BulkTransportFactory> CreateLocal();
};

// Implementations may subclass this to represent the state of a connection.
class ConnectionState : public tsl::ReferenceCounted<ConnectionState> {
 public:
  virtual ~ConnectionState() = default;

  // Publishes a frame of the buffer. Calls on_done when data is finished being
  // used.
  virtual void Send(size_t req_id, const void* data, size_t offset, size_t size,
                    bool is_largest, absl::AnyInvocable<void() &&> on_done) = 0;
};

// Basic rendevous table.
class PullTable {
 public:
  class Entry : public tsl::ReferenceCounted<Entry> {
   public:
    virtual ~Entry() = default;

    // Must call Send() on state when the transfer is ready. Result buffers are
    // offset by base_req_id.
    virtual bool Handle(tsl::RCReference<ConnectionState> state,
                        const SocketTransferPullRequest& req,
                        size_t base_req_id) = 0;
  };

  // Registers an entry in the pull table to be pulled at a later point.
  void AwaitPull(uint64_t uuid, tsl::RCReference<Entry> entry);

  // Will call Send() for each of the copied buffers in req offset by
  // base_req_id (assigned sequentially). (Delegates to PullTable::Entry).
  void Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req, size_t base_req_id);

  // Test-only implementation of PullTable::Entry for a list of strings.
  static tsl::RCReference<PullTable::Entry> MakeStringEntry(
      std::vector<std::string> buffers);

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<uint64_t, tsl::RCReference<Entry>> entries_;
  struct PausedFetch {
    tsl::RCReference<ConnectionState> state;
    SocketTransferPullRequest req;
    size_t base_req_id;
  };
  absl::flat_hash_map<uint64_t, std::vector<PausedFetch>> paused_fetches_;
};

}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_STREAMING_H_
