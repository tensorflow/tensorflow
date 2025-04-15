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
#include "xla/python/transfer/streaming.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux {

class StringFutureChunkDestination : public aux::ChunkDestination {
 public:
  explicit StringFutureChunkDestination(
      xla::PjRtFuture<std::string>::Promise dest)
      : dest_(std::move(dest)) {}
  ~StringFutureChunkDestination() override { dest_.Set(ConsumeFinalResult()); }
  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    {
      absl::MutexLock l(&mu_);
      chunks_.emplace_back(
          offset, std::string(reinterpret_cast<const char*>(data), size));
    }
    std::move(on_done)();
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> ConsumeFinalResult() {
    absl::MutexLock l(&mu_);
    std::sort(chunks_.begin(), chunks_.end());

    std::string result;
    for (auto& chunk : chunks_) {
      if (chunk.first != result.size()) {
        return absl::InvalidArgumentError("There are gaps in the chunk.");
      }
      result += chunk.second;
    }
    return result;
  }

 private:
  absl::Mutex mu_;
  std::vector<std::pair<size_t, std::string>> chunks_;
  xla::PjRtFuture<std::string>::Promise dest_;
};

std::pair<xla::PjRtFuture<std::string>, tsl::RCReference<ChunkDestination>>
ChunkDestination::MakeStringDest() {
  auto promise = xla::PjRtFuture<std::string>::CreatePromise();
  auto result = xla::PjRtFuture<std::string>(promise);
  return std::make_pair(
      result, tsl::MakeRef<StringFutureChunkDestination>(std::move(promise)));
}

class LocalBulkTransport : public BulkTransportInterface {
 public:
  struct QueueState {
    absl::Mutex mu;
    std::deque<Message> buffers;
    absl::Status status;
  };

  LocalBulkTransport(std::shared_ptr<QueueState> send_q,
                     std::shared_ptr<QueueState> recv_q)
      : send_q_(std::move(send_q)), recv_q_(std::move(recv_q)) {}

  ~LocalBulkTransport() override {
    absl::MutexLock l(&send_q_->mu);
    send_q_->status = absl::InternalError("Connection failure for local bond.");
  }
  void Send(SendMessage msg) override {
    absl::MutexLock l(&send_q_->mu);
    std::move(msg.on_send)(0, msg.size);
    send_q_->buffers.push_back(std::move(msg));
  }
  void Recv(size_t size, int bond_id,
            absl::AnyInvocable<void(absl::StatusOr<Message> msg) &&> on_recv)
      override {
    absl::StatusOr<Message> result;
    {
      absl::MutexLock l(&recv_q_->mu);
      auto cond = [&]() {
        return !recv_q_->buffers.empty() || !recv_q_->status.ok();
      };
      recv_q_->mu.Await(absl::Condition(&cond));
      if (recv_q_->buffers.empty()) {
        result = recv_q_->status;
      } else {
        result = std::move(recv_q_->buffers.front());
        recv_q_->buffers.pop_front();
        CHECK_EQ(result->size, size);
      }
    }
    std::move(on_recv)(std::move(result));
  }

 private:
  std::shared_ptr<QueueState> send_q_;
  std::shared_ptr<QueueState> recv_q_;
};

std::pair<std::unique_ptr<BulkTransportInterface>,
          std::unique_ptr<BulkTransportInterface>>
BulkTransportInterface::MakeLocalBulkTransportPair() {
  auto send_q = std::make_shared<LocalBulkTransport::QueueState>();
  auto recv_q = std::make_shared<LocalBulkTransport::QueueState>();
  return std::make_pair(std::make_unique<LocalBulkTransport>(send_q, recv_q),
                        std::make_unique<LocalBulkTransport>(recv_q, send_q));
}

BulkTransportInterface::SendMessage BulkTransportInterface::MakeMessage(
    std::string message,
    absl::AnyInvocable<void(int bond_id, size_t size) &&> on_send) {
  auto tmp = std::make_unique<std::string>(message);
  SendMessage result;
  result.data = tmp->data();
  result.size = tmp->size();
  result.on_send = std::move(on_send);
  result.on_done = [tmp = std::move(tmp)]() {};
  return result;
}

class LocalBulkTransportFactory : public BulkTransportFactory {
 public:
  BulkTransportInitResult InitBulkTransport() override {
    BulkTransportInitResult out;
    absl::MutexLock l(&mu_);
    out.request.set_bulk_transport_impl_kind(
        SocketTransferEstablishBulkTransport::LOCAL);
    out.request.add_bulk_transport_uuid(next_bulk_transport_id_);
    std::unique_ptr<BulkTransportInterface> local_bulk_transport;
    std::unique_ptr<BulkTransportInterface> remote_bulk_transport;
    std::tie(local_bulk_transport, remote_bulk_transport) =
        BulkTransportInterface::MakeLocalBulkTransportPair();
    out.bulk_transport = std::move(local_bulk_transport);
    local_bulk_transports_[next_bulk_transport_id_] =
        std::move(remote_bulk_transport);
    ++next_bulk_transport_id_;
    // No work for local bulk_transports.
    out.start_bulk_transport = [](const SocketTransferEstablishBulkTransport&
                                      remote_bulk_transport_info) {};
    return out;
  }
  BulkTransportRecvResult RecvBulkTransport(
      const SocketTransferEstablishBulkTransport& remote_bulk_transport_info)
      override {
    BulkTransportRecvResult out;
    absl::MutexLock l(&mu_);
    CHECK_EQ(remote_bulk_transport_info.bulk_transport_impl_kind(),
             SocketTransferEstablishBulkTransport::LOCAL);
    CHECK_EQ(remote_bulk_transport_info.bulk_transport_uuid_size(), 1);
    auto it = local_bulk_transports_.find(
        remote_bulk_transport_info.bulk_transport_uuid(0));
    CHECK(it != local_bulk_transports_.end());
    auto bulk_transport_out = std::move(it->second);
    local_bulk_transports_.erase(it);
    out.bulk_transport = std::move(bulk_transport_out);
    out.request.set_bulk_transport_impl_kind(
        SocketTransferEstablishBulkTransport::LOCAL);
    return out;
  }

 private:
  absl::Mutex mu_;
  uint64_t next_bulk_transport_id_ = 0;
  absl::flat_hash_map<uint64_t, std::unique_ptr<BulkTransportInterface>>
      local_bulk_transports_;
};

std::shared_ptr<BulkTransportFactory> BulkTransportFactory::CreateLocal() {
  return std::make_shared<LocalBulkTransportFactory>();
}

SlabAllocator::SlabAllocator(std::shared_ptr<absl::Span<uint8_t>> data,
                             size_t max_allocation_size) {
  state_ = tsl::TakeRef(new State);
  state_->data = data;
  state_->max_allocation_size = max_allocation_size;
  size_t offset = 0;
  for (size_t i = 0; i < data->size() / max_allocation_size; ++i) {
    state_->slots.push_back(data->data() + offset);
    offset += max_allocation_size;
  }
}

SlabAllocator::~SlabAllocator() = default;

size_t SlabAllocator::max_allocation_size() const {
  return state_->max_allocation_size;
}

SlabAllocator::Allocation SlabAllocator::Allocate(size_t size) {
  // TODO(parkers): Review fairness of condition + add sub-allocations for
  // smaller sizes.
  state_->mu.LockWhen(absl::Condition(&State::HasSlots, state_.get()));
  Allocation result;
  result.data = state_->slots.back();
  result.size = std::min(size, state_->max_allocation_size);
  state_->slots.pop_back();
  result.on_done =
      absl::AnyInvocable<void() &&>([state = state_, data = result.data]() {
        absl::MutexLock l(&state->mu);
        state->slots.push_back(data);
      });
  state_->mu.Unlock();
  return result;
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>>
AllocateNetworkPinnedMemory(size_t size) {
  struct pinned_mem_state {
    ~pinned_mem_state() { munmap(buffer, size); }
    absl::Span<uint8_t> data;
    void* buffer;
    size_t size;
  };
  auto out = std::make_shared<pinned_mem_state>();
  int sfd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, 0);
  out->buffer = mmap(nullptr, size, PROT_READ, MAP_SHARED, sfd, 0);
  if (out->buffer == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "tcp-zero-copy-mmap");
  }
  out->size = size;
  close(sfd);
  out->data =
      absl::Span<uint8_t>(reinterpret_cast<uint8_t*>(out->buffer), size);
  return std::shared_ptr<absl::Span<uint8_t>>(out, &out->data);
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> AllocateAlignedMemory(
    size_t size) {
  void* data = nullptr;
  const size_t kCpuPageSize = 4096;
  if (posix_memalign(&data, kCpuPageSize, size) != 0) {
    return absl::InternalError("error in posix_memalign.");
  }
  struct aligned_memory_state {
    ~aligned_memory_state() { free(buffer); }
    absl::Span<uint8_t> data;
    void* buffer;
    size_t size;
  };
  auto owner = std::make_shared<aligned_memory_state>();
  owner->buffer = data;
  owner->size = size;
  owner->data = absl::MakeSpan(reinterpret_cast<uint8_t*>(data), size);
  return std::shared_ptr<absl::Span<uint8_t>>(owner, &owner->data);
}

void PullTable::AwaitPull(uint64_t uuid, tsl::RCReference<Entry> entry) {
  std::vector<PausedFetch> paused_fetches;
  {
    absl::MutexLock l(&mu_);
    auto it = paused_fetches_.find(uuid);
    if (it != paused_fetches_.end()) {
      paused_fetches = std::move(it->second);
      paused_fetches_.erase(it);
    }
    entries_[uuid] = std::move(entry);
  }
  for (auto& v : paused_fetches) {
    Handle(v.state, v.req, v.base_req_id);
  }
}

void PullTable::Handle(tsl::RCReference<ConnectionState> state,
                       const SocketTransferPullRequest& req,
                       size_t base_req_id) {
  tsl::RCReference<Entry> entry;
  {
    absl::MutexLock l(&mu_);
    auto it = entries_.find(req.uuid());
    if (it == entries_.end()) {
      PausedFetch fetch;
      fetch.state = std::move(state);
      fetch.req = req;
      fetch.base_req_id = base_req_id;
      paused_fetches_[req.uuid()].push_back(std::move(fetch));
      return;
    }
    entry = it->second;
  }
  if (entry->Handle(std::move(state), req, base_req_id)) {
    absl::MutexLock l(&mu_);
    auto it = entries_.find(req.uuid());
    entries_.erase(it);
  }
}

class StringVectorPullTableEntry : public PullTable::Entry {
 public:
  explicit StringVectorPullTableEntry(std::vector<std::string> buffers)
      : buffers_(std::move(buffers)) {}

  bool Handle(tsl::RCReference<ConnectionState> state,
              const SocketTransferPullRequest& req,
              size_t base_req_id) override {
    for (uint64_t bid : req.buffer_ids()) {
      auto data_copy = std::make_unique<std::string>(buffers_[bid]);
      auto req_id = base_req_id;
      ++base_req_id;
      auto& data = *data_copy;
      state->Send(req_id, data.data(), 0, data.size(), true,
                  [data = std::move(data_copy)]() {});
    }
    return true;
  }

 private:
  std::vector<std::string> buffers_;
};

tsl::RCReference<PullTable::Entry> PullTable::MakeStringEntry(
    std::vector<std::string> buffers) {
  return tsl::MakeRef<StringVectorPullTableEntry>(std::move(buffers));
}

}  // namespace aux
