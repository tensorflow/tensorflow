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
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux {

bool PullTable::CompareNodes(const HeapNodeBase* a, const HeapNodeBase* b) {
  return a->timeout.value() < b->timeout.value();
}

void PullTable::BubbleUp(std::vector<HeapNodeBase*>& heap, size_t idx) {
  while (idx > 0) {
    size_t parent = (idx - 1) / 2;
    if (CompareNodes(heap[idx], heap[parent])) {
      std::swap(heap[idx], heap[parent]);
      heap[idx]->heap_index = idx;
      heap[parent]->heap_index = parent;
      idx = parent;
    } else {
      break;
    }
  }
}

void PullTable::BubbleDown(std::vector<HeapNodeBase*>& heap, size_t idx) {
  size_t size = heap.size();
  while (2 * idx + 1 < size) {
    size_t left = 2 * idx + 1;
    size_t right = left + 1;
    size_t smallest = left;
    if (right < size && CompareNodes(heap[right], heap[left])) {
      smallest = right;
    }
    if (CompareNodes(heap[smallest], heap[idx])) {
      std::swap(heap[idx], heap[smallest]);
      heap[idx]->heap_index = idx;
      heap[smallest]->heap_index = smallest;
      idx = smallest;
    } else {
      break;
    }
  }
}

void PullTable::PushIntrusive(std::vector<HeapNodeBase*>& heap,
                              HeapNodeBase* item) {
  item->heap_index = heap.size();
  heap.push_back(item);
  BubbleUp(heap, heap.size() - 1);
}

void PullTable::RemoveIntrusive(std::vector<HeapNodeBase*>& heap,
                                HeapNodeBase* item) {
  size_t idx = item->heap_index;
  if (idx == static_cast<size_t>(-1)) {
    return;
  }
  if (idx == heap.size() - 1) {
    heap.pop_back();
    item->heap_index = -1;
    return;
  }
  auto* back = heap.back();
  heap[idx] = back;
  back->heap_index = idx;
  heap.pop_back();
  item->heap_index = -1;

  if (idx > 0) {
    size_t parent = (idx - 1) / 2;
    if (CompareNodes(heap[idx], heap[parent])) {
      BubbleUp(heap, idx);
      return;
    }
  }
  BubbleDown(heap, idx);
}

PullTable::HeapNodeBase* PullTable::PopIntrusive(
    std::vector<HeapNodeBase*>& heap) {
  if (heap.empty()) {
    return nullptr;
  }
  auto* top = heap.front();
  RemoveIntrusive(heap, top);
  return top;
}

class StringFutureChunkDestination : public aux::ChunkDestination {
 public:
  explicit StringFutureChunkDestination(tsl::Promise<std::string> dest)
      : dest_(std::move(dest)) {}
  ~StringFutureChunkDestination() override { dest_.Set(ConsumeFinalResult()); }
  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    {
      absl::MutexLock l(mu_);
      chunks_.emplace_back(
          offset, std::string(reinterpret_cast<const char*>(data), size));
    }
    std::move(on_done)();
    return absl::OkStatus();
  }

  void Poison(absl::Status s) override { CHECK_OK(s); }

  absl::StatusOr<std::string> ConsumeFinalResult() {
    absl::MutexLock l(mu_);
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
  tsl::Promise<std::string> dest_;
};

std::pair<tsl::Future<std::string>, tsl::RCReference<ChunkDestination>>
ChunkDestination::MakeStringDest() {
  auto [promise, result] = tsl::MakePromise<std::string>();
  return std::make_pair(
      std::move(result),
      tsl::MakeRef<StringFutureChunkDestination>(std::move(promise)));
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
    absl::MutexLock l(send_q_->mu);
    send_q_->status = absl::InternalError("Connection failure for local bond.");
  }
  void Send(SendMessage msg) override {
    absl::MutexLock l(send_q_->mu);
    std::move(msg.on_send)(0, msg.size);
    send_q_->buffers.push_back(std::move(msg));
  }
  void Recv(size_t size, int bond_id,
            absl::AnyInvocable<void(absl::StatusOr<Message> msg) &&> on_recv)
      override {
    absl::StatusOr<Message> result;
    {
      absl::MutexLock l(recv_q_->mu);
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
  result.on_send = [on_send = std::move(on_send)](absl::StatusOr<int> bond_id,
                                                  size_t size) mutable {
    std::move(on_send)(bond_id.value(), size);
  };
  result.on_done = [tmp = std::move(tmp)]() {};
  return result;
}

class LocalBulkTransportFactory : public BulkTransportFactory {
 public:
  BulkTransportInitResult InitBulkTransport() override {
    BulkTransportInitResult out;
    absl::MutexLock l(mu_);
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
    absl::MutexLock l(mu_);
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
        absl::MutexLock l(state->mu);
        state->slots.push_back(data);
      });
  state_->mu.unlock();
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

void PullTable::AwaitPull(uint64_t uuid, tsl::RCReference<Entry> entry,
                          std::optional<absl::Time> timeout) {
  FetchList local_fetches;
  {
    absl::MutexLock l(&mu_);
    auto it = paused_fetches_.find(uuid);
    if (it != paused_fetches_.end()) {
      local_fetches.splice(local_fetches.end(), it->second);
      for (auto& v : local_fetches) {
        if (v.timeout.has_value()) {
          RemoveIntrusive(fetch_heap_, &v);
        }
      }
      paused_fetches_.erase(it);
    }

    auto it_entry = entries_.find(uuid);
    if (it_entry != entries_.end()) {
      if (it_entry->second->timeout.has_value()) {
        RemoveIntrusive(entry_heap_, it_entry->second.get());
      }
    }

    auto new_entry = std::make_unique<EntryWithTimeout>();
    new_entry->entry = std::move(entry);
    new_entry->timeout = timeout;
    new_entry->uuid = uuid;
    if (timeout.has_value()) {
      PushIntrusive(entry_heap_, new_entry.get());
    }
    entries_[uuid] = std::move(new_entry);
  }
  for (auto& v : local_fetches) {
    Handle(v.state, v.req, v.base_req_id, v.timeout);
  }
}

void PullTable::Handle(tsl::RCReference<ConnectionState> state,
                       const SocketTransferPullRequest& req, size_t base_req_id,
                       std::optional<absl::Time> timeout) {
  tsl::RCReference<Entry> entry;
  EntryWithTimeout* entry_ptr = nullptr;
  {
    absl::MutexLock l(&mu_);
    auto it = entries_.find(req.uuid());
    if (it == entries_.end()) {
      auto& list = paused_fetches_[req.uuid()];
      list.emplace_back();
      PausedFetch& fetch_ref = list.back();
      fetch_ref.state = std::move(state);
      fetch_ref.req = req;
      fetch_ref.base_req_id = base_req_id;
      fetch_ref.timeout = timeout;
      fetch_ref.list_it = std::prev(list.end());
      if (timeout.has_value()) {
        PushIntrusive(fetch_heap_, &fetch_ref);
      }
      return;
    }
    entry_ptr = it->second.get();
    entry = entry_ptr->entry;
  }
  if (entry->Handle(std::move(state), req, base_req_id)) {
    absl::MutexLock l(&mu_);
    auto it = entries_.find(req.uuid());
    if (it != entries_.end() && it->second->entry == entry) {
      if (it->second->timeout.has_value()) {
        RemoveIntrusive(entry_heap_, it->second.get());
      }
      entries_.erase(it);
    }
  }
}

void PullTable::DropExpiredPulls(absl::Time t) {
  std::vector<std::unique_ptr<EntryWithTimeout>> expired_entries;
  FetchList expired_fetches;
  {
    absl::MutexLock l(&mu_);
    while (!entry_heap_.empty()) {
      auto* top = entry_heap_.front();
      if (top->timeout.value() < t) {
        PopIntrusive(entry_heap_);
        auto* entry_ptr = static_cast<EntryWithTimeout*>(top);
        auto it = entries_.find(entry_ptr->uuid);
        if (it != entries_.end() && it->second.get() == entry_ptr) {
          expired_entries.push_back(std::move(it->second));
          entries_.erase(it);
        }
      } else {
        break;
      }
    }

    while (!fetch_heap_.empty()) {
      auto* top = fetch_heap_.front();
      if (top->timeout.value() < t) {
        PopIntrusive(fetch_heap_);
        auto* fetch_ptr = static_cast<PausedFetch*>(top);
        auto it = paused_fetches_.find(fetch_ptr->req.uuid());
        if (it != paused_fetches_.end()) {
          expired_fetches.splice(expired_fetches.end(), it->second,
                                 fetch_ptr->list_it);
          if (it->second.empty()) {
            paused_fetches_.erase(it);
          }
        }
      } else {
        break;
      }
    }
  }

  for (const auto& fetch : expired_fetches) {
    size_t req_id = fetch.base_req_id;
    for (uint64_t bid : fetch.req.buffer_ids()) {
      (void)bid;
      fetch.state->SendError(req_id, 0, 0, true,
                             absl::DeadlineExceededError("Pull expired"));
      ++req_id;
    }
  }
}

void PullTable::Reset() {
  mu_.lock();
  auto entries = std::move(entries_);
  auto paused_fetches_by_uuid = std::move(paused_fetches_);
  entry_heap_.clear();
  fetch_heap_.clear();
  mu_.unlock();
  // Drop entries without the lock held.
  std::vector<std::pair<uint64_t, FetchList>> sorted_fetches(
      paused_fetches_by_uuid.begin(), paused_fetches_by_uuid.end());
  std::sort(sorted_fetches.begin(), sorted_fetches.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& pair : sorted_fetches) {
    for (const auto& paused_fetch : pair.second) {
      size_t req_id = paused_fetch.base_req_id;
      for (uint64_t bid : paused_fetch.req.buffer_ids()) {
        (void)bid;
        paused_fetch.state->SendError(req_id, 0, 0, true,
                                      absl::InternalError("PullTable::Reset"));
        ++req_id;
      }
    }
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
