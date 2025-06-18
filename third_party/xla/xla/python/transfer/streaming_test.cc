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
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/env.h"

namespace aux {
namespace {

class LocalConnectionState : public ConnectionState {
 public:
  explicit LocalConnectionState(std::shared_ptr<PullTable> table)
      : table_(std::move(table)) {}
  ~LocalConnectionState() override = default;

  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    tsl::RCReference<ChunkDestination> dest;
    {
      absl::MutexLock l(&mu_);
      auto it = dests_.find(req_id);
      CHECK(it != dests_.end());
      if (is_largest) {
        it->second.transferred_size += offset;
      } else {
        it->second.transferred_size -= size;
      }
      if (it->second.transferred_size == 0) {
        dest = std::move(it->second.dest);
        dests_.erase(it);
      } else {
        dest = it->second.dest;
      }
    }
    CHECK_OK(dest->Put(data, offset, size, std::move(on_done)));
  }

  void Pull(uint64_t uuid, int buf_id,
            tsl::RCReference<ChunkDestination> dest) {
    size_t req_id;
    {
      absl::MutexLock l(&mu_);
      dests_[next_req_id_].dest = std::move(dest);
      req_id = next_req_id_;
      ++next_req_id_;
    }
    SocketTransferPullRequest req;
    req.set_uuid(uuid);
    req.add_buffer_ids(buf_id);
    table_->Handle(tsl::FormRef(this), req, req_id);
  }

 private:
  absl::Mutex mu_;
  std::shared_ptr<PullTable> table_;
  uint64_t next_req_id_ = 0;
  struct DestState {
    ssize_t transferred_size = 0;
    tsl::RCReference<ChunkDestination> dest;
  };
  absl::flat_hash_map<uint64_t, DestState> dests_;
};

TEST(BulkTransferInterface, PullTableInterfaces) {
  auto table = std::make_shared<PullTable>();
  std::string msg("secret message");
  uint64_t uuid = 5678;
  table->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));
  auto ref = tsl::MakeRef<LocalConnectionState>(table);
  auto [s, cd] = ChunkDestination::MakeStringDest();
  ref->Pull(uuid, 0, std::move(cd));
  EXPECT_EQ(s.Await().value(), msg);
}

TEST(BulkTransferInterface, AwaitAfterInterfaces) {
  auto table = std::make_shared<PullTable>();
  std::string msg("secret message");
  uint64_t uuid = 5678;
  auto ref = tsl::MakeRef<LocalConnectionState>(table);
  auto [s, cd] = ChunkDestination::MakeStringDest();
  ref->Pull(uuid, 0, std::move(cd));
  table->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));
  EXPECT_EQ(s.Await().value(), msg);
}

TEST(BulkTransferInterface, LocalTransport) {
  auto transport_factory = BulkTransportFactory::CreateLocal();
  auto a = transport_factory->InitBulkTransport();
  auto b = transport_factory->RecvBulkTransport(a.request);
  std::move(a.start_bulk_transport)(b.request);

  std::string test_message = "secret message";

  absl::Notification is_sent;
  int assigned_bond_id;
  size_t send_size;
  a.bulk_transport->Send(BulkTransportInterface::MakeMessage(
      test_message, [&](int bond_id, size_t size) {
        assigned_bond_id = bond_id;
        send_size = size;
        is_sent.Notify();
      }));

  is_sent.WaitForNotification();
  absl::Notification recv_done;
  absl::StatusOr<BulkTransportInterface::Message> recv_message;
  b.bulk_transport->Recv(
      send_size, assigned_bond_id,
      [&](absl::StatusOr<BulkTransportInterface::Message> message) {
        recv_message = std::move(message);
        recv_done.Notify();
      });
  recv_done.WaitForNotification();
  ASSERT_TRUE(recv_message.status().ok()) << recv_message.status();
  EXPECT_EQ(test_message,
            std::string(reinterpret_cast<const char*>(recv_message->data),
                        recv_message->size));
}

TEST(BulkTransferInterface, ClosedLocalTransport) {
  auto transport_factory = BulkTransportFactory::CreateLocal();
  auto a = transport_factory->InitBulkTransport();
  auto b = transport_factory->RecvBulkTransport(a.request);
  std::move(a.start_bulk_transport)(b.request);
  a.bulk_transport = nullptr;
  absl::Notification recv_done;
  absl::StatusOr<BulkTransportInterface::Message> recv_message;
  b.bulk_transport->Recv(
      80, 0, [&](absl::StatusOr<BulkTransportInterface::Message> message) {
        recv_message = std::move(message);
        recv_done.Notify();
      });
  recv_done.WaitForNotification();
  ASSERT_FALSE(recv_message.ok());
}

absl::Status PutSync(ChunkDestination* dest, size_t offset, std::string chunk) {
  absl::Notification on_done;
  auto s = dest->Put(chunk.data(), offset, chunk.size(),
                     [&on_done]() { on_done.Notify(); });
  if (!s.ok()) {
    return s;
  }
  on_done.WaitForNotification();
  return absl::OkStatus();
}

TEST(ChunkDestination, StringChunkDest) {
  auto [result, dest] = ChunkDestination::MakeStringDest();
  auto s = PutSync(dest.get(), 7, "message");
  ASSERT_TRUE(s.ok()) << s;
  s = PutSync(dest.get(), 0, "secret ");
  ASSERT_TRUE(s.ok()) << s;
  dest = {};

  auto value = result.Await();
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_EQ(*value, "secret message");
}

TEST(ChunkDestination, StringChunkDestWithGaps) {
  auto [result, dest] = ChunkDestination::MakeStringDest();
  auto s = PutSync(dest.get(), 8, "message");
  ASSERT_TRUE(s.ok()) << s;
  s = PutSync(dest.get(), 0, "secret ");
  ASSERT_TRUE(s.ok()) << s;
  dest = {};

  auto value = result.Await();
  ASSERT_FALSE(value.ok());
}

TEST(SlabAllocator, BasicSubAllocations) {
  auto alloc1_or = AllocateNetworkPinnedMemory(4096 * 4);
  auto alloc2_or = AllocateAlignedMemory(4096 * 4);
  ASSERT_TRUE(alloc1_or.ok()) << alloc1_or.status();
  ASSERT_TRUE(alloc2_or.ok()) << alloc2_or.status();
  for (auto alloc : {*alloc1_or, *alloc2_or}) {
    absl::Mutex mu;
    std::deque<SlabAllocator::Allocation> allocs;
    auto thread = std::unique_ptr<tsl::Thread>(
        tsl::Env::Default()->StartThread({}, "test-thread", [&] {
          for (size_t i = 0; i < 200; ++i) {
            absl::MutexLock l(&mu);
            auto cond = [&]() {
              return allocs.size() >= std::min(static_cast<size_t>(200 - i),
                                               static_cast<size_t>(4));
            };
            mu.Await(absl::Condition(&cond));
            std::move(allocs.front().on_done)();
            allocs.pop_front();
          }
        }));
    SlabAllocator allocator(alloc, 4096);
    for (size_t i = 0; i < 200; ++i) {
      auto alloc = allocator.Allocate(allocator.max_allocation_size());
      absl::MutexLock l(&mu);
      allocs.push_back(std::move(alloc));
    }
  }
}

TEST(InvalidAllocator, InvalidPinnedAlloc) {
  auto alloc1_or = AllocateNetworkPinnedMemory(1l << 49);
  ASSERT_FALSE(alloc1_or.ok());
}

TEST(InvalidAllocator, InvalidAlignedAlloc) {
  auto alloc2_or = AllocateAlignedMemory(1l << 49);
  ASSERT_FALSE(alloc2_or.ok());
}

}  // namespace
}  // namespace aux
