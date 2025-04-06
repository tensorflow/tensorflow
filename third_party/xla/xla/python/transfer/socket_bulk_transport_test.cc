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
#include "xla/python/transfer/socket_bulk_transport.h"

#include <stdint.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"

namespace aux {

namespace {

absl::Status SetupSocketPairUsingEventLoop(int& send_fd, int& recv_fd) {
  absl::Notification recv_setup;
  auto listener = SocketListener::Listen(
      SocketAddress(), [&](int sockfd, const SocketAddress& addr) {
        recv_fd = sockfd;
        recv_setup.Notify();
      });
  if (!listener.ok()) {
    return listener.status();
  }
  (*listener)->Start();

  auto other_addr = (*listener)->addr();
  send_fd =
      socket(other_addr.address().sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
  connect(send_fd, reinterpret_cast<const struct sockaddr*>(&other_addr),
          sizeof(other_addr));
  int value = 1;
  CHECK_GE(setsockopt(send_fd, SOL_SOCKET, SO_ZEROCOPY, &value, sizeof(value)),
           0)
      << strerror(errno) << " " << errno;

  recv_setup.WaitForNotification();
  return absl::OkStatus();
}

TEST(SendQueue, TestZeroCopyQueueCleanRemoteShutdown) {
  int send_fd, recv_fd;
  auto status = SetupSocketPairUsingEventLoop(send_fd, recv_fd);
  ASSERT_TRUE(status.ok()) << status;

  auto work_queue = SharedSendWorkQueue::Start();
  auto msg_queue = std::make_shared<SharedSendMsgQueue>();

  SharedSendMsgQueue::StartSubConnectionSender(send_fd, 0, msg_queue,
                                               work_queue);

  std::string txt_msg("hello world");
  absl::Notification notify;
  BulkTransportInterface::SendMessage msg;
  msg.data = txt_msg.data();
  msg.size = txt_msg.size();
  msg.on_send = [](int id, size_t size) {};
  msg.on_done = [&notify]() { notify.Notify(); };
  msg_queue->ScheduleSendWork(std::move(msg));
  notify.WaitForNotification();
  close(recv_fd);
  msg_queue->NoMoreMessages();
}

TEST(SendQueue, SendAndRecvQueuesArtificialLimit) {
  size_t packet_size = 1024 * 8;
  SlabAllocator allocator(AllocateNetworkPinnedMemory(packet_size * 4).value(),
                          packet_size);
  SlabAllocator uallocator(AllocateAlignedMemory(packet_size * 4).value(),
                           packet_size);
  auto recv_thread = RecvThreadState::Create(allocator, uallocator);

  int send_fd, recv_fd;
  auto status = SetupSocketPairUsingEventLoop(send_fd, recv_fd);
  ASSERT_TRUE(status.ok()) << status;

  auto work_queue = SharedSendWorkQueue::Start();
  auto msg_queue = std::make_shared<SharedSendMsgQueue>();

  SharedSendMsgQueue::StartSubConnectionSender(send_fd, 0, msg_queue,
                                               work_queue, 64);

  std::string txt_msg;

  while (txt_msg.size() < packet_size) {
    txt_msg += "hello world";
  }
  absl::Mutex mu;
  size_t send_count = 10;

  for (size_t i = 0; i < 10; ++i) {
    txt_msg.resize(packet_size);
    BulkTransportInterface::SendMessage msg;
    msg.data = txt_msg.data();
    msg.size = txt_msg.size();
    msg.on_send = [](int id, size_t size) {};
    msg.on_done = [&mu, &send_count]() {
      absl::MutexLock l(&mu);
      --send_count;
    };
    msg_queue->ScheduleSendWork(std::move(msg));
  }

  for (size_t i = 0; i < 10; ++i) {
    absl::Notification recv_notify;
    absl::StatusOr<aux::BulkTransportInterface::Message> recv_msg;
    recv_thread->ScheduleRecvWork(
        packet_size, recv_fd,
        [&](absl::StatusOr<aux::BulkTransportInterface::Message> msg) {
          recv_msg = std::move(msg);
          recv_notify.Notify();
        });
    recv_notify.WaitForNotification();
    ASSERT_TRUE(recv_msg.ok()) << recv_msg.status();

    EXPECT_EQ(txt_msg,
              absl::string_view(reinterpret_cast<const char*>(recv_msg->data),
                                recv_msg->size));
    std::move(recv_msg->on_done)();
  }
  {
    absl::MutexLock l(&mu);
    auto cond = [&]() { return send_count == 0; };
    mu.Await(absl::Condition(&cond));
  }
}

TEST(SocketBulkTransportFactoryTest, SendAndRecvWithFactory) {
  size_t packet_size = 1024 * 8;
  SlabAllocator allocator(AllocateNetworkPinnedMemory(packet_size * 4).value(),
                          packet_size);
  SlabAllocator uallocator(AllocateAlignedMemory(packet_size * 4).value(),
                           packet_size);

  SocketAddress addr;
  SocketAddress addrv4 = SocketAddress::Parse("0.0.0.0:0").value();
  auto status_or =
      CreateSocketBulkTransportFactory({addr, addrv4}, allocator, uallocator);
  ASSERT_TRUE(status_or.ok()) << status_or.status();
  auto factory = status_or.value();
  status_or =
      CreateSocketBulkTransportFactory({addr, addrv4}, allocator, uallocator);
  ASSERT_TRUE(status_or.ok()) << status_or.status();
  auto factory2 = status_or.value();

  std::unique_ptr<BulkTransportInterface> bulk_transporta;
  std::unique_ptr<BulkTransportInterface> bulk_transportb;
  for (size_t i = 0; i < 4; ++i) {
    auto init_res = factory->InitBulkTransport();
    bulk_transporta = std::move(init_res.bulk_transport);
    auto recv_res = factory2->RecvBulkTransport(init_res.request);
    bulk_transportb = std::move(recv_res.bulk_transport);
    std::move(init_res.start_bulk_transport)(recv_res.request);
  }

  packet_size = 64;

  std::vector<std::string> txt_msgs;
  int num_messages = 10;
  for (size_t i = 0; i < num_messages; ++i) {
    std::string txt_msg;
    while (txt_msg.size() < packet_size) {
      absl::StrAppend(&txt_msg, "hello world: ", i);
    }
    txt_msg.resize(packet_size);
    txt_msgs.push_back(std::move(txt_msg));
  }
  absl::Mutex mu;
  size_t send_count = 10;
  std::deque<std::pair<int, int>> send_queue;

  for (size_t i = 0; i < num_messages; ++i) {
    BulkTransportInterface::SendMessage msg;
    msg.data = txt_msgs[i].data();
    msg.size = txt_msgs[i].size();
    msg.on_send = [&, i](int id, size_t size) {
      absl::MutexLock l(&mu);
      send_queue.push_back({i, id});
    };
    msg.on_done = [&mu, &send_count]() {
      absl::MutexLock l(&mu);
      --send_count;
    };
    bulk_transporta->Send(std::move(msg));
  }

  for (size_t i = 0; i < num_messages; ++i) {
    absl::Notification recv_notify;
    absl::StatusOr<aux::BulkTransportInterface::Message> recv_msg;
    int bond_id = -1;
    int msg_id = -1;
    {
      absl::MutexLock l(&mu);
      auto cond = [&]() { return !send_queue.empty(); };
      mu.Await(absl::Condition(&cond));
      std::tie(msg_id, bond_id) = send_queue.front();
      send_queue.pop_front();
    }
    bulk_transportb->Recv(
        packet_size, bond_id,
        [&](absl::StatusOr<aux::BulkTransportInterface::Message> msg) {
          recv_msg = std::move(msg);
          recv_notify.Notify();
        });
    recv_notify.WaitForNotification();
    ASSERT_TRUE(recv_msg.ok()) << recv_msg.status();

    EXPECT_EQ(txt_msgs[msg_id],
              absl::string_view(reinterpret_cast<const char*>(recv_msg->data),
                                recv_msg->size));
    std::move(recv_msg->on_done)();
  }
  {
    absl::MutexLock l(&mu);
    auto cond = [&]() { return send_count == 0; };
    mu.Await(absl::Condition(&cond));
  }
}

void HandleAckAndExpectDone(ZeroCopySendAckTable& table, uint32_t ack_id,
                            size_t exp_seal_id, std::vector<size_t>& ack_list) {
  EXPECT_EQ(ack_list.size(), 0);
  table.HandleAck(ack_id);
  EXPECT_EQ(ack_list.size(), 1);
  if (ack_list.size() == 1) {
    EXPECT_EQ(ack_list[0], exp_seal_id);
    ack_list.clear();
  }
}

size_t SealNextMessage(ZeroCopySendAckTable& table, size_t& next_seal_id,
                       std::vector<size_t>& ack_list) {
  size_t seal_id = next_seal_id;
  ++next_seal_id;
  table.Seal([seal_id, &ack_list]() { ack_list.push_back(seal_id); });
  return seal_id;
}

TEST(AckTableTest, Basic) {
  ZeroCopySendAckTable table;
  std::vector<size_t> ack_list;
  size_t next_seal_id = 0;
  uint32_t n = -10;
  table.PretendCloseToRolloverForTests(n);
  for (size_t i = 0; i < 10; ++i) {
    table.Send();
    table.HandleAck(n + 1);
    table.Send();
    size_t a = SealNextMessage(table, next_seal_id, ack_list);
    size_t b = SealNextMessage(table, next_seal_id, ack_list);
    size_t c = SealNextMessage(table, next_seal_id, ack_list);
    HandleAckAndExpectDone(table, n + 3, b, ack_list);
    size_t d = SealNextMessage(table, next_seal_id, ack_list);
    table.HandleAck(n + 2);
    HandleAckAndExpectDone(table, n + 5, d, ack_list);
    HandleAckAndExpectDone(table, n + 4, c, ack_list);
    HandleAckAndExpectDone(table, n + 0, a, ack_list);
    n += 6;
  }
  auto [ids_count, cbs_count] = table.GetTableSizes();
  EXPECT_EQ(ids_count, 0);
  EXPECT_EQ(cbs_count, 0);
}

}  // namespace
}  // namespace aux
