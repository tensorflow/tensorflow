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

#include <linux/errqueue.h>
#include <linux/tcp.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "xla/tsl/platform/env.h"

namespace aux {

ZeroCopySendAckTable::ZeroCopySendAckTable() {
  acks_.emplace_back();
  ack_ids_.push_back(acks_start_ + static_cast<uint32_t>(acks_.size() - 1));
}

void ZeroCopySendAckTable::Send() {
  absl::MutexLock l(&mu_);
  ++n_acks_in_batch_;
  ack_ids_.push_back(acks_start_ + static_cast<uint32_t>(acks_.size() - 1));
}

uint32_t ZeroCopySendAckTable::Seal(absl::AnyInvocable<void() &&> on_done) {
  uint32_t ack_id;
  {
    absl::MutexLock l(&mu_);
    ++n_acks_in_batch_;
    auto& ack = acks_.back();
    ack_id = static_cast<uint32_t>(ack_ids_.size() - 1) + ack_ids_start_;
    ack.acks_count += 1 + n_acks_in_batch_;
    if (ack.acks_count != 0) {
      ack.on_done = std::move(on_done);
      on_done = nullptr;
    }
    n_acks_in_batch_ = 0;
    acks_.emplace_back();
    ack_ids_.push_back(acks_start_ + static_cast<uint32_t>(acks_.size() - 1));
    GCTables();
  }
  if (on_done) {
    std::move(on_done)();
  }
  return ack_id;
}

void ZeroCopySendAckTable::HandleAck(uint32_t v) {
  absl::AnyInvocable<void() &&> on_done;
  {
    absl::MutexLock l(&mu_);
    v -= ack_ids_start_;
    auto& ack_id = ack_ids_[v];
    auto& ack = acks_[*ack_id - acks_start_];
    ack_id = std::nullopt;
    DCHECK_NE(ack.acks_count, 0);
    --ack.acks_count;
    if (ack.acks_count == 0) {
      on_done = std::move(ack.on_done);
      ack.on_done = nullptr;
    }
    GCTables();
  }
  if (on_done) {
    std::move(on_done)();
  }
}

void ZeroCopySendAckTable::PretendCloseToRolloverForTests(uint32_t bump) {
  absl::MutexLock l(&mu_);
  for (auto& ack : ack_ids_) {
    if (ack.has_value()) {
      *ack += bump;
    }
  }
  ack_ids_start_ += bump;
  acks_start_ += bump;
}

std::pair<size_t, size_t> ZeroCopySendAckTable::GetTableSizes() {
  absl::MutexLock l(&mu_);
  // Always 1 ahead.
  return std::make_pair(acks_.size() - 1, ack_ids_.size() - 1);
}

void ZeroCopySendAckTable::GCTables() {
  while (!ack_ids_.empty() && !ack_ids_.front().has_value()) {
    ack_ids_.pop_front();
    ++ack_ids_start_;
  }
  while (!acks_.empty() && acks_.front().acks_count == 0) {
    acks_.pop_front();
    ++acks_start_;
  }
}

absl::Status ZeroCopySendAckTable::HandleSocketErrors(int fd) {
  char control[64];
  struct msghdr msg = {};
  msg.msg_control = control;
  msg.msg_controllen = sizeof control;
  int res = recvmsg(fd, &msg, MSG_ERRQUEUE);
  if (res == -1) {
    return absl::ErrnoToStatus(errno, "Unknown error while handling acks.");
  }
  for (auto* cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_IP && cmsg->cmsg_type == IP_RECVERR) {
    } else if (cmsg->cmsg_level == SOL_IPV6 &&
               cmsg->cmsg_type == IPV6_RECVERR) {
    } else {
      return absl::InternalError(
          absl::StrCat("Unknown cmsg level: ", cmsg->cmsg_level,
                       " type: ", cmsg->cmsg_type));
    }
    auto* err = reinterpret_cast<struct sock_extended_err*>(CMSG_DATA(cmsg));
    if (err->ee_origin != SO_EE_ORIGIN_ZEROCOPY) {
      return absl::InternalError(
          absl::StrCat("Unknown cmsg origin: ", err->ee_origin));
    }
    for (uint32_t i = err->ee_info; i != err->ee_data + 1; ++i) {
      HandleAck(i);
    }
  }
  return absl::OkStatus();
}

void ZeroCopySendAckTable::ClearAll() {
  absl::MutexLock l(&mu_);
  for (auto& ack : acks_) {
    if (ack.on_done) {
      std::move(ack.on_done)();
      ack.on_done = nullptr;
    }
  }
}

class SendConnectionHandler : public PollEventLoop::Handler {
 public:
  SendConnectionHandler(int fd, int bond_id,
                        std::shared_ptr<SharedSendMsgQueue> msg_queue,
                        std::shared_ptr<SharedSendWorkQueue> work_queue,
                        size_t artificial_send_limit)
      : fd_(fd),
        bond_id_(bond_id),
        msg_queue_(std::move(msg_queue)),
        work_queue_(std::move(work_queue)),
        artificial_send_limit_(artificial_send_limit) {}

  ~SendConnectionHandler() override {
#ifdef MSG_ZEROCOPY
    table_.ClearAll();
#endif
    close(fd_);
  }

  void ScheduleSendWork(aux::BulkTransportInterface::SendMessage msg) {
    work_queue_->ScheduleWork(this, std::move(msg));
  }

  void DoSend(aux::BulkTransportInterface::SendMessage msg) {
    std::move(msg.on_send)(bond_id_, msg.size);
    msg.on_send = nullptr;
#ifndef MSG_ZEROCOPY
    size_t offset = 0;
    while (offset < msg.size) {
      ssize_t send_count =
          send(fd_, reinterpret_cast<char*>(msg.data) + offset,
               std::min(msg.size - offset, artificial_send_limit_), 0);
      if (send_count <= 0) {
        break;
      }
      offset += send_count;
      if (offset == msg.size) {
        PollEventLoop::GetDefault()->Schedule(std::move(msg.on_done));
        return ReturnToNotReadyFromSending();
      }
    }
    PollEventLoop::GetDefault()->Schedule(std::move(msg.on_done));
    no_more_messages_.store(true);
#else
    size_t offset = 0;
    while (offset < msg.size) {
      ssize_t send_count = send(
          fd_, reinterpret_cast<char*>(msg.data) + offset,
          std::min(msg.size - offset, artificial_send_limit_), MSG_ZEROCOPY);
      if (send_count <= 0) {
        break;
      }
      offset += send_count;
      if (offset == msg.size) {
        table_.Seal(std::move(msg.on_done));
        return ReturnToNotReadyFromSending();
      }
      table_.Send();
    }
    auto ack_id = table_.Seal(std::move(msg.on_done));
    no_more_messages_.store(true);
    table_.HandleAck(ack_id);
#endif
    ReturnToNotReadyFromSending();
  }

  void ReturnToNotReadyFromSending() {
    auto* l = loop();
    SocketState expected = SocketState::kSending;
    if (state_.compare_exchange_strong(expected, SocketState::kNotReady)) {
      l->SendWake(this);
      return;
    }
    if (expected == SocketState::kError) {
      // The sender expects us to wake.
      delete this;
      return;
    }
  }

  void NoMoreMessages() {
    no_more_messages_.store(true);
    ReturnToNotReadyFromSending();
  }

  void PopulatePollInfo(pollfd& events) override {
    events.fd = fd_;
    events.events = POLLERR | POLLRDHUP;
    if (state_.load() == SocketState::kNotReady) {
      events.events |= POLLOUT;
    }
  }

  bool HandleEvents(const pollfd& events) override {
    if (events.revents & POLLRDHUP) {
      HandleRdHup();
      return false;
    } else if (events.revents & POLLERR) {
#ifdef MSG_ZEROCOPY
      CHECK_OK(table_.HandleSocketErrors(fd_));
#endif
    } else if (events.revents & POLLOUT) {
      if (no_more_messages_.load() == true) {
        delete this;
        return false;
      } else {
        state_.store(SocketState::kSending);
        msg_queue_->ReportReadyToSend(this);
      }
    }
    return true;
  }

  void HandleRdHup() {
    while (true) {
      auto state = state_.load();
      if (state == SocketState::kNotReady) {
        if (state_.compare_exchange_strong(state, SocketState::kError)) {
          // No sender thread, so we delete.
          delete this;
          return;
        }
      } else if (state == SocketState::kSending) {
        if (state_.compare_exchange_strong(state, SocketState::kError)) {
          // The sender thread will delete.
          return;
        }
      } else {
        LOG(FATAL) << "Cannot already be in error.";
      }
    }
  }

 private:
  enum class SocketState : uint32_t {
    kNotReady,
    kSending,
    kError,
  };
  std::atomic<SocketState> state_{SocketState::kNotReady};
  std::atomic<bool> no_more_messages_{false};
  std::atomic<bool> no_more_events_{false};
#ifdef MSG_ZEROCOPY
  ZeroCopySendAckTable table_;
#endif
  int fd_;
  int bond_id_;
  std::shared_ptr<SharedSendMsgQueue> msg_queue_;
  std::shared_ptr<SharedSendWorkQueue> work_queue_;
  size_t artificial_send_limit_;
};

void SharedSendWorkQueue::ScheduleWork(
    SendConnectionHandler* handler,
    aux::BulkTransportInterface::SendMessage msg) {
  absl::MutexLock l(&mu_);
  work_items_.push_back({handler, std::move(msg)});
}

void SharedSendWorkQueue::Run() {
  while (true) {
    auto cond = [this]() { return !work_items_.empty() || shutdown_; };
    mu_.LockWhen(absl::Condition(&cond));
    if (work_items_.empty() && shutdown_) {
      mu_.Unlock();
      break;
    }
    auto work = std::move(work_items_.front());
    work_items_.pop_front();
    mu_.Unlock();
    work.handler->DoSend(std::move(work.msg));
  }
  aux::PollEventLoop::GetDefault()->Schedule(
      [thread = std::move(thread_)]() {});
  delete this;
}

std::shared_ptr<SharedSendWorkQueue> SharedSendWorkQueue::Start() {
  auto result = std::shared_ptr<SharedSendWorkQueue>(
      new SharedSendWorkQueue(), [](SharedSendWorkQueue* result) {
        absl::MutexLock l(&result->mu_);
        result->shutdown_ = true;
      });
  result->thread_ =
      std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
          {}, "send-thread", [s = result.get()]() { s->Run(); }));
  return result;
}

void SharedSendMsgQueue::ReportReadyToSend(SendConnectionHandler* handler) {
  mu_.Lock();
  if (!work_items_.empty()) {
    auto msg = std::move(work_items_.front());
    work_items_.pop_front();
    mu_.Unlock();
    handler->ScheduleSendWork(std::move(msg));
  } else if (shutdown_) {
    mu_.Unlock();
    handler->NoMoreMessages();
  } else {
    handlers_.push_back(handler);
    mu_.Unlock();
  }
}

void SharedSendMsgQueue::ScheduleSendWork(
    aux::BulkTransportInterface::SendMessage msg) {
  mu_.Lock();
  DCHECK(!shutdown_);
  if (work_items_.empty() && !handlers_.empty()) {
    auto* handler = handlers_.front();
    handlers_.pop_front();
    mu_.Unlock();
    handler->ScheduleSendWork(std::move(msg));
  } else {
    work_items_.push_back(std::move(msg));
    mu_.Unlock();
  }
}

void SharedSendMsgQueue::NoMoreMessages() {
  std::deque<SendConnectionHandler*> handlers;
  {
    absl::MutexLock l(&mu_);
    shutdown_ = true;
    if (work_items_.empty()) {
      std::swap(handlers_, handlers);
    }
  }
  for (auto* handler : handlers) {
    handler->NoMoreMessages();
  }
}

void SharedSendMsgQueue::StartSubConnectionSender(
    int fd, int bond_id, std::shared_ptr<SharedSendMsgQueue> msg_queue,
    std::shared_ptr<SharedSendWorkQueue> work_queue,
    size_t artificial_send_limit) {
  auto* handler =
      new SendConnectionHandler(fd, bond_id, std::move(msg_queue),
                                std::move(work_queue), artificial_send_limit);
  handler->Register();
}

RecvThreadState::RecvThreadState(std::optional<SlabAllocator> allocator,
                                 SlabAllocator uallocator)
    : allocator_(std::move(allocator)), uallocator_(std::move(uallocator)) {}

void RecvThreadState::DoRecvWork() {
  size_t i = 0;
  size_t zc_send_count = 0;
  size_t non_zc_send_count = 0;
  while (true) {
    auto cond = [this]() {
      return !recv_work_items_.empty() || recv_shutdown_;
    };
    recv_mu_.LockWhen(absl::Condition(&cond));
    if (recv_work_items_.empty() && recv_shutdown_) {
      recv_mu_.Unlock();
      break;
    }
    auto work = std::move(recv_work_items_.front());
    recv_work_items_.pop_front();
    recv_mu_.Unlock();
    auto status = HandleRecvItem(work, zc_send_count, non_zc_send_count);
    if (!status.ok()) {
      std::move(work.on_recv)(status);
    }
    i += 1;
  }
  aux::PollEventLoop::GetDefault()->Schedule(
      [thread = std::move(recv_thread_)]() {});
  delete this;
}

void RecvThreadState::ScheduleRecvWork(
    size_t recv_size, int fd,
    absl::AnyInvocable<
        void(absl::StatusOr<aux::BulkTransportInterface::Message> msg) &&>
        on_recv) {
  absl::MutexLock l(&recv_mu_);
  recv_work_item work;
  work.recv_size = recv_size;
  work.fd = fd;
  work.on_recv = std::move(on_recv);
  recv_work_items_.push_back(std::move(work));
}

std::shared_ptr<RecvThreadState> RecvThreadState::Create(
    std::optional<SlabAllocator> allocator, SlabAllocator uallocator) {
  auto result = std::shared_ptr<RecvThreadState>(
      new RecvThreadState(allocator, uallocator), [](RecvThreadState* result) {
        {
          absl::MutexLock l(&result->recv_mu_);
          result->recv_shutdown_ = true;
        }
      });
  result->recv_thread_ =
      std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
          {}, "recv-thread", [s = result.get()]() { s->DoRecvWork(); }));
  return result;
}

absl::Status RecvThreadState::HandleRecvItem(recv_work_item& work,
                                             size_t& zc_send_count,
                                             size_t& non_zc_send_count) {
  SlabAllocator::Allocation alloc;
  size_t offset = 0;
  if (work.recv_size > uallocator_.max_allocation_size() ||
      (allocator_.has_value() &&
       work.recv_size > allocator_->max_allocation_size())) {
    return absl::UnimplementedError(
        "TODO(parkers): implement frame segmenting");
  }
#ifdef TCP_ZEROCOPY_RECEIVE
  if (allocator_.has_value()) {
    alloc = allocator_->Allocate(work.recv_size);
    while (offset != work.recv_size) {
      if ((offset & (4096 - 1)) != 0) {
        break;
      }
      struct tcp_zerocopy_receive zc;
      socklen_t zc_len = sizeof(zc);
      int res;
      memset(&zc, 0, sizeof(zc));
      zc.address = (uint64_t)(alloc.data) + offset;
      zc.length = work.recv_size - offset;

      struct pollfd fds[1];
      fds[0] = {.fd = work.fd, .events = POLLIN, .revents = 0};
      poll(&fds[0], 1, -1);

      res =
          getsockopt(work.fd, IPPROTO_TCP, TCP_ZEROCOPY_RECEIVE, &zc, &zc_len);
      if (res == -1) {
        std::move(alloc.on_done)();
        return absl::ErrnoToStatus(errno, "zero-copy-recv");
      }
      offset += zc.length;
      zc_send_count += zc.length;
      if (zc.length == 0) {
        break;
      }
    }
  }
#endif
  if (offset != work.recv_size) {
    // Zero copy failed early, have to do copies for the rest.
    if (offset == 0) {
      if (alloc.on_done) {
        std::move(alloc.on_done)();
      }
      alloc = uallocator_.Allocate(work.recv_size);
    } else {
      auto old_alloc = std::move(alloc);
      alloc = uallocator_.Allocate(work.recv_size);
      memcpy(alloc.data, old_alloc.data, offset);
      std::move(old_alloc.on_done)();
    }
    while (offset != work.recv_size) {
      ssize_t recv_count =
          recv(work.fd, reinterpret_cast<char*>(alloc.data) + offset,
               work.recv_size - offset, 0);
      if (recv_count < 0) {
        std::move(alloc.on_done)();
        return absl::ErrnoToStatus(errno, "recv-fallback");
      }
      non_zc_send_count += recv_count;
      offset += recv_count;
    }
  }
  aux::BulkTransportInterface::Message msg;
  msg.data = alloc.data;
  msg.size = work.recv_size;
  msg.on_done = std::move(alloc.on_done);
  std::move(work.on_recv)(std::move(msg));
  return absl::OkStatus();
}

class SocketBulkTransport : public BulkTransportInterface {
 public:
  SocketBulkTransport(
      std::vector<std::shared_ptr<RecvThreadState>> thread_states,
      std::vector<std::shared_ptr<SharedSendWorkQueue>> send_work_queues) {
    msg_queue_ = std::make_shared<SharedSendMsgQueue>();
    for (int i = 0; i < thread_states.size(); ++i) {
      auto conn = std::make_shared<Conn>();
      conn->thread_state = std::move(thread_states[i]);
      conn->send_msg_queue = msg_queue_;
      conn->send_work_queue = std::move(send_work_queues[i]);
      conn->connection_id = i;
      connections_.push_back(std::move(conn));
    }
  }

  ~SocketBulkTransport() override { msg_queue_->NoMoreMessages(); }

  void Send(SendMessage msg) override {
    msg_queue_->ScheduleSendWork(std::move(msg));
  }

  void Recv(size_t size, int bond_id,
            absl::AnyInvocable<void(absl::StatusOr<Message> msg) &&> on_recv)
      override {
    auto& conn = connections_[bond_id];
    absl::MutexLock l(&conn->mu);
    if (conn->fd == -1) {
      conn->pending_recvs.push_back({size, std::move(on_recv)});
    } else {
      conn->thread_state->ScheduleRecvWork(size, conn->fd, std::move(on_recv));
    }
  }

  struct Conn {
    absl::Mutex mu;
    int connection_id;
    struct PendingRecv {
      size_t size;
      absl::AnyInvocable<void(absl::StatusOr<Message> msg) &&> on_recv;
    };
    std::shared_ptr<RecvThreadState> thread_state;
    std::shared_ptr<SharedSendMsgQueue> send_msg_queue;
    std::shared_ptr<SharedSendWorkQueue> send_work_queue;
    std::vector<PendingRecv> pending_recvs;
    int fd = -1;

    void AcceptSock(int accept_fd) {
      SharedSendMsgQueue::StartSubConnectionSender(
          accept_fd, connection_id, send_msg_queue, send_work_queue);
      {
        absl::MutexLock l(&mu);
        fd = accept_fd;
        for (auto& pending_recv : pending_recvs) {
          thread_state->ScheduleRecvWork(pending_recv.size, accept_fd,
                                         std::move(pending_recv.on_recv));
        }
      }
      pending_recvs = std::vector<PendingRecv>();
    }
  };

  const std::vector<std::shared_ptr<Conn>>& connections() {
    return connections_;
  }

 private:
  std::shared_ptr<SharedSendMsgQueue> msg_queue_;
  std::vector<std::shared_ptr<Conn>> connections_;
};

class SocketBulkTransportFactory : public BulkTransportFactory {
 public:
  SocketBulkTransportFactory(
      std::vector<std::shared_ptr<RecvThreadState>> thread_states,
      std::vector<std::shared_ptr<SharedSendWorkQueue>> send_work_queues)
      : thread_states_(std::move(thread_states)),
        send_work_queues_(std::move(send_work_queues)) {}

  BulkTransportInitResult InitBulkTransport() override {
    BulkTransportInitResult result;
    auto bulk_transport = std::make_unique<SocketBulkTransport>(
        thread_states_, send_work_queues_);
    for (auto& listener : listeners_) {
      result.request.add_bulk_transport_address(listener->addr().ToString());
    }
    result
        .start_bulk_transport = [conns = bulk_transport->connections(),
                                 addrs = addrs_](
                                    const SocketTransferEstablishBulkTransport&
                                        remote_bulk_transport_info) {
      uint64_t next_id = remote_bulk_transport_info.bulk_transport_uuid(0);
      for (uint64_t i = 0;
           i < remote_bulk_transport_info.bulk_transport_address_size(); ++i) {
        uint64_t uuid = next_id + i;
        SocketAddress addr =
            SocketAddress::Parse(
                remote_bulk_transport_info.bulk_transport_address(i))
                .value();
        int cfd =
            socket(addrs[i].address().sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
        CHECK_EQ(bind(cfd, reinterpret_cast<const struct sockaddr*>(&addrs[i]),
                      sizeof(SocketAddress)),
                 0)
            << strerror(errno) << " " << errno;
        CHECK_EQ(connect(cfd, reinterpret_cast<const struct sockaddr*>(&addr),
                         sizeof(addr)),
                 0)
            << strerror(errno) << " " << errno;

        int value = 1;
        CHECK_GE(
            setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value)), 0)
            << strerror(errno) << " " << errno;
        CHECK_GE(
            setsockopt(cfd, SOL_SOCKET, SO_ZEROCOPY, &value, sizeof(value)), 0)
            << strerror(errno) << " " << errno;
        CHECK_GE(cfd, 0) << strerror(errno) << " " << errno;
        CHECK_EQ(send(cfd, &uuid, sizeof(uuid), 0), sizeof(uuid))
            << strerror(errno) << " " << errno;
        conns[i]->AcceptSock(cfd);
      }
    };
    result.bulk_transport = std::move(bulk_transport);
    return result;
  }

  BulkTransportRecvResult RecvBulkTransport(
      const SocketTransferEstablishBulkTransport& remote_bulk_transport_info)
      override {
    BulkTransportRecvResult result;
    auto bulk_transport = std::make_unique<SocketBulkTransport>(
        thread_states_, send_work_queues_);
    for (auto& listener : listeners_) {
      result.request.add_bulk_transport_address(listener->addr().ToString());
    }
    uint64_t next_uuid =
        recv_state_->AllocateUUIDs(bulk_transport->connections());
    result.bulk_transport = std::move(bulk_transport);
    result.request.add_bulk_transport_uuid(next_uuid);
    return result;
  }

  static absl::StatusOr<std::shared_ptr<SocketBulkTransportFactory>> Create(
      std::vector<SocketAddress> addrs,
      std::vector<std::shared_ptr<RecvThreadState>> thread_states,
      std::vector<std::shared_ptr<SharedSendWorkQueue>> send_work_queues) {
    auto result = std::make_shared<SocketBulkTransportFactory>(
        std::move(thread_states), std::move(send_work_queues));
    result->addrs_ = addrs;
    for (auto& addr : addrs) {
      auto listener_or = SocketListener::Listen(
          addr,
          [state = result->recv_state_](int sockfd, const SocketAddress& addr) {
            uint64_t uuid;
            if (recv(sockfd, &uuid, sizeof(uuid), 0) != sizeof(uuid)) {
              close(sockfd);
            } else {
              state->DoAccept(sockfd, uuid);
            }
          });
      if (!listener_or.ok()) {
        return listener_or.status();
      }
      result->listeners_.push_back(*std::move(listener_or));
    }
    for (auto& listener : result->listeners_) {
      listener->Start();
    }
    return result;
  }

 private:
  struct RecvState {
    absl::Mutex mu;
    uint64_t next_id = 0;
    absl::flat_hash_map<uint64_t, std::shared_ptr<SocketBulkTransport::Conn>>
        waiting_for_connect;
    void DoAccept(int sockfd, uint64_t uuid) {
      absl::MutexLock l(&mu);
      auto it = waiting_for_connect.find(uuid);
      if (it == waiting_for_connect.end()) {
        close(sockfd);
        return;
      }
      auto conn = std::move(it->second);
      waiting_for_connect.erase(it);
      conn->AcceptSock(sockfd);
    }
    uint64_t AllocateUUIDs(
        std::vector<std::shared_ptr<SocketBulkTransport::Conn>> connections) {
      absl::MutexLock l(&mu);
      uint64_t result = next_id;
      next_id += connections.size();
      for (uint64_t i = 0; i < connections.size(); ++i) {
        waiting_for_connect[result + i] = connections[i];
      }
      return result;
    }
  };
  std::vector<std::shared_ptr<RecvThreadState>> thread_states_;
  std::vector<std::shared_ptr<SharedSendWorkQueue>> send_work_queues_;
  std::vector<SocketAddress> addrs_;
  std::shared_ptr<RecvState> recv_state_ = std::make_shared<RecvState>();
  std::vector<std::unique_ptr<SocketListener>> listeners_;
};

absl::StatusOr<std::shared_ptr<BulkTransportFactory>>
CreateSocketBulkTransportFactory(std::vector<SocketAddress> addrs,
                                 std::optional<SlabAllocator> allocator,
                                 SlabAllocator unpinned_allocator) {
  size_t num_connections = addrs.size();

  std::vector<std::shared_ptr<RecvThreadState>> thread_states;
  std::vector<std::shared_ptr<SharedSendWorkQueue>> send_work_queues;
  thread_states.reserve(num_connections);
  send_work_queues.reserve(num_connections);
  for (int i = 0; i < num_connections; ++i) {
    thread_states.push_back(
        RecvThreadState::Create(allocator, unpinned_allocator));
    send_work_queues.push_back(SharedSendWorkQueue::Start());
  }
  return SocketBulkTransportFactory::Create(addrs, std::move(thread_states),
                                            std::move(send_work_queues));
}

}  // namespace aux
