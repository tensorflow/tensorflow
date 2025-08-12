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
#include "xla/python/transfer/socket-server.h"

#include <netinet/tcp.h>  // for TCP_NODELAY

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace aux {

class SocketServer::SocketNetworkState : public PollEventLoop::Handler {
 public:
  explicit SocketNetworkState(std::shared_ptr<PullTable> table,
                              std::shared_ptr<BulkTransportFactory> factory,
                              int fd)
      : table_(std::move(table)), factory_(std::move(factory)), fd_(fd) {
    is_connected_ = true;
  }
  explicit SocketNetworkState(std::shared_ptr<PullTable> table,
                              std::shared_ptr<BulkTransportFactory> factory,
                              const SocketAddress& addr)
      : table_(std::move(table)),
        factory_(std::move(factory)),
        fd_(-1),
        remote_addr_(addr) {
    StartConnect();
  }
  ~SocketNetworkState() override { close(fd_); }

  void StartConnect() {
    int send_fd = socket(remote_addr_.address().sa_family,
                         SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    connect(send_fd, reinterpret_cast<const struct sockaddr*>(&remote_addr_),
            sizeof(remote_addr_));
    int value = 1;
    CHECK_GE(
        setsockopt(send_fd, SOL_SOCKET, SO_ZEROCOPY, &value, sizeof(value)), 0)
        << strerror(errno) << " " << errno;
    CHECK_GE(
        setsockopt(send_fd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value)), 0)
        << strerror(errno) << " " << errno;
    fd_ = send_fd;
  }

  void PopulatePollInfo(pollfd& events) override {
    events.fd = fd_;
    events.events = POLLIN;
    if (!can_send_) {
      events.events = POLLOUT;
    }
  }

  bool HandleEvents(const pollfd& events) override {
    if (!is_connected_) {
      // poll() may remind us that fd_ is invalid while waiting to reconnect.
      if (fd_ == -1) {
        return true;
      }
      // If HUP with an error happens, then schedule a reconnect.
      if ((events.revents & POLLHUP) && (events.revents & POLLERR)) {
        fd_ = -1;
        loop()->ScheduleAt(absl::Now() + absl::Seconds(2),
                           [this]() { StartConnect(); });
        return true;
      }
      if (!(events.revents & POLLOUT)) {
        return true;
      }
      is_connected_ = true;
    }
    if (events.revents & POLLIN) {
      ssize_t recv_size =
          recv(fd_, network_buffer_.get(), 4096 - recv_count_, 0);
      if (recv_size == 0) {
        {
          absl::MutexLock l(&mu_);
          is_poisoned_ = true;
          peer_is_closed_ = true;
          poison_status_ = absl::InternalError(
              "SocketServer: Connection closed recv() == 0.");
        }
        ClearDestTable();
      } else if (recv_size == -1 && errno == EAGAIN) {
      } else {
        if (recv_size < 0) {
          Poison(absl::InternalError(
              absl::StrFormat("%ld = recv() failed errno: %d err: %s",
                              recv_size, errno, strerror(errno))));
          return true;
        }
        recv_count_ += recv_size;
        while (recv_count_ >= sizeof(uint32_t)) {
          uint32_t frame_size;
          memcpy(&frame_size, network_buffer_.get(), sizeof(uint32_t));
          if (frame_size < 0 || frame_size > 4096 - sizeof(uint32_t)) {
            Poison(absl::InternalError(
                absl::StrFormat("frame_size is too large: %lu", frame_size)));
            return true;
          }
          size_t total_frame_size =
              static_cast<size_t>(frame_size) + sizeof(uint32_t);
          // Needs more input.
          if (total_frame_size > recv_count_) {
            break;
          }
          absl::string_view buffer(network_buffer_.get() + sizeof(uint32_t),
                                   frame_size);
          SocketTransferRequest req;
          if (!req.ParseFromArray(buffer.data(), buffer.size())) {
            Poison(
                absl::InternalError("Could not parse SocketTransferRequest."));
            return true;
          }
          HandlePacket(req);
          if (total_frame_size < recv_count_) {
            memmove(network_buffer_.get(),
                    network_buffer_.get() + total_frame_size,
                    recv_count_ - total_frame_size);
          }
          recv_count_ -= total_frame_size;
        }
      }
    }
    if (events.revents & POLLOUT) {
      can_send_ = true;
    }
    mu_.Lock();
    while (!frames_.empty() && can_send_) {
      auto& packet_to_send = frames_.front();
      if (packet_to_send.empty()) {
        shutdown(fd_, SHUT_WR);
        break;
      }
      const void* base = packet_to_send.data() + write_offset_;
      size_t size = packet_to_send.size() - write_offset_;
      ssize_t send_size = send(fd_, base, size, 0);
      if (send_size > 0) {
        write_offset_ += send_size;
        if (send_size == size) {
          write_offset_ = 0;
          frames_.pop_front();
        } else {
          can_send_ = false;
        }
      } else {
        mu_.Unlock();
        Poison(absl::InternalError(
            absl::StrFormat("%ld = send() failed errno: %d err: %s", send_size,
                            errno, strerror(errno))));
        return true;
      }
    }
    if (peer_is_closed_ && num_refs_ == 0) {
      mu_.Unlock();
      delete this;
      return false;
    }
    mu_.Unlock();
    return true;
  }

  bool can_send_ = false;
  bool is_connected_ = false;
  size_t write_offset_ = 0;
  std::deque<std::string> frames_;

  void SendFrame(const SocketTransferRequest& req) {
    uint32_t header = req.ByteSizeLong();
    std::string opacket = std::string(absl::string_view(
        reinterpret_cast<const char*>(&header), sizeof(header)));
    req.AppendToString(&opacket);
    {
      absl::MutexLock l(&mu_);
      frames_.push_back(std::move(opacket));
    }
    loop()->SendWake(this);
  }

  tsl::RCReference<ChunkDestination> GetNextDest(size_t req_id, size_t offset,
                                                 size_t size, bool is_largest) {
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
    return dest;
  }

  std::optional<size_t> InstallPull(tsl::RCReference<ChunkDestination> dest) {
    mu_.Lock();
    if (is_poisoned_) {
      auto poison_status = poison_status_;
      dest->Poison(std::move(poison_status));
      mu_.Unlock();
      return std::nullopt;
    }
    dests_[next_req_id_].dest = std::move(dest);
    size_t req_id = next_req_id_;
    ++next_req_id_;
    mu_.Unlock();
    return req_id;
  }

  std::optional<size_t> InstallPullList(
      std::vector<tsl::RCReference<ChunkDestination>> dests) {
    mu_.Lock();
    if (is_poisoned_) {
      auto poison_status = poison_status_;
      for (auto& dest : dests) {
        dest->Poison(poison_status);
      }
      mu_.Unlock();
      return std::nullopt;
    }
    size_t req_id = next_req_id_;
    for (auto& dest : dests) {
      dests_[next_req_id_].dest = std::move(dest);
      ++next_req_id_;
    }
    mu_.Unlock();
    return req_id;
  }

  void StartBulkTransporting() {
    auto info = factory_->InitBulkTransport();
    bulk_transport_ = std::move(info.bulk_transport);
    SocketTransferRequest response;
    *response.mutable_bulk_transport() = std::move(info.request);
    start_bulk_transport_ = std::move(info.start_bulk_transport);
    SendFrame(response);
  }

  BulkTransportInterface* bulk_transport() { return bulk_transport_.get(); }

  void HandlePacket(const SocketTransferPullRequest& req) {
    class SocketConnectionState : public ConnectionState {
     public:
      explicit SocketConnectionState(SocketNetworkState* state)
          : state_(state) {}
      ~SocketConnectionState() override { state_->DropRef(); }

      void Send(size_t req_id, const void* data, size_t offset, size_t size,
                bool is_largest,
                absl::AnyInvocable<void() &&> on_done) override {
        BulkTransportInterface::SendMessage msg;
        msg.data = const_cast<void*>(data);
        msg.size = size;
        msg.on_send = [val = tsl::FormRef(this), offset, req_id, is_largest](
                          int bond_id, size_t size) {
          SocketTransferRequest response;
          auto* packet = response.mutable_packet();
          packet->set_bulk_transport_id(bond_id);
          packet->set_offset(offset);
          packet->set_size(size);
          packet->set_req_id(req_id);
          packet->set_is_largest(is_largest);
          PollEventLoop::GetDefault()->Schedule(
              [val, response]() { val->state_->SendFrame(response); });
        };
        msg.on_done = std::move(on_done);
        state_->bulk_transport_->Send(std::move(msg));
      }

      void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                     absl::Status error) override {
        SocketTransferRequest response;
        auto* packet = response.mutable_error_packet();
        if (error.message().size() < 2048) {
          packet->set_error_message(std::string(error.message()));
        } else {
          packet->set_error_message(absl::StrCat(
              error.message().substr(0, 2048), "... truncated ..."));
        }
        packet->set_offset(offset);
        packet->set_size(size);
        packet->set_req_id(req_id);
        packet->set_is_largest(is_largest);
        state_->SendFrame(response);
      }

     private:
      SocketNetworkState* state_;
    };
    {
      absl::MutexLock l(&mu_);
      ++num_refs_;
    }
    table_->Handle(tsl::MakeRef<SocketConnectionState>(this), req,
                   req.req_id());
  }

  void HandlePacket(const SocketTransferRequest& req) {
    switch (req.msg_case()) {
      case SocketTransferRequest::kPacket:
        return HandlePacket(req.packet());
      case SocketTransferRequest::kPull:
        return HandlePacket(req.pull());
      case SocketTransferRequest::kBulkTransport:
        return HandlePacket(req.bulk_transport());
      case SocketTransferRequest::kHalfClose:
        return HandlePacket(req.half_close());
      case SocketTransferRequest::kErrorPacket:
        return HandlePacket(req.error_packet());
      default:
        LOG(FATAL) << "Implement: " << req.DebugString();
    }
  }

  void HandlePacket(const SocketTransferPacketErrorHeader& packet) {
    auto dest = GetNextDest(packet.req_id(), packet.offset(), packet.size(),
                            packet.is_largest());
    dest->Poison(absl::InternalError(
        absl::StrCat("Error while transferring: ", packet.error_message())));
  }

  void HandlePacket(const SocketTransferEstablishBulkTransport& req) {
    if (start_bulk_transport_) {
      std::move(start_bulk_transport_)(req);
      start_bulk_transport_ = nullptr;
    } else {
      auto info = factory_->RecvBulkTransport(req);
      bulk_transport_ = std::move(info.bulk_transport);
      SocketTransferRequest response;
      *response.mutable_bulk_transport() = std::move(info.request);
      SendFrame(response);
    }
  }

  void HandlePacket(const SocketTransferPacketHeader& packet) {
    auto dest = GetNextDest(packet.req_id(), packet.offset(), packet.size(),
                            packet.is_largest());
    bulk_transport_->Recv(
        packet.size(), packet.bulk_transport_id(),
        [offset = packet.offset(), dest = std::move(dest)](
            absl::StatusOr<BulkTransportInterface::Message> msgor) {
          auto msg = std::move(msgor).value();
          CHECK_OK(
              dest->Put(msg.data, offset, msg.size, std::move(msg.on_done)));
        });
  }

  void DropRef() {
    {
      absl::MutexLock l(&mu_);
      CHECK_NE(num_refs_, 0);
      --num_refs_;
      ShutdownIfNeeded();
    }
  }

  void NoMorePulls() {
    SocketTransferRequest msg;
    msg.mutable_half_close();
    SendFrame(msg);
    DropRef();
  }

  void HandlePacket(const SocketTransferHalfClose& half_close) {
    mu_.Lock();
    CHECK(!peer_is_closed_);
    peer_is_closed_ = true;
    ShutdownIfNeeded();
    mu_.Unlock();
  }

  void ShutdownIfNeeded() {
    if (!peer_is_closed_ || num_refs_ != 0) {
      return;
    }
    loop()->SendWake(this);
  }

  void Pull(uint64_t uuid, int buf_id,
            tsl::RCReference<ChunkDestination> dest) {
    std::optional<size_t> req_id = InstallPull(std::move(dest));
    if (!req_id.has_value()) {
      return;
    }
    SocketTransferRequest msg;
    SocketTransferPullRequest& req = *msg.mutable_pull();
    req.set_uuid(uuid);
    req.add_buffer_ids(buf_id);
    req.set_req_id(*req_id);
    SendFrame(msg);
  }

  void Pull(uint64_t uuid, absl::Span<const int> buffer_ids,
            std::vector<tsl::RCReference<ChunkDestination>> dests) {
    std::optional<size_t> req_id = InstallPullList(std::move(dests));
    if (!req_id.has_value()) {
      return;
    }
    constexpr size_t kBufferIdChunkSize = 256;
    SocketTransferRequest msg;
    SocketTransferPullRequest& req = *msg.mutable_pull();
    req.set_uuid(uuid);
    req.set_req_id(*req_id);
    for (int buf_id : buffer_ids) {
      req.add_buffer_ids(buf_id);
      if (req.buffer_ids_size() == kBufferIdChunkSize) {
        SendFrame(msg);
        req.set_req_id(req.req_id() + kBufferIdChunkSize);
        req.clear_buffer_ids();
      }
    }
    if (req.buffer_ids_size() > 0) {
      SendFrame(msg);
    }
  }

  void InjectFailure() {
    uint32_t header = 12341024;
    std::string opacket = std::string(absl::string_view(
        reinterpret_cast<const char*>(&header), sizeof(header)));
    opacket += "Injected Failure.";
    {
      absl::MutexLock l(&mu_);
      frames_.push_back(std::move(opacket));
    }
    loop()->SendWake(this);
  }

  static void Accept(std::shared_ptr<PullTable> table,
                     std::shared_ptr<BulkTransportFactory> factory,
                     int sockfd) {
    auto* remote = new SocketNetworkState(table, factory, sockfd);
    remote->Register();
  }

  void ClearDestTable() {
    absl::Status poison_status;
    absl::flat_hash_map<uint64_t, DestState> dests;
    {
      absl::MutexLock l(&mu_);
      std::swap(dests, dests_);
      poison_status = poison_status_;
    }
    for (auto& v : dests) {
      v.second.dest->Poison(poison_status);
    }
  }

  void Poison(absl::Status s) {
    {
      absl::MutexLock l(&mu_);
      is_poisoned_ = true;
      shutdown(fd_, SHUT_RDWR);
      poison_status_ = s;
    }
    ClearDestTable();
  }

 private:
  std::shared_ptr<PullTable> table_;
  std::shared_ptr<BulkTransportFactory> factory_;
  absl::Mutex mu_;
  size_t num_refs_ = 1;
  bool peer_is_closed_ = false;
  bool is_poisoned_ = false;
  absl::Status poison_status_;
  int fd_ = -1;
  SocketAddress remote_addr_;
  size_t recv_count_ = 0;
  std::unique_ptr<char[]> network_buffer_ =
      std::unique_ptr<char[]>(new char[4096]);

  uint64_t next_req_id_ = 0;
  struct DestState {
    ssize_t transferred_size = 0;
    tsl::RCReference<ChunkDestination> dest;
  };
  absl::flat_hash_map<uint64_t, DestState> dests_;

  std::unique_ptr<BulkTransportInterface> bulk_transport_;
  absl::AnyInvocable<void(const SocketTransferEstablishBulkTransport&
                              remote_bulk_transport_info) &&>
      start_bulk_transport_ = nullptr;
};

SocketServer::Connection::~Connection() { local_->NoMorePulls(); }

void SocketServer::Connection::Pull(uint64_t uuid, int buffer_id,
                                    tsl::RCReference<ChunkDestination> dest) {
  local_->Pull(uuid, buffer_id, std::move(dest));
}

void SocketServer::Connection::Pull(
    uint64_t uuid, absl::Span<const int> buffer_ids,
    std::vector<tsl::RCReference<ChunkDestination>> dests) {
  local_->Pull(uuid, buffer_ids, std::move(dests));
}

void SocketServer::Connection::InjectFailure() { local_->InjectFailure(); }

absl::Status SocketServer::Start(
    const SocketAddress& addr,
    std::shared_ptr<BulkTransportFactory> bulk_transport_factory) {
  bulk_transport_factory_ = bulk_transport_factory;
  auto v = SocketListener::Listen(
      addr,
      [pull_table = pull_table_, factory = bulk_transport_factory_](
          int sockfd, const SocketAddress& addr) {
        int value = 1;
        CHECK_GE(
            setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value)),
            0)
            << strerror(errno) << " " << errno;
        SocketNetworkState::Accept(pull_table, factory, sockfd);
      },
      SOCK_NONBLOCK);
  if (!v.ok()) {
    return v.status();
  }
  listener_ = *std::move(v);
  listener_->Start();
  return absl::OkStatus();
}

tsl::RCReference<SocketServer::Connection> SocketServer::Connect(
    const SocketAddress& other_addr) {
  auto* local_ =
      new SocketNetworkState(pull_table_, bulk_transport_factory_, other_addr);
  local_->Register();
  local_->StartBulkTransporting();
  return tsl::MakeRef<Connection>(local_);
}

}  // namespace aux
