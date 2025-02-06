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

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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
      : table_(std::move(table)), factory_(std::move(factory)), fd_(fd) {}
  ~SocketNetworkState() override { close(fd_); }

  void PopulatePollInfo(pollfd& events) override {
    events.fd = fd_;
    events.events = POLLIN;
    if (!can_send_) {
      events.events = POLLOUT;
    }
  }

  bool HandleEvents(const pollfd& events) override {
    if (events.revents & POLLIN) {
      ssize_t recv_size =
          recv(fd_, network_buffer_.get(), 4096 - recv_count_, 0);
      if (recv_size == 0) {
        absl::MutexLock l(&mu_);
        peer_is_closed_ = true;
      } else if (recv_size == -1 && errno == EAGAIN) {
      } else {
        CHECK_GE(recv_size, 0) << strerror(errno) << " " << errno;
        recv_count_ += recv_size;
        while (recv_count_ >= sizeof(uint32_t)) {
          uint32_t frame_size;
          memcpy(&frame_size, network_buffer_.get(), sizeof(uint32_t));
          CHECK_GE(frame_size, 0);
          CHECK_LE(frame_size, 4096 - sizeof(uint32_t));
          size_t total_frame_size =
              static_cast<size_t>(frame_size) + sizeof(uint32_t);
          // Needs more input.
          if (total_frame_size > recv_count_) {
            break;
          }
          absl::string_view buffer(network_buffer_.get() + sizeof(uint32_t),
                                   frame_size);
          SocketTransferRequest req;
          CHECK(req.ParseFromArray(buffer.data(), buffer.size()));
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

  size_t InstallPull(tsl::RCReference<ChunkDestination> dest) {
    size_t req_id;
    {
      absl::MutexLock l(&mu_);
      dests_[next_req_id_].dest = std::move(dest);
      req_id = next_req_id_;
      ++next_req_id_;
    }
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
      default:
        LOG(FATAL) << "Implement: " << req.DebugString();
    }
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
    size_t req_id = InstallPull(std::move(dest));
    SocketTransferRequest msg;
    SocketTransferPullRequest& req = *msg.mutable_pull();
    req.set_uuid(uuid);
    req.add_buffer_ids(buf_id);
    req.set_req_id(req_id);
    SendFrame(msg);
  }

  static void Accept(std::shared_ptr<PullTable> table,
                     std::shared_ptr<BulkTransportFactory> factory,
                     int sockfd) {
    auto* remote = new SocketNetworkState(table, factory, sockfd);
    remote->Register();
  }

 private:
  std::shared_ptr<PullTable> table_;
  std::shared_ptr<BulkTransportFactory> factory_;
  absl::Mutex mu_;
  size_t num_refs_ = 0;
  bool peer_is_closed_ = false;
  int fd_;
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

absl::Status SocketServer::Start(
    const SocketAddress& addr,
    std::shared_ptr<BulkTransportFactory> bulk_transport_factory) {
  bulk_transport_factory_ = bulk_transport_factory;
  auto v = SocketListener::Listen(
      addr,
      [pull_table = pull_table_, factory = bulk_transport_factory_](
          int sockfd, const SocketAddress& addr) {
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
  int send_fd = socket(other_addr.address().sa_family,
                       SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  connect(send_fd, reinterpret_cast<const struct sockaddr*>(&other_addr),
          sizeof(other_addr));
  int value = 1;
  CHECK_GE(setsockopt(send_fd, SOL_SOCKET, SO_ZEROCOPY, &value, sizeof(value)),
           0)
      << strerror(errno) << " " << errno;
  auto* local_ =
      new SocketNetworkState(pull_table_, bulk_transport_factory_, send_fd);
  local_->Register();
  local_->StartBulkTransporting();
  return tsl::MakeRef<Connection>(local_);
}

}  // namespace aux
