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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/profiler/lib/traceme.h"

namespace aux {

class SocketServer::SocketNetworkState : public SocketFdPacketState {
 public:
  explicit SocketNetworkState(std::shared_ptr<ConnectionList> connections,
                              std::shared_ptr<PullTable> table,
                              std::shared_ptr<BulkTransportFactory> factory,
                              int fd)
      : table_(std::move(table)),
        factory_(std::move(factory)),
        connections_(std::move(connections)) {
    RegisterFd(fd, /*start_connected=*/true);
    absl::MutexLock l(connections_->mu);
    connections_->list.push_back(this);
    connection_it_ = --connections_->list.end();
  }
  explicit SocketNetworkState(std::shared_ptr<PullTable> table,
                              std::shared_ptr<BulkTransportFactory> factory,
                              const SocketAddress& addr)
      : table_(std::move(table)),
        factory_(std::move(factory)),
        remote_addr_(addr) {
  }

  ~SocketNetworkState() override {
    if (connections_) {
      absl::MutexLock l(connections_->mu);
      connections_->list.erase(connection_it_);
    }
  }

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
    RegisterFd(send_fd, /*start_connected=*/false);
  }

  void ConnectFailed() override {
    loop()->ScheduleAt(absl::Now() + absl::Seconds(2),
                       [this]() { StartConnect(); });
  }

  void RecvClosed(absl::Status error) override {
    Shutdown(SHUT_RDWR);
    if (error.ok()) {
      error =
          absl::InternalError("SocketServer: Connection closed recv() == 0.");
    }
    Poison(error);
    DropSysRef();
  }

  void SendClosed(absl::Status error) override {
    Shutdown(SHUT_RDWR);
    {
      absl::MutexLock l(mu_);
      is_poisoned_ = true;
      poison_status_ =
          absl::InternalError("SocketServer: Connection closed recv() == 0.");
    }
    DropSysRef();
  }

  bool SendFrame(const SocketTransferRequest& req) {
    uint32_t header = req.ByteSizeLong();
    std::string opacket = std::string(absl::string_view(
        reinterpret_cast<const char*>(&header), sizeof(header)));
    req.AppendToString(&opacket);
    return SendRawFrame(std::move(opacket));
  }

  std::optional<tsl::RCReference<ChunkDestination>> GetNextDest(
      size_t req_id, size_t offset, size_t size, bool is_largest) {
    tsl::RCReference<ChunkDestination> dest;
    {
      absl::MutexLock l(mu_);
      if (is_poisoned_) {
        return std::nullopt;
      }
      auto it = dests_.find(req_id);
      if (it == dests_.end()) {
        Shutdown(SHUT_RDWR);
        is_poisoned_ = true;
        poison_status_ =
            absl::InternalError("SocketServer: it != dests_.end()");
        return std::nullopt;
      }
      if (is_largest) {
        it->second.transferred_size += offset;
      } else {
        it->second.transferred_size -= size;
      }
      if (it->second.transferred_size == 0) {
        dest = std::move(it->second.dest);
        dests_.erase(it);
        CheckSendNoMorePulls();
      } else {
        dest = it->second.dest;
      }
    }
    return dest;
  }

  std::optional<size_t> InstallPull(tsl::RCReference<ChunkDestination> dest) {
    mu_.lock();
    if (is_poisoned_) {
      auto poison_status = poison_status_;
      dest->Poison(std::move(poison_status));
      mu_.unlock();
      return std::nullopt;
    }
    dests_[next_req_id_].dest = std::move(dest);
    size_t req_id = next_req_id_;
    ++next_req_id_;
    mu_.unlock();
    return req_id;
  }

  std::optional<size_t> InstallPullList(
      std::vector<tsl::RCReference<ChunkDestination>> dests) {
    mu_.lock();
    if (is_poisoned_) {
      auto poison_status = poison_status_;
      for (auto& dest : dests) {
        dest->Poison(poison_status);
      }
      mu_.unlock();
      return std::nullopt;
    }
    size_t req_id = next_req_id_;
    for (auto& dest : dests) {
      dests_[next_req_id_].dest = std::move(dest);
      ++next_req_id_;
    }
    mu_.unlock();
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

  void HandlePacket(absl::string_view buffer) override {
    SocketTransferRequest req;
    if (!req.ParseFromString(buffer)) {
      Poison(absl::InternalError("Could not parse SocketTransferRequest."));
      return;
    }
    HandlePacket(req);
  }

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
                          absl::StatusOr<int> bond_id, size_t size) {
          if (!bond_id.ok()) {
            val->SendError(req_id, offset, size, is_largest, bond_id.status());
            return;
          }
          SocketTransferRequest response;
          auto* packet = response.mutable_packet();
          packet->set_bulk_transport_id(*bond_id);
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
      absl::MutexLock l(mu_);
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
    if (!dest.has_value()) {
      return;
    }
    (*dest)->Poison(absl::InternalError(
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
    if (!dest.has_value()) {
      return;
    }
    bulk_transport_->Recv(
        packet.size(), packet.bulk_transport_id(),
        [offset = packet.offset(), dest = *std::move(dest)](
            absl::StatusOr<BulkTransportInterface::Message> msgor) {
          if (!msgor.ok()) {
            dest->Poison(msgor.status());
          } else {
            auto msg = std::move(msgor).value();
            CHECK_OK(
                dest->Put(msg.data, offset, msg.size, std::move(msg.on_done)));
          }
        });
  }

  std::unique_ptr<SocketNetworkState> DropRef() {
    absl::MutexLock l(mu_);
    CHECK_NE(num_refs_, 0);
    --num_refs_;
    ShutdownIfNeeded();
    return ReturnCheckIfRefsAreZero();
  }

  std::unique_ptr<SocketNetworkState> DropSysRef() {
    absl::MutexLock l(mu_);
    CHECK_NE(num_sys_refs_, 0);
    --num_sys_refs_;
    ShutdownIfNeeded();
    return ReturnCheckIfRefsAreZero();
  }

  std::unique_ptr<SocketNetworkState> ReturnCheckIfRefsAreZero()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (num_refs_ == 0 && num_sys_refs_ == 0) {
      // destroy outside of mutex scope.
      return std::unique_ptr<SocketNetworkState>(this);
    }
    return {};
  }

  void CheckSendNoMorePulls() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (dests_.empty() && no_more_pulls_) {
      SocketTransferRequest msg;
      msg.mutable_half_close();
      SendFrame(msg);
    }
  }

  void IncRef() {
    absl::MutexLock l(mu_);
    ++num_refs_;
  }

  std::unique_ptr<SocketNetworkState> NoMorePulls() {
    absl::MutexLock l(mu_);
    no_more_pulls_ = true;
    CheckSendNoMorePulls();
    CHECK_NE(num_refs_, 0);
    --num_refs_;
    return ReturnCheckIfRefsAreZero();
  }

  void HandlePacket(const SocketTransferHalfClose& half_close) {
    mu_.lock();
    peer_half_closed_ = true;
    ShutdownIfNeeded();
    mu_.unlock();
  }

  void ShutdownIfNeeded() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (num_refs_ == 0 && peer_half_closed_) {
      Shutdown(SHUT_RDWR);
    }
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

  void InjectFailure(Connection::FailureKind kind) {
    if (kind == Connection::kProtocolFailure) {
      uint32_t header = 12341024;
      std::string opacket = std::string(absl::string_view(
          reinterpret_cast<const char*>(&header), sizeof(header)));
      opacket += "Injected Failure.";
      SendRawFrame(std::move(opacket));
    } else {
      Poison(absl::InternalError("RECOVERABLE InjectFailure"));
    }
  }

  static void Accept(std::shared_ptr<ConnectionList> connections,
                     std::shared_ptr<PullTable> table,
                     std::shared_ptr<BulkTransportFactory> factory,
                     int sockfd) {
    new SocketNetworkState(std::move(connections), std::move(table),
                           std::move(factory), sockfd);
  }

  void ClearDestTable() {
    absl::Status poison_status;
    absl::flat_hash_map<uint64_t, DestState> dests;
    {
      absl::MutexLock l(mu_);
      std::swap(dests, dests_);
      poison_status = poison_status_;
    }
    for (auto& v : dests) {
      v.second.dest->Poison(poison_status);
    }
  }

  void Poison(absl::Status s) {
    {
      absl::MutexLock l(mu_);
      is_poisoned_ = true;
      Shutdown(SHUT_RDWR);
      poison_status_ = s;
    }
    ClearDestTable();
  }

 private:
  std::shared_ptr<PullTable> table_;
  std::shared_ptr<BulkTransportFactory> factory_;
  absl::Mutex mu_;
  size_t num_refs_ ABSL_GUARDED_BY(mu_) = 0;
  size_t num_sys_refs_ ABSL_GUARDED_BY(mu_) = 2;
  bool no_more_pulls_ = false;
  bool peer_half_closed_ = false;
  bool is_poisoned_ = false;
  absl::Status poison_status_;
  SocketAddress remote_addr_;

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
  std::shared_ptr<ConnectionList> connections_;
  std::list<SocketNetworkState*>::iterator connection_it_;
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

void SocketServer::Connection::InjectFailure(FailureKind kind) {
  local_->InjectFailure(kind);
}

absl::Status SocketServer::Start(
    const SocketAddress& addr,
    std::shared_ptr<BulkTransportFactory> bulk_transport_factory) {
  bulk_transport_factory_ = bulk_transport_factory;
  auto v = SocketListener::Listen(
      addr,
      [pull_table = pull_table_, connections = connections_,
       factory = bulk_transport_factory_](int sockfd,
                                          const SocketAddress& addr) {
        int value = 1;
        CHECK_GE(
            setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value)),
            0)
            << strerror(errno) << " " << errno;
        SocketNetworkState::Accept(connections, pull_table, factory, sockfd);
      },
      SOCK_NONBLOCK);
  if (!v.ok()) {
    return v.status();
  }
  listener_ = *std::move(v);
  listener_->Start();
  return absl::OkStatus();
}

SocketServer::~SocketServer() {
  listener_.reset();
  if (bulk_transport_factory_.use_count() == 1) {
    bulk_transport_factory_->BlockingShutdown();
  }
}

tsl::RCReference<SocketServer::Connection> SocketServer::Connect(
    const SocketAddress& other_addr) {
  auto* local_ =
      new SocketNetworkState(pull_table_, bulk_transport_factory_, other_addr);
  local_->StartBulkTransporting();
  local_->IncRef();
  local_->StartConnect();
  return tsl::MakeRef<Connection>(local_);
}

void SocketServer::WaitForQuiesce() {
  absl::MutexLock l(connections_->mu);
  auto cond = [&]() { return connections_->list.empty(); };
  connections_->mu.Await(absl::Condition(&cond));
}

}  // namespace aux
