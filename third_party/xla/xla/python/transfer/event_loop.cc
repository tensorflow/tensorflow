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
#include "xla/python/transfer/event_loop.h"

#include <arpa/inet.h>
#include <linux/tcp.h>
#include <netdb.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

#include <atomic>
#include <cerrno>
#include <charconv>
#include <memory>
#include <queue>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tsl/platform/env.h"
#include "tsl/profiler/lib/traceme.h"

namespace aux {

class PollEventLoopImpl : public PollEventLoop {
 public:
  PollEventLoopImpl() {
    thread_ = std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
        {}, "event-loop-thread", [this]() { Run(); }));
  }

  void RegisterHandler(Handler* handler) override {
    absl::MutexLock l(mu_);
    inserts_.push_back(handler);
    WakeInternal();
  }
  void SendWake(Handler* handler) override {
    absl::MutexLock l(mu_);
    wakes_.insert(handler);
    WakeInternal();
  }

  void Schedule(absl::AnyInvocable<void() &&> cb) override {
    absl::MutexLock l(mu_);
    cbs_.push_back(std::move(cb));
    WakeInternal();
  }

  void ScheduleAt(absl::Time t, absl::AnyInvocable<void() &&> cb) override {
    absl::MutexLock l(mu_);
    bool needs_wake = timeout_cbs_.empty() || timeout_cbs_.top().t > t;
    timeout_cbs_.push({t, std::move(cb)});
    if (needs_wake) {
      WakeInternal();
    }
  }

 private:
  void Run() {
    // TODO(parkers): switch to epoll if handlers.size() is too big.
    std::vector<Handler*> handlers;
    std::vector<Handler*> new_handlers;
    std::vector<pollfd> fds;
    absl::Time wake_time = absl::InfiniteFuture();
    while (true) {
      fds.resize(handlers.size() + 1);
      for (size_t i = 0; i < handlers.size(); ++i) {
        memset(&fds[i], 0, sizeof(pollfd));
        handlers[i]->PopulatePollInfo(fds[i]);
      }
      fds[handlers.size()] = {.fd = event_fd_, .events = POLLIN, .revents = 0};
      {
        auto poll_time = absl::Now();
        if (wake_time < poll_time) {
        } else if (wake_time < absl::InfiniteFuture()) {
          auto poll_duration = absl::ToTimespec(wake_time - poll_time);
          ppoll(&fds[0], fds.size(), &poll_duration, nullptr);
        } else {
          poll(&fds[0], fds.size(), -1);
        }
      }
      absl::InlinedVector<Handler*, 4> inserts;
      absl::flat_hash_set<Handler*> wakes;
      std::vector<absl::AnyInvocable<void() &&>> cbs;
      // Consume eventfd wake.
      if (fds[handlers.size()].revents & POLLIN) {
        uint64_t counter;
        eventfd_read(event_fd_, &counter);
      }
      {
        absl::MutexLock l(mu_);
        std::swap(wakes_, wakes);
        std::swap(inserts, inserts_);
        std::swap(cbs, cbs_);
        {
          auto woken_time = absl::Now();
          while (!timeout_cbs_.empty() && timeout_cbs_.top().t < woken_time) {
            cbs.push_back(std::move(std::move(timeout_cbs_.top().cb)));
            timeout_cbs_.pop();
          }
          wake_time = timeout_cbs_.empty() ? absl::InfiniteFuture()
                                           : timeout_cbs_.top().t;
        }
        needs_wake_ = true;
      }
      for (auto& cb : cbs) {
        std::move(cb)();
      }
      new_handlers.clear();
      for (size_t i = 0; i < handlers.size(); ++i) {
        if ((fds[i].revents != 0 || wakes.contains(handlers[i])) &&
            !handlers[i]->HandleEvents(fds[i])) {
        } else {
          new_handlers.push_back(handlers[i]);
        }
      }
      for (auto* handler : inserts) {
        new_handlers.push_back(handler);
      }
      std::swap(new_handlers, handlers);
    }
  }

  void WakeInternal() {
    if (needs_wake_) {
      eventfd_write(event_fd_, 1);
      needs_wake_ = false;
    }
  }

  absl::Mutex mu_;
  // Suppresses multiple wakes from calling eventfd for each.
  bool needs_wake_ = true;
  int event_fd_ = eventfd(0, EFD_CLOEXEC);
  std::vector<absl::AnyInvocable<void() &&>> cbs_;
  struct TimeoutWork {
    absl::Time t;
    mutable absl::AnyInvocable<void() &&> cb;
  };
  struct TimeoutOrder {
    bool operator()(const TimeoutWork& a, const TimeoutWork& b) const {
      return a.t < b.t;
    }
  };
  std::priority_queue<TimeoutWork, std::vector<TimeoutWork>, TimeoutOrder>
      timeout_cbs_;
  absl::InlinedVector<Handler*, 4> inserts_;
  absl::flat_hash_set<Handler*> wakes_;
  std::unique_ptr<tsl::Thread> thread_;
};

void PollEventLoop::Handler::Register(PollEventLoop* loop) {
  loop_ = loop;
  loop->RegisterHandler(this);
}

PollEventLoop* PollEventLoop::GetDefault() {
  static auto* const loop = new PollEventLoopImpl;
  return loop;
}

class SocketListener::Handler : public PollEventLoop::Handler {
 public:
  using AcceptHandler =
      absl::AnyInvocable<void(int socket_fd, const SocketAddress& addr)>;

  Handler(int fd, int accept_flags, AcceptHandler on_accept)
      : on_accept_(std::make_shared<AcceptHandler>(std::move(on_accept))),
        fd_(fd),
        accept_flags_(accept_flags) {}
  ~Handler() override { close(fd_); }

  void PopulatePollInfo(pollfd& events) override {
    events.fd = fd_;
    events.events = POLLIN;
  }

  bool HandleEvents(const pollfd& events) override {
    if (shutdown_requested_.load()) {
      delete this;
      return false;
    }
    SocketAddress recv_addr;
    auto cfd_or = Accept(recv_addr);
    if (!cfd_or.ok()) {
      LOG(WARNING) << cfd_or.status();
      return true;
    }
    std::shared_ptr<AcceptHandler> on_accept;
    {
      absl::MutexLock l(mu_);
      on_accept = on_accept_;
    }
    (*on_accept)(*cfd_or, recv_addr);
    return true;
  }

  void Shutdown() {
    std::shared_ptr<AcceptHandler> on_accept;
    {
      absl::MutexLock l(mu_);
      std::swap(on_accept, on_accept_);
    }
    auto* l = loop();
    shutdown_requested_.store(true);
    l->SendWake(this);
  }

  absl::StatusOr<int> Accept(SocketAddress& recv_addr) {
    socklen_t addr_len = sizeof(recv_addr.mutable_address());
    int cfd = accept4(fd_, &recv_addr.mutable_address(), &addr_len,
                      accept_flags_ | SOCK_CLOEXEC);
    if (cfd == -1) {
      return absl::ErrnoToStatus(errno, "accept");
    }
    int value = 1;
    if (setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value)) != 0) {
      close(cfd);
      return absl::ErrnoToStatus(errno, "setsockopt TCP_NODELAY");
    }
    if (setsockopt(cfd, SOL_SOCKET, SO_ZEROCOPY, &value, sizeof(value)) != 0) {
      close(cfd);
      return absl::ErrnoToStatus(errno, "setsockopt SO_ZEROCOPY");
    }
    return cfd;
  }

 private:
  absl::Mutex mu_;
  std::shared_ptr<AcceptHandler> on_accept_ ABSL_GUARDED_BY(mu_);
  std::atomic<bool> shutdown_requested_{false};
  int fd_;
  int accept_flags_;
};

absl::StatusOr<std::unique_ptr<SocketListener>> SocketListener::Listen(
    const SocketAddress& addr,
    absl::AnyInvocable<void(int socket_fd, const SocketAddress& addr)>
        on_accept,
    int accept_flags) {
  auto result = std::make_unique<SocketListener>();
  int sfd = socket(addr.address().sa_family,
                   SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (sfd == -1) {
    return absl::ErrnoToStatus(errno, "Could not open socket.");
  }

  int value = 1;
  if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value)) != 0) {
    return absl::ErrnoToStatus(errno, "setsockopt");
  }
  if (setsockopt(sfd, SOL_SOCKET, SO_REUSEPORT, &value, sizeof(value)) != 0) {
    return absl::ErrnoToStatus(errno, "setsockopt");
  }
  if (bind(sfd, &addr.address(), addr.len()) != 0) {
    return absl::ErrnoToStatus(errno, "bind");
  }
  if (listen(sfd, 1024) != 0) {
    return absl::ErrnoToStatus(errno, "listen");
  }
  result->handler_ =
      new SocketListener::Handler(sfd, accept_flags, std::move(on_accept));

  sockaddr_storage bound_storage;
  socklen_t bound_len = sizeof(bound_storage);
  if (getsockname(sfd, (sockaddr*)&bound_storage, &bound_len) != 0) {
    return absl::ErrnoToStatus(errno, "getsockname");
  }
  result->addr_ = SocketAddress(bound_storage);
  return result;
}

SocketListener::~SocketListener() {
  if (started_) {
    handler_->Shutdown();
  } else {
    delete handler_;
  }
}

void SocketListener::Start() {
  if (started_) {
    return;
  }
  started_ = true;
  handler_->Register();
}

SocketAddress::SocketAddress() {
  memset(&storage_, 0, sizeof(storage_));
  storage_.ss_family = AF_INET6;
}

SocketAddress::SocketAddress(const sockaddr_in& saddr) {
  memcpy(&storage_, &saddr, sizeof(saddr));
}

SocketAddress::SocketAddress(const sockaddr_in6& saddr) {
  memcpy(&storage_, &saddr, sizeof(saddr));
}

SocketAddress::SocketAddress(const sockaddr_storage& saddr) : storage_(saddr) {}

socklen_t SocketAddress::len() const {
  switch (storage_.ss_family) {
    case AF_INET6:
      return sizeof(sockaddr_in6);
    case AF_INET:
      return sizeof(sockaddr_in);
    default:
      return sizeof(sockaddr_storage);
  }
}

std::string SocketAddress::ToString() const {
  char host[NI_MAXHOST], serv[NI_MAXSERV];
  int flags = NI_NUMERICHOST | NI_NUMERICSERV;
  if (getnameinfo(&address(), len(), host, sizeof(host), serv, sizeof(serv),
                  flags) == 0) {
    if (storage_.ss_family == AF_INET6) {
      return absl::StrCat("[", host, "]:", serv);
    }
    return absl::StrCat(host, ":", serv);
  }
  LOG(FATAL) << "Invalid IPAddress";
}

absl::StatusOr<uint16_t> ParsePort(absl::string_view colon_port) {
  if (!absl::ConsumePrefix(&colon_port, ":")) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing colon for port: '", colon_port, "'"));
  }
  uint16_t parsed_port;
  const char* last = colon_port.data() + colon_port.size();
  auto [ptr, ec] =
      std::from_chars(colon_port.data(), last, parsed_port, /*base=*/10);
  if (ec != std::errc{}) {
    return absl::ErrnoToStatus(static_cast<int>(ec),
                               absl::StrCat("std::from_chars could not parse '",
                                            colon_port, "' as a valid port"));
  }
  if (ptr != last) {
    return absl::InvalidArgumentError(
        absl::StrCat("Encountered non-numeric characters while parsing port: '",
                     colon_port, "'"));
  }
  return parsed_port;
}

absl::StatusOr<SocketAddress> SocketAddress::Parse(absl::string_view addr) {
  SocketAddress out;
  memset(&out.storage_, 0, sizeof(out.storage_));
  std::string ip_address;
  absl::string_view colon_port;
  if (absl::ConsumePrefix(&addr, "[")) {
    size_t it = addr.find(']');
    if (it == std::string::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("IPv6 address missing closing bracket: '", addr, "'"));
    }
    ip_address = addr.substr(0, it);
    colon_port = addr.substr(it + 1);
    out.storage_.ss_family = AF_INET6;
  } else {
    size_t it = addr.find(':');
    if (it == std::string::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("IPv4 address missing colon for port: '", addr, "'"));
    }
    ip_address = addr.substr(0, it);
    colon_port = addr.substr(it);
    out.storage_.ss_family = AF_INET;
  }
  absl::StatusOr<uint16_t> parsed_port = ParsePort(colon_port);
  if (!parsed_port.ok()) {
    return parsed_port.status();
  }

  void* sin_addr_dst;
  if (out.storage_.ss_family == AF_INET6) {
    sockaddr_in6* v6 = (sockaddr_in6*)&out.storage_;
    v6->sin6_port = htons(*parsed_port);
    sin_addr_dst = &v6->sin6_addr;
  } else {
    CHECK_EQ(out.storage_.ss_family, AF_INET);
    sockaddr_in* v4 = (sockaddr_in*)&out.storage_;
    v4->sin_port = htons(*parsed_port);
    sin_addr_dst = &v4->sin_addr;
  }
  if (inet_pton(out.storage_.ss_family, ip_address.c_str(), sin_addr_dst) ==
      -1) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("inet_pton failed when parsing address: '",
                            ip_address, "'"));
  }
  return out;
}

SocketFdPacketState::~SocketFdPacketState() { CHECK_EQ(fd_, -1); }

void SocketFdPacketState::PopulatePollInfo(pollfd& events) {
  events.fd = fd_;
  events.events = 0;
  if (!read_closed_) {
    events.events |= POLLIN;
  }
  if (!can_send_ && !write_closed_) {
    events.events |= POLLOUT;
  }
}

bool SocketFdPacketState::HandleEvents(const pollfd& events) {
  tsl::profiler::TraceMe __trace("SocketServer::HandleEvents");
  if (!is_connected_) {
    // If HUP with an error happens, then schedule a reconnect.
    if ((events.revents & POLLHUP) && (events.revents & POLLERR)) {
      mu_.lock();
      read_closed_ = true;
      write_closed_ = true;
      bool result = CloseIfNeeded();
      mu_.unlock();
      ConnectFailed();
      return result;
    }
    if (!(events.revents & POLLOUT)) {
      return true;
    }
    absl::MutexLock l(mu_);
    is_connected_ = true;
  }
  if ((events.revents & POLLIN) && !read_closed_) {
    ssize_t recv_size =
        recv(fd_, network_buffer_.get() + recv_count_, 4096 - recv_count_, 0);
    if (recv_size > 0) {
      recv_count_ += recv_size;
      while (recv_count_ >= sizeof(uint32_t)) {
        uint32_t frame_size;
        memcpy(&frame_size, network_buffer_.get(), sizeof(uint32_t));
        if (frame_size > 4096 - sizeof(uint32_t)) {
          mu_.lock();
          shutdown(fd_, SHUT_RD);
          read_closed_ = true;
          bool result = CloseIfNeeded();
          mu_.unlock();
          RecvClosed(absl::InternalError(
              absl::StrFormat("frame_size is too large: %lu", frame_size)));
          return result;
        }
        size_t total_frame_size =
            static_cast<size_t>(frame_size) + sizeof(uint32_t);
        // Needs more input.
        if (total_frame_size > recv_count_) {
          break;
        }
        absl::string_view buffer(network_buffer_.get() + sizeof(uint32_t),
                                 frame_size);
        HandlePacket(buffer);
        if (total_frame_size < recv_count_) {
          memmove(network_buffer_.get(),
                  network_buffer_.get() + total_frame_size,
                  recv_count_ - total_frame_size);
        }
        recv_count_ -= total_frame_size;
      }
    } else if (recv_size == -1 && errno == EAGAIN) {
    } else {
      mu_.lock();
      shutdown(fd_, SHUT_RD);
      read_closed_ = true;
      bool result = CloseIfNeeded();
      mu_.unlock();
      if (recv_size == 0) {
        RecvClosed(absl::OkStatus());
      } else {
        RecvClosed(absl::InternalError(
            absl::StrFormat("%ld = recv() failed errno: %d err: %s", recv_size,
                            errno, strerror(errno))));
      }
      return result;
    }
  }
  if (events.revents & POLLOUT) {
    can_send_ = true;
  }
  if (can_send_ && !write_closed_) {
    mu_.lock();
    while (!frames_.empty() && can_send_) {
      auto& packet_to_send = frames_.front();
      if (packet_to_send.empty()) {
        shutdown(fd_, SHUT_WR);
        write_closed_ = true;
        bool result = CloseIfNeeded();
        mu_.unlock();
        SendClosed(absl::OkStatus());
        return result;
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
      } else if (send_size < 0 && errno == EAGAIN) {
        can_send_ = false;
      } else {
        shutdown(fd_, SHUT_WR);
        write_closed_ = true;
        bool result = CloseIfNeeded();
        mu_.unlock();
        if (send_size == 0) {
          SendClosed(absl::OkStatus());
        } else {
          SendClosed(absl::InternalError(
              absl::StrFormat("%ld = send() failed errno: %d err: %s",
                              send_size, errno, strerror(errno))));
        }
        return result;
      }
    }
    mu_.unlock();
  }
  if ((events.revents & POLLHUP) && !write_closed_) {
    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(fd_, SOL_SOCKET, SO_ERROR, &error, &len) != 0) {
      error = errno;
    }

    mu_.lock();
    write_closed_ = true;
    bool result = CloseIfNeeded();
    mu_.unlock();
    if (error == 0) {
      SendClosed(absl::OkStatus());
    } else {
      SendClosed(absl::InternalError(absl::StrFormat(
          "fd failed with hup: errno: %d err: %s", error, strerror(error))));
    }
    return result;
  }
  return true;
}

bool SocketFdPacketState::SendRawFrame(std::string opacket) {
  bool should_send_wake = false;
  {
    absl::MutexLock l(mu_);
    // Allow buffering only before connect.
    if (is_connected_ && write_closed_) {
      return false;
    }
    should_send_wake = frames_.empty() && fd_ != -1;
    frames_.push_back(std::move(opacket));
  }
  if (should_send_wake) {
    loop()->SendWake(this);
  }
  return true;
}

void SocketFdPacketState::RegisterFd(int fd, bool start_connected) {
  {
    absl::MutexLock l(mu_);
    fd_ = fd;
    is_connected_ = start_connected;
    read_closed_ = false;
    write_closed_ = false;
  }
  Register();
}

void SocketFdPacketState::Shutdown(int how) {
  absl::MutexLock l(mu_);
  shutdown(fd_, how);
}

bool SocketFdPacketState::CloseIfNeeded() {
  if (is_connected_ && write_closed_) {
    frames_.clear();
  }
  if (write_closed_) {
    shutdown(fd_, SHUT_WR);
  }
  if (read_closed_) {
    shutdown(fd_, SHUT_RD);
  }
  bool result = !read_closed_ || !write_closed_;
  if (!result) {
    close(fd_);
    fd_ = -1;
  }
  return result;
}

}  // namespace aux
