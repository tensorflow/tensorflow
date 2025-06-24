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
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

#include <atomic>
#include <cerrno>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tsl/platform/env.h"

namespace aux {

class PollEventLoopImpl : public PollEventLoop {
 public:
  PollEventLoopImpl() {
    thread_ = std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
        {}, "event-loop-thread", [this]() { Run(); }));
  }

  void RegisterHandler(Handler* handler) override {
    absl::MutexLock l(&mu_);
    inserts_.push_back(handler);
    WakeInternal();
  }
  void SendWake(Handler* handler) override {
    absl::MutexLock l(&mu_);
    wakes_.insert(handler);
    WakeInternal();
  }

  void Schedule(absl::AnyInvocable<void() &&> cb) override {
    absl::MutexLock l(&mu_);
    cbs_.push_back(std::move(cb));
    WakeInternal();
  }

  void ScheduleAt(absl::Time t, absl::AnyInvocable<void() &&> cb) override {
    absl::MutexLock l(&mu_);
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
        absl::MutexLock l(&mu_);
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
  Handler(int fd, int accept_flags,
          absl::AnyInvocable<void(int socket_fd, const SocketAddress& addr)>
              on_accept)
      : on_accept_(std::move(on_accept)),
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
    on_accept_(*cfd_or, recv_addr);
    return true;
  }

  void Shutdown() {
    auto* l = loop();
    shutdown_requested_.store(true);
    l->SendWake(this);
  }

  absl::StatusOr<int> Accept(SocketAddress& recv_addr) {
    socklen_t addr_len = sizeof(recv_addr);
    int cfd = accept4(fd_, reinterpret_cast<sockaddr*>(&recv_addr), &addr_len,
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
  absl::AnyInvocable<void(int socket_fd, const SocketAddress& addr)> on_accept_;
  std::atomic<bool> shutdown_requested_{false};
  int fd_;
  int accept_flags_;
};

absl::StatusOr<std::unique_ptr<SocketListener>> SocketListener::Listen(
    const SocketAddress& addr,
    absl::AnyInvocable<void(int socket_fd, const SocketAddress& addr)>
        on_accept,
    int accept_flags) {
  std::unique_ptr<SocketListener> result(new SocketListener());
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
  if (bind(sfd, reinterpret_cast<const struct sockaddr*>(&addr.address()),
           addr.address().sa_family == AF_INET6 ? sizeof(sockaddr_in6)
                                                : sizeof(sockaddr_in)) != 0) {
    return absl::ErrnoToStatus(errno, "bind");
  }
  if (listen(sfd, 1024) != 0) {
    return absl::ErrnoToStatus(errno, "listen");
  }
  result->handler_ =
      new SocketListener::Handler(sfd, accept_flags, std::move(on_accept));
  if (addr.address().sa_family == AF_INET6) {
    sockaddr_in6 new_sock_name = addr.address_ipv6();
    sockaddr_in6 new_sock_name2;
    socklen_t addr_len = sizeof(new_sock_name2);
    if (getsockname(sfd, reinterpret_cast<struct sockaddr*>(&new_sock_name2),
                    &addr_len) != 0) {
      return absl::ErrnoToStatus(errno, "getsockname");
    }
    new_sock_name.sin6_port = new_sock_name2.sin6_port;
    result->addr_ = SocketAddress(new_sock_name);
  } else {
    sockaddr_in new_sock_name = addr.address_ipv4();
    sockaddr_in new_sock_name2;
    socklen_t addr_len = sizeof(new_sock_name2);
    if (getsockname(sfd, reinterpret_cast<struct sockaddr*>(&new_sock_name2),
                    &addr_len) != 0) {
      return absl::ErrnoToStatus(errno, "getsockname");
    }
    new_sock_name.sin_port = new_sock_name2.sin_port;
    result->addr_ = SocketAddress(new_sock_name);
  }
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
  memset(this, 0, sizeof(SocketAddress));
  saddr6_.sin6_family = AF_INET6;
}

SocketAddress::SocketAddress(const sockaddr_in& saddr) {
  memcpy(&saddr4_, &saddr, sizeof(saddr));
}

SocketAddress::SocketAddress(const sockaddr_in6& saddr) {
  memcpy(&saddr6_, &saddr, sizeof(saddr));
}

std::string SocketAddress::ToString() const {
  if (saddr_.sa_family == AF_INET6) {
    char tmp[INET6_ADDRSTRLEN + 16];
    tmp[0] = '[';
    inet_ntop(AF_INET6, &saddr6_.sin6_addr, &tmp[1], sizeof(tmp) - 1);
    int pos = strlen(&tmp[0]);
    pos += snprintf(&tmp[pos], sizeof(tmp) - pos, "]:%d",
                    ntohs(saddr6_.sin6_port));
    return std::string(tmp, pos);
  } else if (saddr_.sa_family == AF_INET) {
    char tmp[INET_ADDRSTRLEN + 16];
    inet_ntop(AF_INET, &saddr4_.sin_addr, &tmp[0], sizeof(tmp) - 1);
    int pos = strlen(&tmp[0]);
    pos +=
        snprintf(&tmp[pos], sizeof(tmp) - pos, ":%d", ntohs(saddr4_.sin_port));
    return std::string(tmp, pos);
  } else {
    LOG(FATAL) << "Invalid IPAddress";
  }
}

int ParsePort(const std::string& addr, size_t it, uint32_t& parsed_port) {
  size_t port_pos = addr.find(':', it);
  if (port_pos == std::string::npos) {
    return -1;
  }
  for (size_t i = port_pos + 1; i < addr.size(); ++i) {
    if (!(addr[i] >= '0' && addr[i] <= '9')) {
      return -1;
    }
    parsed_port = parsed_port * 10 + (addr[i] - '0');
    if (parsed_port >= 65536) {
      return -1;
    }
  }
  return 0;
}

int SocketAddress::Parse(const std::string& addr, SocketAddress& out) {
  memset(&out, 0, sizeof(SocketAddress));
  if (!addr.empty() && addr.data()[0] == '[') {
    size_t it = addr.find(']');
    if (it == std::string::npos) {
      return -1;
    }
    if (it - 1 >= INET6_ADDRSTRLEN) {
      return -1;
    }
    char tmp[INET6_ADDRSTRLEN];
    uint32_t parsed_port = 0;
    if (ParsePort(addr, it, parsed_port) != 0) {
      return -1;
    }
    out.saddr6_.sin6_family = AF_INET6;
    out.saddr6_.sin6_port = htons(static_cast<uint16_t>(parsed_port));
    memcpy(&tmp[0], &addr.data()[1], it - 1);
    tmp[it - 1] = 0;
    return inet_pton(AF_INET6, &tmp[0], &out.saddr6_.sin6_addr);
  } else {
    size_t it = addr.find(':');
    if (it == std::string::npos) {
      return -1;
    }
    if (it >= INET_ADDRSTRLEN) {
      return -1;
    }
    uint32_t parsed_port = 0;
    if (ParsePort(addr, it, parsed_port) != 0) {
      return -1;
    }
    char tmp[INET_ADDRSTRLEN];
    memcpy(&tmp[0], &addr.data()[0], it);
    tmp[it] = 0;
    out.saddr4_.sin_family = AF_INET;
    out.saddr4_.sin_port = htons(static_cast<uint16_t>(parsed_port));
    return inet_pton(AF_INET, &tmp[0], &out.saddr4_.sin_addr);
  }
  return -1;
}

absl::StatusOr<SocketAddress> SocketAddress::Parse(const std::string& addr) {
  SocketAddress out;
  if (Parse(addr, out) == -1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse ip address: ", addr));
  }
  return out;
}

}  // namespace aux
