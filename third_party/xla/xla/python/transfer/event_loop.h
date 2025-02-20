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
#ifndef XLA_PYTHON_TRANSFER_EVENT_LOOP_H_
#define XLA_PYTHON_TRANSFER_EVENT_LOOP_H_

#include <netinet/in.h>
#include <poll.h>

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

// socket.h in conda sysroot include directory does not define
// SO_ZEROCOPY and SO_EE_ORIGIN_ZEROCOPY that were introduced in a
// newer kernel version compared to one used by the conda sysroot, see
// openxla/xla#22083.
#ifndef SO_ZEROCOPY
#define SO_ZEROCOPY 60
#endif
#ifndef SO_EE_ORIGIN_ZEROCOPY
#define SO_EE_ORIGIN_ZEROCOPY 5
#endif

namespace aux {

// Basic event loop using poll().
class PollEventLoop {
 public:
  virtual ~PollEventLoop() = default;

  // Default global event loop.
  static PollEventLoop* GetDefault();

  class Handler {
   public:
    virtual ~Handler() = default;

    // On poll, populate events with the current requested events.
    virtual void PopulatePollInfo(pollfd& events) = 0;

    // If events is nonzero, handle events.
    virtual bool HandleEvents(const pollfd& events) = 0;

    // Registers the handler with an event loop.
    void Register(PollEventLoop* loop = GetDefault());

    // The loop this handler is registered on.
    PollEventLoop* loop() { return loop_; }

   private:
    PollEventLoop* loop_ = nullptr;
  };

  // Run callback on the event loop.
  virtual void Schedule(absl::AnyInvocable<void() &&> cb) = 0;

  // Notifies the EventLoop to call HandleEvents with a spurious wake.
  virtual void SendWake(Handler* handler) = 0;

  // Run callback on the event loop at some point in the future.
  virtual void ScheduleAt(absl::Time time,
                          absl::AnyInvocable<void() &&> cb) = 0;

 private:
  // Implementation detail of Handler::Register.
  virtual void RegisterHandler(Handler* handler) = 0;
};

// Basic ipv6 + ipv4 socket address.
class SocketAddress {
 public:
  SocketAddress();
  explicit SocketAddress(const sockaddr_in& saddr);
  explicit SocketAddress(const sockaddr_in6& saddr);

  // Fetch address.
  const sockaddr& address() const { return saddr_; }
  const sockaddr_in6& address_ipv6() const { return saddr6_; }
  const sockaddr_in& address_ipv4() const { return saddr4_; }

  // To String (parsable with Parse).
  std::string ToString() const;

  // Inverse of ToString().
  static absl::StatusOr<SocketAddress> Parse(const std::string& addr);

 private:
  static int Parse(const std::string& addr, SocketAddress& out);

  union {
    sockaddr saddr_;
    sockaddr_in saddr4_;
    sockaddr_in6 saddr6_;
  };
};

// Calls accept() on sockets.
class SocketListener {
 public:
  ~SocketListener();

  const SocketAddress& addr() const { return addr_; }

  static absl::StatusOr<std::unique_ptr<SocketListener>> Listen(
      const SocketAddress& addr,
      absl::AnyInvocable<void(int socket_fd, const SocketAddress& addr)>
          on_accept,
      int accept_flags = 0);

  void Start();

 private:
  class Handler;
  SocketAddress addr_;
  bool started_ = false;
  Handler* handler_ = nullptr;
};

}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_EVENT_LOOP_H_
