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
#include <sys/socket.h>

#include <deque>
#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
  explicit SocketAddress(const sockaddr_storage& saddr);

  // Fetch address.
  const sockaddr& address() const {
    // NOLINTNEXTLINE
    return *reinterpret_cast<const sockaddr*>(&storage_);
  }
  sockaddr& mutable_address() {
    // NOLINTNEXTLINE
    return *reinterpret_cast<sockaddr*>(&storage_);
  }
  const sockaddr_in6& address_ipv6() const {
    CHECK_EQ(storage_.ss_family, AF_INET6);
    // NOLINTNEXTLINE
    return *reinterpret_cast<const sockaddr_in6*>(&storage_);
  }
  const sockaddr_in& address_ipv4() const {
    CHECK_EQ(storage_.ss_family, AF_INET);
    // NOLINTNEXTLINE
    return *reinterpret_cast<const sockaddr_in*>(&storage_);
  }

  socklen_t len() const;

  // To String (parsable with Parse).
  std::string ToString() const;

  // Inverse of ToString().
  static absl::StatusOr<SocketAddress> Parse(absl::string_view addr);

 private:
  sockaddr_storage storage_;
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

class SocketFdPacketState : public PollEventLoop::Handler {
 public:
  // Must be closed and cleared properly before destruction.
  ~SocketFdPacketState() override;

  // Subclasses must handle the incoming packet entirely during this call
  // (or else copy).
  virtual void HandlePacket(absl::string_view packet) = 0;

  // All of these may destroy the handler if both directions are closed.
  virtual void ConnectFailed() = 0;
  // Clean half close.
  virtual void RecvClosed(absl::Status error) = 0;
  // Clean half close
  virtual void SendClosed(absl::Status error) = 0;

  // Schedules the frame (returns false if send is closed).
  bool SendRawFrame(std::string opacket);

  // Starts listening for fd.
  // ConnectFailed() only called if start_connected=false.
  void RegisterFd(int fd, bool start_connected);

  // Calls shutdown on the fd.
  void Shutdown(int how);

 private:
  void PopulatePollInfo(pollfd& events) override;

  bool HandleEvents(const pollfd& events) override;

  bool CloseIfNeeded() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex mu_;
  int fd_ = -1;
  bool can_send_ = false;
  bool is_connected_ = false;
  bool read_closed_ = false;
  bool write_closed_ = false;
  size_t write_offset_ = 0;
  size_t recv_count_ = 0;
  std::unique_ptr<char[]> network_buffer_ =
      std::unique_ptr<char[]>(new char[4096]);
  std::deque<std::string> frames_ ABSL_GUARDED_BY(mu_);
};

}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_EVENT_LOOP_H_
