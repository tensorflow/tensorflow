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

#include <errno.h>
#include <string.h>
#include <sys/socket.h>

#include <atomic>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tsl/platform/statusor.h"

namespace aux {
namespace {

class BasicEchoFdPoll : public PollEventLoop::Handler {
 public:
  BasicEchoFdPoll(int fd, absl::Notification& done_notify)
      : fd_(fd), done_notify_(&done_notify) {}
  ~BasicEchoFdPoll() override {
    close(fd_);
    done_notify_->Notify();
  }

  void PopulatePollInfo(pollfd& events) override {
    events.fd = fd_;
    events.events = POLLIN;
  }
  bool HandleEvents(const pollfd& events) override {
    if (needs_delete_.load()) {
      delete this;
      return false;
    }
    char tmp[20];
    ssize_t read_len = read(fd_, &tmp, 20);
    CHECK_NE(read_len, -1) << strerror(errno) << " " << errno;
    CHECK_EQ(read_len, write(fd_, &tmp, read_len))
        << strerror(errno) << " " << errno;
    return true;
  }

  void ScheduleDelete() {
    auto* l = loop();
    needs_delete_.store(true);
    l->SendWake(this);
  }

 private:
  std::atomic<bool> needs_delete_{false};
  int fd_;
  absl::Notification* done_notify_;
};

TEST(EventLoopTest, TestBasic) {
  int fd[2];
  ASSERT_NE(-1, socketpair(PF_LOCAL, SOCK_STREAM, 0, fd))
      << strerror(errno) << " " << errno;

  absl::Notification done_notify;

  auto* handler = (new BasicEchoFdPoll(fd[1], done_notify));
  handler->Register();

  std::string example_data = "secret_message";
  ASSERT_EQ(write(fd[0], example_data.data(), example_data.size()),
            example_data.size());
  char tmp[20];
  ssize_t read_len = read(fd[0], &tmp, 20);
  ASSERT_EQ(read_len, example_data.size());
  ASSERT_EQ(absl::string_view(&tmp[0], read_len), example_data);
  handler->ScheduleDelete();
  done_notify.WaitForNotification();
}

TEST(EventLoopTest, TestSocketListen) {
  sockaddr_in6 addr;
  memset(&addr, 0, sizeof(sockaddr_in6));
  addr.sin6_family = AF_INET6;

  TF_ASSERT_OK_AND_ASSIGN(
      auto listener,
      SocketListener::Listen(
          SocketAddress(addr), [](int sockfd, const SocketAddress& addr) {
            char msg2[128];
            auto l = recv(sockfd, msg2, 128, 0);
            CHECK_GE(l, 0) << absl::ErrnoToStatus(errno, "recv");
            send(sockfd, msg2, l, 0);
          }));
  listener->Start();

  auto other_addr = SocketAddress::Parse(listener->addr().ToString()).value();
  int cfd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, 0);
  connect(cfd, &other_addr.address(), sizeof(sockaddr_in6));
  std::string msg = "secret";
  ASSERT_EQ(send(cfd, msg.data(), msg.size(), 0), msg.size())
      << absl::ErrnoToStatus(errno, "send");
  char msg2[128];
  ASSERT_EQ(recv(cfd, msg2, msg.size(), 0), msg.size())
      << absl::ErrnoToStatus(errno, "recv");
  ASSERT_EQ(absl::string_view(msg2, msg.size()), msg);
}

TEST(EventLoopTest, TestSocketListenIPV4) {
  sockaddr_in addr;
  memset(&addr, 0, sizeof(sockaddr_in));
  addr.sin_family = AF_INET;

  TF_ASSERT_OK_AND_ASSIGN(
      auto listener,
      SocketListener::Listen(
          SocketAddress(addr), [](int sockfd, const SocketAddress& addr) {
            char msg2[128];
            auto l = recv(sockfd, msg2, 128, 0);
            CHECK_GE(l, 0) << absl::ErrnoToStatus(errno, "recv");
            send(sockfd, msg2, l, 0);
          }));
  listener->Start();

  auto other_addr = SocketAddress::Parse(listener->addr().ToString()).value();
  LOG(INFO) << "Listening on: " << listener->addr().ToString();
  int cfd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  connect(cfd, &other_addr.address(), sizeof(sockaddr_in));
  std::string msg = "secret";
  ASSERT_EQ(send(cfd, msg.data(), msg.size(), 0), msg.size())
      << absl::ErrnoToStatus(errno, "send");
  char msg2[128];
  ASSERT_EQ(recv(cfd, msg2, msg.size(), 0), msg.size())
      << absl::ErrnoToStatus(errno, "recv");
  ASSERT_EQ(absl::string_view(msg2, msg.size()), msg);
}

TEST(EventLoopTest, TestSchedule) {
  absl::Notification done_notify;
  PollEventLoop::GetDefault()->Schedule(
      [&done_notify]() { done_notify.Notify(); });
  done_notify.WaitForNotification();
}

TEST(EventLoopTest, TestScheduleAt) {
  absl::Notification done_notify;
  auto wake_time = absl::Now() + absl::Seconds(2);
  PollEventLoop::GetDefault()->ScheduleAt(
      wake_time, [&done_notify]() { done_notify.Notify(); });
  done_notify.WaitForNotification();
  ASSERT_GE(absl::Now(), wake_time);
}

}  // namespace
}  // namespace aux
