/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/net.h"

#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/platform.h"

#if !defined(PLATFORM_WINDOWS)
#include <errno.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#endif  // PLATFORM_WINDOWS

namespace tsl {
namespace net {
#if !defined(PLATFORM_WINDOWS)
namespace {

bool CanBindAndListen(int port, int family, int socktype) {
  int protocol = (socktype == SOCK_STREAM) ? IPPROTO_TCP : 0;
  int sock = socket(family, socktype, protocol);
  if (sock < 0) {
    // If we can't create a socket for this family, maybe it's not supported.
    if (errno == EAFNOSUPPORT || errno == EPROTONOSUPPORT) {
      LOG(INFO) << "Skipping bind test for family " << family
                << ": socket() failed with errno=" << errno;
      return true;
    }
    LOG(ERROR) << "socket(" << family << ", " << socktype << ", " << protocol
               << ") failed: " << strerror(errno);
    return false;
  }

  struct sockaddr_storage addr_storage;
  socklen_t addr_len;
  if (family == AF_INET6) {
    struct sockaddr_in6* addr =
        reinterpret_cast<struct sockaddr_in6*>(&addr_storage);
    addr_len = sizeof(*addr);
    memset(addr, 0, addr_len);
    addr->sin6_family = AF_INET6;
    addr->sin6_addr = in6addr_any;
    addr->sin6_port = htons(port);
  } else {
    struct sockaddr_in* addr =
        reinterpret_cast<struct sockaddr_in*>(&addr_storage);
    addr_len = sizeof(*addr);
    memset(addr, 0, addr_len);
    addr->sin_family = AF_INET;
    addr->sin_addr.s_addr = INADDR_ANY;
    addr->sin_port = htons(port);
  }

  // Use SO_REUSEADDR like IsPortAvailable does.
  int one = 1;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
    LOG(ERROR) << "setsockopt(SO_REUSEADDR) failed: " << strerror(errno);
    close(sock);
    return false;
  }

  if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr_storage), addr_len) !=
      0) {
    LOG(ERROR) << "bind(" << port << ") failed for family " << family << ": "
               << strerror(errno);
    close(sock);
    return false;
  }

  if (socktype == SOCK_STREAM) {
    if (listen(sock, 1) != 0) {
      LOG(ERROR) << "listen(" << port << ") failed for family " << family
                 << ": " << strerror(errno);
      close(sock);
      return false;
    }
  }

  close(sock);
  return true;
}

}  // namespace
#endif  // PLATFORM_WINDOWS

TEST(Net, PickUnusedPortOrDie) {
  int port0 = PickUnusedPortOrDie();
  int port1 = PickUnusedPortOrDie();
  CHECK_GE(port0, 0);
  CHECK_LT(port0, 65536);
  CHECK_GE(port1, 0);
  CHECK_LT(port1, 65536);
  CHECK_NE(port0, port1);
  RecycleUnusedPort(port0);
  RecycleUnusedPort(port1);
}

#if !defined(PLATFORM_WINDOWS)
TEST(Net, PickedPortIsBindable) {
  int port = PickUnusedPortOrDie();
  ASSERT_GT(port, 0);
  EXPECT_TRUE(CanBindAndListen(port, AF_INET, SOCK_STREAM));
  EXPECT_TRUE(CanBindAndListen(port, AF_INET6, SOCK_STREAM));
  EXPECT_TRUE(CanBindAndListen(port, AF_INET, SOCK_DGRAM));
  EXPECT_TRUE(CanBindAndListen(port, AF_INET6, SOCK_DGRAM));
  RecycleUnusedPort(port);
}
#endif  // PLATFORM_WINDOWS

TEST(Net, RecycleUnusedPort) {
  for (int i = 0; i < 1000; ++i) {
    int port0 = PickUnusedPortOrDie();
    CHECK_GE(port0, 0);
    CHECK_LT(port0, 65536);
    RecycleUnusedPort(port0);
  }
}

TEST(Net, RecycleUnusedPortTwiceShallFail) {
  int port0 = PickUnusedPortOrDie();
  CHECK_GE(port0, 0);
  CHECK_LT(port0, 65536);
  RecycleUnusedPort(port0);

  EXPECT_DEATH(RecycleUnusedPort(port0), "");
}

TEST(Net, RecycleUnusedPortNegativeShallFail) {
  EXPECT_DEATH(RecycleUnusedPort(-1), "");
}

}  // namespace net
}  // namespace tsl
