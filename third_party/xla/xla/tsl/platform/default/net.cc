/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>

#include "absl/base/thread_annotations.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/logging.h"

// https://en.wikipedia.org/wiki/Ephemeral_port
#define MAX_EPHEMERAL_PORT 60999
#define MIN_EPHEMERAL_PORT 32768

namespace tsl {
namespace net {

bool IsPortAvailable(int* port, bool is_tcp, std::string* error) {
  const int socktype = is_tcp ? SOCK_STREAM : SOCK_DGRAM;
  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  bool got_socket = false;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, MAX_EPHEMERAL_PORT);

  for (int family : {AF_INET6, AF_INET}) {
    const int fd = socket(family, socktype, protocol);
    if (fd < 0) {
      continue;
    }
    got_socket = true;

    // SO_REUSEADDR lets us start up a server immediately after it exists.
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
      if (error) {
        *error = absl::StrCat("setsockopt: ", strerror(errno));
      } else {
        PLOG(ERROR) << "setsockopt() failed";
      }
      if (close(fd) < 0) {
        PLOG(ERROR) << "close() failed";
      }
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
      addr->sin6_port = htons(static_cast<uint16_t>(*port));
    } else {
      struct sockaddr_in* addr =
          reinterpret_cast<struct sockaddr_in*>(&addr_storage);
      addr_len = sizeof(*addr);
      memset(addr, 0, addr_len);
      addr->sin_family = AF_INET;
      addr->sin_addr.s_addr = INADDR_ANY;
      addr->sin_port = htons(static_cast<uint16_t>(*port));
    }

    if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr_storage), addr_len) <
        0) {
      if (error) {
        *error =
            absl::StrCat("bind(port=", *port, ") failed: ", strerror(errno));
      } else {
        PLOG(WARNING) << "bind(port=" << *port << ") failed";
      }
      if (close(fd) < 0) {
        PLOG(ERROR) << "close() failed";
      }
      return false;
    }

    // Get the bound port number.
    struct sockaddr_storage bound_addr_storage;
    socklen_t bound_addr_len = sizeof(bound_addr_storage);
    if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&bound_addr_storage),
                    &bound_addr_len) < 0) {
      if (error) {
        *error = absl::StrCat("getsockname failed: ", strerror(errno));
      } else {
        PLOG(WARNING) << "getsockname() failed";
      }
      if (close(fd) < 0) {
        PLOG(ERROR) << "close() failed";
      }
      return false;
    }
    int actual_port = -1;
    if (bound_addr_storage.ss_family == AF_INET6) {
      actual_port =
          ntohs(reinterpret_cast<struct sockaddr_in6*>(&bound_addr_storage)
                    ->sin6_port);
    } else if (bound_addr_storage.ss_family == AF_INET) {
      actual_port = ntohs(
          reinterpret_cast<struct sockaddr_in*>(&bound_addr_storage)->sin_port);
    } else {
      if (error) {
        *error = absl::StrCat("getsockname returned unknown family: ",
                              bound_addr_storage.ss_family);
      } else {
        LOG(ERROR) << "getsockname() returned unknown family: "
                   << bound_addr_storage.ss_family;
      }
      if (close(fd) < 0) {
        PLOG(ERROR) << "close() failed";
      }
      return false;
    }

    CHECK_GT(actual_port, 0);
    if (*port == 0) {
      *port = actual_port;
    } else {
      CHECK_EQ(*port, actual_port);
    }

    if (is_tcp) {
      if (listen(fd, 1) < 0) {
        if (error) {
          *error = absl::StrCat("listen failed: ", strerror(errno));
        } else {
          PLOG(ERROR) << "listen() failed";
        }
        close(fd);
        return false;
      }
    }

    if (close(fd) < 0) {
      PLOG(ERROR) << "close() failed";
    }
  }

  if (!got_socket) {
    if (error) {
      *error = absl::StrCat("socket failed for all address families: ",
                            strerror(errno));
    } else {
      PLOG(ERROR) << "socket() failed for all address families";
    }
    return false;
  }

  return true;
}

namespace {
// Manages the set of ports that have been chosen by PickUnusedPort().
// This class is a singleton and is thread-safe.
class ChosenPorts {
 public:
  static ChosenPorts& GetChosenPorts() {
    static ChosenPorts chosen_ports;
    return chosen_ports;
  }

  // Returns true if the port is in the chosen set.
  bool Contains(int port) {
    absl::MutexLock l(mu_);
    return ports_.count(port) > 0;
  }

  // If the port is not in the chosen set, inserts it and returns true.
  // Otherwise, returns false.
  bool Insert(int port) {
    absl::MutexLock l(mu_);
    return ports_.insert(port).second;
  }

  // Erases the port from the chosen set. Returns true if the port was present.
  bool Erase(int port) {
    absl::MutexLock l(mu_);
    return ports_.erase(port) > 0;
  }

 private:
  ChosenPorts() = default;
  absl::Mutex mu_;
  std::unordered_set<int> ports_ ABSL_GUARDED_BY(mu_);
};

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

}  // namespace

int PickUnusedPortOrDie() {
  int port = PickUnusedPort();
  CHECK_GT(port, 0) << "PickUnusedPort() failed";
  return port;
}

int PickUnusedPort() {
  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  std::default_random_engine rgen(std::random_device{}());
  std::uniform_int_distribution<int> rdist(MIN_EPHEMERAL_PORT,
                                           MAX_EPHEMERAL_PORT - 1);
  while (true) {
    int port;
    trial++;
    if (trial > kMaximumTrials) {
      LOG(ERROR) << "Failed to pick an unused port for testing.";
      return -1;
    }
    if (trial == 1) {
      port = getpid() % (MAX_EPHEMERAL_PORT - MIN_EPHEMERAL_PORT) +
             MIN_EPHEMERAL_PORT;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rdist(rgen);
    } else {
      port = 0;
    }

    if (ChosenPorts::GetChosenPorts().Contains(port)) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp, nullptr)) {
      continue;
    }
    if (port <= 0) {
      return -1;
    }
    if (!IsPortAvailable(&port, !is_tcp, nullptr)) {
      is_tcp = !is_tcp;
      continue;
    }

    if (ChosenPorts::GetChosenPorts().Insert(port)) {
      return port;
    }
  }

  return -1;
}

void RecycleUnusedPort(int port) {
  if (port <= 0 || !ChosenPorts::GetChosenPorts().Erase(port)) {
    LOG(FATAL)
        << "Port " << port
        << " is not a valid port to be recycled. It must be a positive "
           "number that was previously returned by PickUnusedPort[OrDie](), "
           "and not yet recycled.";
  }
}
}  // namespace net
}  // namespace tsl
