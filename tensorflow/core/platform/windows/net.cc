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

#include "tensorflow/core/platform/net.h"

#include <sys/types.h>
#include <winsock2.h>

#include <cstdlib>
#include <unordered_set>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/windows/error_windows.h"

#undef ERROR

namespace tensorflow {
namespace internal {

namespace {

bool IsPortAvailable(int* port, bool is_tcp) {
  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  SOCKET sock = socket(AF_INET, is_tcp ? SOCK_STREAM : SOCK_DGRAM, protocol);

  struct sockaddr_in addr;
  int addr_len = static_cast<int>(sizeof(addr));
  int actual_port;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, 65535);
  if (sock == INVALID_SOCKET) {
    LOG(ERROR) << "socket() failed: " << WindowsWSAGetLastErrorMessage();
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exits.
  const int one = 1;
  int result = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                          reinterpret_cast<const char*>(&one), sizeof(one));
  if (result == SOCKET_ERROR) {
    LOG(ERROR) << "setsockopt() failed: " << WindowsWSAGetLastErrorMessage();
    closesocket(sock);
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)*port);
  result = bind(sock, (struct sockaddr*)&addr, sizeof(addr));
  if (result == SOCKET_ERROR) {
    LOG(WARNING) << "bind(port=" << *port
                 << ") failed: " << WindowsWSAGetLastErrorMessage();
    closesocket(sock);
    return false;
  }

  // Get the bound port number.
  result = getsockname(sock, (struct sockaddr*)&addr, &addr_len);
  if (result == SOCKET_ERROR) {
    LOG(WARNING) << "getsockname() failed: " << WindowsWSAGetLastErrorMessage();
    closesocket(sock);
    return false;
  }

  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);
  if (*port == 0) {
    *port = actual_port;
  } else {
    CHECK_EQ(*port, actual_port);
  }

  closesocket(sock);
  return true;
}

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

}  // namespace

int PickUnusedPortOrDie() {
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != NO_ERROR) {
    LOG(ERROR) << "Error at WSAStartup()";
    return false;
  }

  static std::unordered_set<int> chosen_ports;

  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  while (true) {
    int port;
    trial++;
    CHECK_LE(trial, kMaximumTrials)
        << "Failed to pick an unused port for testing.";
    if (trial == 1) {
      port = GetCurrentProcessId() % (65536 - 30000) + 30000;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rand() % (65536 - 30000) + 30000;
    } else {
      port = 0;
    }

    if (chosen_ports.find(port) != chosen_ports.end()) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp)) {
      continue;
    }

    CHECK_GT(port, 0);
    if (!IsPortAvailable(&port, !is_tcp)) {
      is_tcp = !is_tcp;
      continue;
    }

    chosen_ports.insert(port);
    WSACleanup();
    return port;
  }

  return 0;
}

}  // namespace internal
}  // namespace tensorflow
