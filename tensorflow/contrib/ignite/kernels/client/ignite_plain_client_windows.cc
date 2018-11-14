/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/ignite/kernels/client/ignite_plain_client.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Mswsock.lib")
#pragma comment(lib, "AdvApi32.lib")

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

PlainClient::PlainClient(string host, int port, bool big_endian)
    : Client(big_endian),
      host_(std::move(host)),
      port_(port),
      sock_(INVALID_SOCKET) {}

PlainClient::~PlainClient() {
  if (IsConnected()) {
    Status status = Disconnect();
    if (!status.ok()) LOG(WARNING) << status.ToString();
  }
}

Status PlainClient::Connect() {
  WSADATA wsaData;
  addrinfo *result = NULL, *ptr = NULL, hints;

  int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (res != 0) return errors::Internal("WSAStartup failed with error: ", res);

  ZeroMemory(&hints, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  res = getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints,
                    &result);
  if (res != 0) return errors::Internal("Getaddrinfo failed with error: ", res);

  auto clean = gtl::MakeCleanup([result] { freeaddrinfo(result); });

  for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {
    sock_ = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
    if (sock_ == INVALID_SOCKET) {
      WSACleanup();
      return errors::Internal("Socket failed with error: ", WSAGetLastError());
    }

    res = connect(sock_, ptr->ai_addr, (int)ptr->ai_addrlen);
    if (res == SOCKET_ERROR) {
      closesocket(sock_);
      sock_ = INVALID_SOCKET;
      continue;
    }

    break;
  }

  if (sock_ == INVALID_SOCKET) {
    WSACleanup();
    return errors::Internal("Unable to connect to server");
  }

  LOG(INFO) << "Connection to \"" << host_ << ":" << port_ << "\" established";

  return Status::OK();
}

Status PlainClient::Disconnect() {
  int res = shutdown(sock_, SD_SEND);
  closesocket(sock_);
  WSACleanup();

  if (res == SOCKET_ERROR)
    return errors::Internal("Shutdown failed with error: ", WSAGetLastError());
  else
    return Status::OK();
}

bool PlainClient::IsConnected() { return sock_ != INVALID_SOCKET; }

int PlainClient::GetSocketDescriptor() { return sock_; }

Status PlainClient::ReadData(uint8_t *buf, const int32_t length) {
  int received = 0;

  while (received < length) {
    int res = recv(sock_, (char *)buf, length - received, 0);

    if (res < 0)
      return errors::Internal("Error occurred while reading from socket: ",
                              res);

    if (res == 0) return errors::Internal("Server closed connection");

    received += res;
    buf += res;
  }

  return Status::OK();
}

Status PlainClient::WriteData(const uint8_t *buf, const int32_t length) {
  int sent = 0;

  while (sent < length) {
    int res = send(sock_, (char *)buf, length - sent, 0);

    if (res < 0)
      return errors::Internal("Error occurred while writing into socket: ",
                              res);

    sent += res;
    buf += res;
  }

  return Status::OK();
}

}  // namespace tensorflow
