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

#include "ignite_plain_client.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Mswsock.lib")
#pragma comment(lib, "AdvApi32.lib")

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace ignite {

PlainClient::PlainClient(std::string host, int port)
    : host(host), port(port), sock(INVALID_SOCKET) {}

PlainClient::~PlainClient() {
  if (IsConnected()) {
    tensorflow::Status status = Disconnect();
    if (!status.ok()) LOG(WARNING) << status.ToString();
  }
}

tensorflow::Status PlainClient::Connect() {
  WSADATA wsaData;
  addrinfo *result = NULL, *ptr = NULL, hints;

  int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (res != 0)
    return tensorflow::errors::Internal("WSAStartup failed with error: ", res);

  ZeroMemory(&hints, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  res =
      getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result);
  if (res != 0)
    return tensorflow::errors::Internal("Getaddrinfo failed with error: ", res);

  for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {
    sock = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
    if (sock == INVALID_SOCKET) {
      WSACleanup();
      return tensorflow::errors::Internal("Socket failed with error: ",
                                          WSAGetLastError());
    }

    res = connect(sock, ptr->ai_addr, (int)ptr->ai_addrlen);
    if (res == SOCKET_ERROR) {
      closesocket(sock);
      sock = INVALID_SOCKET;
      continue;
    }

    break;
  }

  freeaddrinfo(result);

  if (sock == INVALID_SOCKET) {
    WSACleanup();
    return tensorflow::errors::Internal("Unable to connect to server");
  }

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" established";

  return tensorflow::Status::OK();
}

tensorflow::Status PlainClient::Disconnect() {
  int res = shutdown(sock, SD_SEND);
  closesocket(sock);
  WSACleanup();

  if (res == SOCKET_ERROR)
    return tensorflow::errors::Internal("Shutdown failed with error: ",
                                        WSAGetLastError());
  else
    return tensorflow::Status::OK();
}

bool PlainClient::IsConnected() { return sock != INVALID_SOCKET; }

int PlainClient::GetSocketDescriptor() { return sock; }

tensorflow::Status PlainClient::ReadData(uint8_t *buf, int32_t length) {
  int recieved = 0;

  while (recieved < length) {
    int res = recv(sock, buf, length - recieved, 0);

    if (res < 0)
      return tensorflow::errors::Internal(
          "Error occured while reading from socket: ", res);

    if (res == 0)
      return tensorflow::errors::Internal("Server closed connection");

    recieved += res;
    buf += res;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status PlainClient::WriteData(uint8_t *buf, int32_t length) {
  int sent = 0;

  while (sent < length) {
    int res = send(sock, buf, length - sent, 0);

    if (res < 0)
      return tensorflow::errors::Internal(
          "Error occured while writing into socket: ", res);

    sent += res;
    buf += res;
  }

  return tensorflow::Status::OK();
}

}  // namespace ignite
