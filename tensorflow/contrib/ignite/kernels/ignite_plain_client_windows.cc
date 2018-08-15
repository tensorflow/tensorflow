/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ignite {

PlainClient::PlainClient(std::string host, int port) :
  host(host),
  port(port),
  sock(INVALID_SOCKET) {}

PlainClient::~PlainClient() {
  if (IsConnected()) {
    tensorflow::Status status = Disconnect();
    if (!status.ok())
      LOG(WARNING) << status.ToString();
  }
}

tensorflow::Status PlainClient::Connect() {
  WSADATA wsaData;
  struct addrinfo *result = NULL, *ptr = NULL, hints;

  int res = WSAStartup(MAKEWORD(2,2), &wsaData);
  if (res != 0)
    return tensorflow::errors::Internal("WSAStartup failed with error: ", res);

  ZeroMemory(&hints, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  
  res = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result);
  if (res != 0)
    return tensorflow::errors::Internal("Getaddrinfo failed with error: ", res);

  for(ptr=result; ptr != NULL ;ptr=ptr->ai_next) {
    sock = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
    if (sock == INVALID_SOCKET) {
      WSACleanup();
      return tensorflow::errors::Internal("Socket failed with error: ", WSAGetLastError());
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
    return tensorflow::errors::Internal("Shutdown failed with error: ", WSAGetLastError());
  else
    return tensorflow::Status::OK();
}

bool PlainClient::IsConnected() {
  return sock != INVALID_SOCKET;
}

int PlainClient::GetSocketDescriptor() {
  return sock;
}

char PlainClient::ReadByte() {
  char res;
  ReadData((char*) &res, 1);

  return res;
}

short PlainClient::ReadShort() {
  short res;
  ReadData((char*) &res, 2);

  return res;
}

int PlainClient::ReadInt() {
  int res;
  ReadData((char*) &res, 4);

  return res;
}

long PlainClient::ReadLong() {
  long res;
  ReadData((char*) &res, 8);

  return res;
}

void PlainClient::ReadData(char *buf, int length) {
  int recieved = 0;

  while (recieved < length) {
    int res = recv(sock, buf, length - recieved, 0);

    if (res < 0) {
      LOG(WARNING) << "Error occured while reading from socket: " << res;
      break;
    }

    recieved += res;
    buf += res;
  }
}

void PlainClient::WriteByte(char data) { WriteData((char*) &data, 1); }

void PlainClient::WriteShort(short data) { WriteData((char*) &data, 2); }

void PlainClient::WriteInt(int data) { WriteData((char*) &data, 4); }

void PlainClient::WriteLong(long data) { WriteData((char*) &data, 8); }

void PlainClient::WriteData(char *buf, int length) { 
  int sent = 0;

  while (sent < length) {
    int res = send(sock, buf, length - sent, 0);

    if (res < 0) {
      LOG(WARNING) << "Error occured while reading from socket: " << res;
      break;
    }

    sent += res;
    buf += res;
  } 
}

}  // namespace ignite
