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

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <map>

#include <iostream>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace ignite {

PlainClient::PlainClient(std::string host, int port)
    : host(host), port(port), sock(-1) {}

PlainClient::~PlainClient() {
  if (IsConnected()) {
    tensorflow::Status status = Disconnect();
    if (!status.ok()) LOG(WARNING) << status.ToString();
  }
}

tensorflow::Status PlainClient::Connect() {
  if (sock == -1) {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
      return tensorflow::errors::Internal("Failed not create socket");
  }

  struct sockaddr_in server;

  if (inet_addr(host.c_str()) == -1) {
    struct hostent *he;
    struct in_addr **addr_list;

    if ((he = gethostbyname(host.c_str())) == NULL)
      return tensorflow::errors::Internal("Failed to resolve hostname \"", host,
                                          "\"");

    addr_list = (struct in_addr **)he->h_addr_list;
    for (int i = 0; addr_list[i] != NULL; i++) {
      server.sin_addr = *addr_list[i];
      break;
    }
  } else {
    server.sin_addr.s_addr = inet_addr(host.c_str());
  }

  server.sin_family = AF_INET;
  server.sin_port = htons(port);

  if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0)
    return tensorflow::errors::Internal("Failed to connect to \"", host, ":",
                                        port, "\"");

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" established";

  return tensorflow::Status::OK();
}

tensorflow::Status PlainClient::Disconnect() {
  int close_res = close(sock);
  sock = -1;

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" is closed";

  return close_res == 0 ? tensorflow::Status::OK()
                        : tensorflow::errors::Internal(
                              "Failed to correctly close connection");
}

bool PlainClient::IsConnected() { return sock != -1; }

int PlainClient::GetSocketDescriptor() { return sock; }

char PlainClient::ReadByte() {
  char res;
  ReadData((char *)&res, 1);

  return res;
}

short PlainClient::ReadShort() {
  short res;
  ReadData((char *)&res, 2);

  return res;
}

int PlainClient::ReadInt() {
  int res;
  ReadData((char *)&res, 4);

  return res;
}

long PlainClient::ReadLong() {
  long res;
  ReadData((char *)&res, 8);

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

void PlainClient::WriteByte(char data) { WriteData((char *)&data, 1); }

void PlainClient::WriteShort(short data) { WriteData((char *)&data, 2); }

void PlainClient::WriteInt(int data) { WriteData((char *)&data, 4); }

void PlainClient::WriteLong(long data) { WriteData((char *)&data, 8); }

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
