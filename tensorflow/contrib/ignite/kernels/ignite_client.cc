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

#include "ignite_client.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <map>

#include <iostream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ignite {

Client::Client(std::string host, int port) :
  host(host),
  port(port),
  sock(-1) {}

tensorflow::Status Client::Connect() {
  if (sock == -1) {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
      return tensorflow::errors::Internal("Failed not create socket");
  }

  if (inet_addr(host.c_str()) == -1) {
    struct hostent *he;
    struct in_addr **addr_list;

    if ((he = gethostbyname(host.c_str())) == NULL)
      return tensorflow::errors::Internal("Failed to resolve hostname \"", host, "\"");

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
    return tensorflow::errors::Internal("Failed to connect to \"", host, ":", port, "\"");

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" established";

  return tensorflow::Status::OK();
}

tensorflow::Status Client::Disconnect() {
  int close_res = close(sock);
  sock = -1;

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" is closed";
 
  return close_res == 0 ? tensorflow::Status::OK() : tensorflow::errors::Internal("Failed to correctly close connection");
}

bool Client::IsConnected() {
  return sock != -1;
}

char Client::ReadByte() {
  char res;
  recv(sock, &res, 1, 0);
  return res;
}

short Client::ReadShort() {
  short res;
  recv(sock, &res, 2, 0);
  return res;
}

int Client::ReadInt() {
  int res;
  recv(sock, &res, 4, 0);
  return res;
}

long Client::ReadLong() {
  long res;
  recv(sock, &res, 8, 0);
  return res;
}

void Client::ReadData(char *buf, int length) {
  int recieved = 0;
  while (recieved < length) {
    int res = recv(sock, buf, length - recieved, 0);
    recieved += res;
    buf += res;
  }
}

void Client::WriteByte(char data) { send(sock, &data, 1, 0); }

void Client::WriteShort(short data) { send(sock, &data, 2, 0); }

void Client::WriteInt(int data) { send(sock, &data, 4, 0); }

void Client::WriteLong(long data) { send(sock, &data, 8, 0); }

void Client::WriteData(char *buf, int length) { send(sock, buf, length, 0); }

}  // namespace ignite
