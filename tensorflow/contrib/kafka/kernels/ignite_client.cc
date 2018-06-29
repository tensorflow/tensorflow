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

// #include "ignite_binary_object_parser.h"

namespace ignite {

// std::map<int, BinaryType*>* cache = new std::map<int, BinaryType*>();

Client::Client(std::string host, int port) {
  this->host = host;
  this->port = port;
  this->sock = -1;
}

void Client::Connect() {
  // create socket if it is not already created
  if (sock == -1) {
    // Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
      perror("Could not create socket");
    }
  } else {
    /* OK , nothing */
  }

  // setup address structure
  if (inet_addr(host.c_str()) == -1) {
    struct hostent *he;
    struct in_addr **addr_list;

    // resolve the hostname, its not an ip address
    if ((he = gethostbyname(host.c_str())) == NULL) {
      perror("Failed to resolve hostname");
      return;
    }

    // Cast the h_addr_list to in_addr , since h_addr_list also has the ip
    // address in long format only
    addr_list = (struct in_addr **)he->h_addr_list;

    for (int i = 0; addr_list[i] != NULL; i++) {
      // strcpy(ip , inet_ntoa(*addr_list[i]) );
      server.sin_addr = *addr_list[i];

      break;
    }
  } else {
    server.sin_addr.s_addr = inet_addr(host.c_str());
  }

  server.sin_family = AF_INET;
  server.sin_port = htons(port);

  // Connect to remote server
  if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
    perror("connect failed. Error");
  }
}

void Client::Disconnect() {}

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
