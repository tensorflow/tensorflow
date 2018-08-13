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

#include <map>
#include <iostream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ignite {

PlainClient::PlainClient(std::string host, int port) :
  host(host),
  port(port),
  sock(-1) {}

tensorflow::Status PlainClient::Connect() {
  return tensorflow::Status::OK();
}

tensorflow::Status PlainClient::Disconnect() {
  return tensorflow::Status::OK();
}

bool PlainClient::IsConnected() {
  return sock != -1;
}

char PlainClient::ReadByte() {
  return 0;
}

short PlainClient::ReadShort() {
  return 0;
}

int PlainClient::ReadInt() {
  return 0;
}

long PlainClient::ReadLong() {
  return 0;
}

void PlainClient::ReadData(char *buf, int length) {
  // Nothing.
}

void PlainClient::WriteByte(char data) { 
  // Nothing. 
}

void PlainClient::WriteShort(short data) { 
  // Nothing. 
}

void PlainClient::WriteInt(int data) { 
  // Nothing. 
}

void PlainClient::WriteLong(long data) { 
  // Nothing. 
}

void PlainClient::WriteData(char *buf, int length) {
  // Nothing.
}

}  // namespace ignite
