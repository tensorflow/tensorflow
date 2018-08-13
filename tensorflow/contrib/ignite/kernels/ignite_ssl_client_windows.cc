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

#include "ignite_ssl_client.h"

#include <map>
#include <iostream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ignite {

SslClient::SslClient(std::string host, int port, std::string certfile, std::string keyfile, std::string cert_password) :
  host(host),
  port(port),
  certfile(certfile),
  keyfile(keyfile),
  cert_password(cert_password),
  sock(-1),
  ctx(NULL) {}

SslClient::~SslClient() {
  // Do nothing.
}

tensorflow::Status SslClient::InitSslContext() {
  return tensorflow::Status::OK();
}

tensorflow::Status SslClient::Connect() {
  return tensorflow::Status::OK();
}

tensorflow::Status SslClient::Disconnect() {
  return tensorflow::Status::OK();
}

bool SslClient::IsConnected() {
  return sock != -1;
}

char SslClient::ReadByte() {
  return 0;
}

short SslClient::ReadShort() {
  return 0;
}

int SslClient::ReadInt() {
  return 0;
}

long SslClient::ReadLong() {
  return 0;
}

void SslClient::ReadData(char *buf, int length) {
  // Do nothing.
}

void SslClient::WriteByte(char data) { 
  // Do nothing.
}

void SslClient::WriteShort(short data) { 
  // Do nothing.
}

void SslClient::WriteInt(int data) { 
  // Do nothing.
}

void SslClient::WriteLong(long data) {
  // Do nothing.
}

void SslClient::WriteData(char *buf, int length) { 
  // Do nothing.
}

}  // namespace ignite
