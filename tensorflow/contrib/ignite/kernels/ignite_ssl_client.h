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

#include <openssl/ssl.h>

#ifndef IGNITE_CLIENT_H
#define IGNITE_CLIENT_H
#include "ignite_client.h"
#endif

#include <string>

namespace ignite {

class SslClient: public Client {
 public:
  SslClient(std::string host, int port, std::string certfile, std::string keyfile, std::string cert_password);
  ~SslClient();

  virtual tensorflow::Status Connect();
  virtual tensorflow::Status Disconnect();
  virtual bool IsConnected();

  virtual char ReadByte();
  virtual short ReadShort();
  virtual int ReadInt();
  virtual long ReadLong();
  virtual void ReadData(char* buf, int length);

  virtual void WriteByte(char data);
  virtual void WriteShort(short data);
  virtual void WriteInt(int data);
  virtual void WriteLong(long data);
  virtual void WriteData(char* buf, int length);

 private:
  std::string host;
  int port;
  std::string certfile;
  std::string keyfile;
  std::string cert_password;
  int sock;
  SSL_CTX *ctx;
  SSL *ssl;
  tensorflow::Status InitSslContext();
};

}  // namespace ignite
