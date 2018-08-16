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

#ifndef IGNITE_CLIENT_H
#define IGNITE_CLIENT_H
#include "ignite_client.h"
#endif

#include <openssl/ssl.h>
#include <string>

namespace ignite {

class SslWrapper : public Client {
 public:
  SslWrapper(std::shared_ptr<Client> client, std::string certfile,
             std::string keyfile, std::string cert_password);
  ~SslWrapper();

  virtual tensorflow::Status Connect();
  virtual tensorflow::Status Disconnect();
  virtual bool IsConnected();
  virtual int GetSocketDescriptor();

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
  std::shared_ptr<Client> client;
  std::string certfile;
  std::string keyfile;
  std::string cert_password;
  SSL_CTX* ctx;
  SSL* ssl;
  tensorflow::Status InitSslContext();
};

}  // namespace ignite
