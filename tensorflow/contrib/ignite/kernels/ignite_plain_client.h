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

#include <string>

namespace ignite {

class PlainClient : public Client {
 public:
  PlainClient(std::string host, int port);
  ~PlainClient();

  virtual tensorflow::Status Connect();
  virtual tensorflow::Status Disconnect();
  virtual bool IsConnected();
  virtual int GetSocketDescriptor();
  virtual tensorflow::Status ReadData(uint8_t* buf, int32_t length);
  virtual tensorflow::Status WriteData(uint8_t* buf, int32_t length);

 private:
  std::string host;
  int port;
  int sock;
};

}  // namespace ignite
