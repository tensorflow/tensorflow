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

#include <string>
// #include <map>
#include <netinet/in.h>

namespace ignite {

class Client {
 public:
  Client(std::string host, int port);
  
  void Connect();
  void Disconnect();

  char ReadByte();
  short ReadShort();
  int ReadInt();
  long ReadLong();
  void ReadData(char* buf, int length);

  void WriteByte(char data);
  void WriteShort(short data);
  void WriteInt(int data);
  void WriteLong(long data);
  void WriteData(char* buf, int length);
 private:
  std::string cache_name;
  std::string host;
  int port;
  int sock;
  struct sockaddr_in server;
};

} // namespace ignite

