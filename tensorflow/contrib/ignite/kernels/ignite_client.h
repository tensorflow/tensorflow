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

#include "tensorflow/core/lib/core/status.h"

namespace ignite {

class Client {
 public:
  virtual tensorflow::Status Connect() = 0;
  virtual tensorflow::Status Disconnect() = 0;
  virtual bool IsConnected() = 0;

  virtual char ReadByte() = 0;
  virtual short ReadShort() = 0;
  virtual int ReadInt() = 0;
  virtual long ReadLong() = 0;
  virtual void ReadData(char* buf, int length) = 0;

  virtual void WriteByte(char data) = 0;
  virtual void WriteShort(short data) = 0;
  virtual void WriteInt(int data) = 0;
  virtual void WriteLong(long data) = 0;
  virtual void WriteData(char* buf, int length) = 0;
};

}  // namespace ignite
