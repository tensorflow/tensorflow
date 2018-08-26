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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGNITE_CLIENT_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGNITE_CLIENT_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class Client {
 public:
  virtual Status Connect() = 0;
  virtual Status Disconnect() = 0;
  virtual bool IsConnected() = 0;
  virtual int GetSocketDescriptor() = 0;
  virtual Status ReadData(uint8_t* buf, int32_t length) = 0;
  virtual Status WriteData(uint8_t* buf, int32_t length) = 0;

  inline Status ReadByte(uint8_t* data) { return ReadData(data, 1); }

  inline Status ReadShort(int16_t* data) { return ReadData((uint8_t*)data, 2); }

  inline Status ReadInt(int32_t* data) { return ReadData((uint8_t*)data, 4); }

  inline Status ReadLong(int64_t* data) { return ReadData((uint8_t*)data, 8); }

  inline Status WriteByte(uint8_t data) { return WriteData(&data, 1); }

  inline Status WriteShort(int16_t data) {
    return WriteData((uint8_t*)&data, 2);
  }

  inline Status WriteInt(int32_t data) { return WriteData((uint8_t*)&data, 4); }

  inline Status WriteLong(int64_t data) {
    return WriteData((uint8_t*)&data, 8);
  }
};

}  // namespace tensorflow

#endif
