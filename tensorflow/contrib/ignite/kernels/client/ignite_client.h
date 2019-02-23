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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_CLIENT_IGNITE_CLIENT_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_CLIENT_IGNITE_CLIENT_H_

#include "tensorflow/contrib/ignite/kernels/client/ignite_byte_swapper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class Client {
 public:
  Client(bool big_endian) : byte_swapper_(ByteSwapper(big_endian)) {}
  virtual Status Connect() = 0;
  virtual Status Disconnect() = 0;
  virtual bool IsConnected() = 0;
  virtual int GetSocketDescriptor() = 0;
  virtual Status ReadData(uint8_t *buf, const int32_t length) = 0;
  virtual Status WriteData(const uint8_t *buf, const int32_t length) = 0;

  Status ReadByte(uint8_t *data) { return ReadData(data, 1); }

  Status ReadShort(int16_t *data) {
    TF_RETURN_IF_ERROR(ReadData((uint8_t *)data, 2));
    byte_swapper_.SwapIfRequiredInt16(data);

    return Status::OK();
  }

  Status ReadInt(int32_t *data) {
    TF_RETURN_IF_ERROR(ReadData((uint8_t *)data, 4));
    byte_swapper_.SwapIfRequiredInt32(data);

    return Status::OK();
  }

  Status ReadLong(int64_t *data) {
    TF_RETURN_IF_ERROR(ReadData((uint8_t *)data, 8));
    byte_swapper_.SwapIfRequiredInt64(data);

    return Status::OK();
  }

  Status WriteByte(const uint8_t data) { return WriteData(&data, 1); }

  Status WriteShort(const int16_t data) {
    int16_t tmp = data;
    byte_swapper_.SwapIfRequiredInt16(&tmp);
    return WriteData((uint8_t *)&tmp, 2);
  }

  Status WriteInt(const int32_t data) {
    int32_t tmp = data;
    byte_swapper_.SwapIfRequiredInt32(&tmp);
    return WriteData((uint8_t *)&tmp, 4);
  }

  Status WriteLong(const int64_t data) {
    int64_t tmp = data;
    byte_swapper_.SwapIfRequiredInt64(&tmp);
    return WriteData((uint8_t *)&tmp, 8);
  }

 private:
  const ByteSwapper byte_swapper_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_CLIENT_IGNITE_CLIENT_H_
