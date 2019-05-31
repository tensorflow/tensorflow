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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_EXTENDED_TCP_CLIENT_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_EXTENDED_TCP_CLIENT_H_

#include "tensorflow/contrib/ignite/kernels/client/ignite_plain_client.h"

namespace tensorflow {

class ExtendedTCPClient : public PlainClient {
 public:
  ExtendedTCPClient(const string &host, int port, bool big_endian);
  Status ReadData(uint8_t *buf, const int32_t length) override;
  Status WriteData(const uint8_t *buf, const int32_t length) override;
  Status Ignore(int n);
  Status SkipToPos(int target_pos);
  Status ReadBool(bool *res);
  Status ReadNullableString(string *res);
  Status ReadString(string *res);
  Status ReadStringMap(std::map<string, string> *res);
  Status WriteSize(std::map<string, string>::size_type s);
  Status FillWithZerosUntil(int n);
  Status WriteBool(bool val);
  Status WriteString(string str);
  Status WriteStringMap(std::map<string, string> map);
  void reset();

 private:
  int pos_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_EXTENDED_TCP_CLIENT_H_
