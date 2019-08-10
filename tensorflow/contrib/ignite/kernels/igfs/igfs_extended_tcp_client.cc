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

#include "tensorflow/contrib/ignite/kernels/igfs/igfs_extended_tcp_client.h"

namespace tensorflow {

ExtendedTCPClient::ExtendedTCPClient(const string &host, int port,
                                     bool big_endian)
    : PlainClient(host, port, big_endian), pos_(0) {}

Status ExtendedTCPClient::ReadData(uint8_t *buf, const int32_t length) {
  TF_RETURN_IF_ERROR(PlainClient::ReadData(buf, length));
  pos_ += length;

  return Status::OK();
}

Status ExtendedTCPClient::WriteData(const uint8_t *buf, const int32_t length) {
  TF_RETURN_IF_ERROR(PlainClient::WriteData(buf, length));
  pos_ += length;

  return Status::OK();
}

Status ExtendedTCPClient::Ignore(int n) {
  uint8_t buf[n];
  return ReadData(buf, n);
}

Status ExtendedTCPClient::SkipToPos(int target_pos) {
  return Ignore(std::max(0, target_pos - pos_));
}

Status ExtendedTCPClient::ReadBool(bool *res) {
  uint8_t buf = 0;
  TF_RETURN_IF_ERROR(ReadData(&buf, 1));
  *res = buf != 0;

  return Status::OK();
}

Status ExtendedTCPClient::ReadNullableString(string *res) {
  bool is_empty = false;
  TF_RETURN_IF_ERROR(ReadBool(&is_empty));

  if (!is_empty) {
    TF_RETURN_IF_ERROR(ReadString(res));
  }

  return Status::OK();
}

Status ExtendedTCPClient::ReadString(string *res) {
  int16_t length;
  TF_RETURN_IF_ERROR(ReadShort(&length));

  uint8_t *buf = new uint8_t[length];
  Status status = ReadData(buf, length);

  if (status.ok()) res->assign(reinterpret_cast<char *>(buf), length);

  delete[] buf;
  return status;
}

Status ExtendedTCPClient::ReadStringMap(std::map<string, string> *res) {
  int size;
  TF_RETURN_IF_ERROR(ReadInt(&size));

  for (int i = 0; i < size; i++) {
    string key;
    string val;
    TF_RETURN_IF_ERROR(ReadString(&key));
    TF_RETURN_IF_ERROR(ReadString(&val));

    res->insert(std::pair<string, string>(std::move(key), std::move(val)));
  }

  return Status::OK();
}

Status ExtendedTCPClient::WriteSize(std::map<string, string>::size_type s) {
  return WriteInt(s);
}

Status ExtendedTCPClient::FillWithZerosUntil(int n) {
  int to_skip = std::max(0, n - pos_);

  for (int i = 0; i < to_skip; i++) {
    TF_RETURN_IF_ERROR(WriteByte(0));
  }

  return Status::OK();
}

Status ExtendedTCPClient::WriteBool(bool val) {
  return WriteByte((char)(val ? 1 : 0));
}

Status ExtendedTCPClient::WriteString(string str) {
  if (!str.empty()) {
    TF_RETURN_IF_ERROR(WriteBool(false));
    size_t l = str.length();
    if (l > std::numeric_limits<int16_t>::max())
      return errors::InvalidArgument("String is too long");

    TF_RETURN_IF_ERROR(WriteShort(l));
    TF_RETURN_IF_ERROR(WriteData(reinterpret_cast<const uint8_t *>(str.c_str()),
                                 str.length()));
  } else {
    TF_RETURN_IF_ERROR(WriteBool(true));
  }

  return Status::OK();
}

Status ExtendedTCPClient::WriteStringMap(std::map<string, string> map) {
  std::map<string, string>::size_type size = map.size();
  TF_RETURN_IF_ERROR(WriteSize(size));

  for (auto &x : map) {
    TF_RETURN_IF_ERROR(WriteString(x.first));
    TF_RETURN_IF_ERROR(WriteString(x.second));
  }

  return Status::OK();
}

void ExtendedTCPClient::reset() { pos_ = 0; }

}  // namespace tensorflow
