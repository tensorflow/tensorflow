/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/tensor_coding.h"

#include <vector>

#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
#include "strings/cord_varint.h"
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

namespace tensorflow {
namespace port {

void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* out) {
  out->assign(src.data(), src.size());
}

void EncodeStringList(const tstring* strings, int64 n, string* out) {
  out->clear();
  for (int i = 0; i < n; ++i) {
    core::PutVarint32(out, strings[i].size());
  }
  for (int i = 0; i < n; ++i) {
    out->append(strings[i]);
  }
}

bool DecodeStringList(const string& src, tstring* strings, int64 n) {
  std::vector<uint32> sizes(n);
  StringPiece reader(src);
  int64 tot = 0;
  for (auto& v : sizes) {
    if (!core::GetVarint32(&reader, &v)) return false;
    tot += v;
  }
  if (tot != static_cast<int64>(reader.size())) {
    return false;
  }

  tstring* data = strings;
  for (int64 i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.size()) {
      return false;
    }
    data->assign(reader.data(), size);
    reader.remove_prefix(size);
  }

  return true;
}

void CopyFromArray(string* s, const char* base, size_t bytes) {
  s->assign(base, bytes);
}

class StringListEncoderImpl : public StringListEncoder {
 public:
  explicit StringListEncoderImpl(string* out) : out_(out) {}
  ~StringListEncoderImpl() override = default;

  void Append(const protobuf::MessageLite& m) override {
    core::PutVarint32(out_, m.ByteSizeLong());
    tensorflow::string serialized_message;
    m.AppendToString(&serialized_message);
    strings::StrAppend(&rest_, serialized_message);
  }

  void Append(const string& s) override {
    core::PutVarint32(out_, s.length());
    strings::StrAppend(&rest_, s);
  }

  void Finalize() override { strings::StrAppend(out_, rest_); }

 private:
  string* out_;
  string rest_;
};

class StringListDecoderImpl : public StringListDecoder {
 public:
  explicit StringListDecoderImpl(const string& in) : reader_(in) {}
  ~StringListDecoderImpl() override = default;

  bool ReadSizes(std::vector<uint32>* sizes) override {
    int64 total = 0;
    for (auto& size : *sizes) {
      if (!core::GetVarint32(&reader_, &size)) return false;
      total += size;
    }
    if (total != static_cast<int64>(reader_.size())) {
      return false;
    }
    return true;
  }

  const char* Data(uint32 size) override {
    const char* data = reader_.data();
    reader_.remove_prefix(size);
    return data;
  }

 private:
  StringPiece reader_;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out) {
  return std::unique_ptr<StringListEncoder>(new StringListEncoderImpl(out));
}

std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in) {
  return std::unique_ptr<StringListDecoder>(new StringListDecoderImpl(in));
}

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
void AssignRefCounted(StringPiece src, core::RefCounted* obj, absl::Cord* out) {
  obj->Ref();
  out->Clear();
  // Defines a lambda to unref "obj" when Cord deletes this piece of
  // memory. +[] converts the lambda to a C style function pointer.
  auto cleanup = +[](absl::string_view donotcare, void* obj) {
    reinterpret_cast<core::RefCounted*>(obj)->Unref();
  };
  out->AppendExternalMemory(absl::string_view(src.data(), src.size()), obj,
                            cleanup);
}

void EncodeStringList(const tstring* strings, int64 n, absl::Cord* out) {
  out->Clear();
  for (int i = 0; i < n; ++i) {
    ::strings::CordAppendVarint(strings[i].size(), out);
  }
  for (int i = 0; i < n; ++i) {
    out->Append(strings[i]);
  }
}

bool DecodeStringList(const absl::Cord& src, string* strings, int64 n) {
  std::vector<uint32> sizes(n);
  CordReader reader(src);
  int64 tot = 0;
  for (auto& v : sizes) {
    if (!::strings::CordReaderReadVarint(&reader, &v)) return false;
    tot += v;
  }
  if (tot != reader.Available()) {
    return false;
  }
  string* data = strings;
  for (int i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.Available()) {
      return false;
    }
    gtl::STLStringResizeUninitialized(data, size);
    reader.ReadN(size, gtl::string_as_array(data));
  }
  return true;
}

bool DecodeStringList(const absl::Cord& src, tstring* strings, int64 n) {
  std::vector<uint32> sizes(n);
  CordReader reader(src);
  int64 tot = 0;
  for (auto& v : sizes) {
    if (!::strings::CordReaderReadVarint(&reader, &v)) return false;
    tot += v;
  }
  if (tot != reader.Available()) {
    return false;
  }
  tstring* data = strings;
  for (int i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.Available()) {
      return false;
    }
    data->resize_uninitialized(size);
    reader.ReadN(size, data->data());
  }
  return true;
}

void CopyFromArray(absl::Cord* c, const char* base, size_t bytes) {
  c->CopyFrom(base, bytes);
}

class CordStringListEncoderImpl : public StringListEncoder {
 public:
  explicit CordStringListEncoderImpl(absl::Cord* out) : out_(out) {}
  ~CordStringListEncoderImpl() override = default;

  void Append(const protobuf::MessageLite& m) override {
    ::strings::CordAppendVarint(m.ByteSizeLong(), out_);
    m.AppendToString(&rest_);
  }

  void Append(const string& s) override {
    ::strings::CordAppendVarint(s.length(), out_);
    rest_.append(s.data(), s.size());
  }

  void Finalize() override { out_->Append(rest_); }

 private:
  absl::Cord* out_;
  string rest_;
};

class CordStringListDecoderImpl : public StringListDecoder {
 public:
  explicit CordStringListDecoderImpl(const absl::Cord& in) : reader_(in) {}
  ~CordStringListDecoderImpl() override = default;

  bool ReadSizes(std::vector<uint32>* sizes) override {
    int64 total = 0;
    for (auto& size : *sizes) {
      if (!::strings::CordReaderReadVarint(&reader_, &size)) return false;
      total += size;
    }
    if (total != static_cast<int64>(reader_.Available())) {
      return false;
    }
    return true;
  }

  const char* Data(uint32 size) override {
    tmp_.resize(size);
    reader_.ReadN(size, tmp_.data());
    return tmp_.data();
  }

 private:
  CordReader reader_;
  std::vector<char> tmp_;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(absl::Cord* out) {
  return std::unique_ptr<StringListEncoder>(new CordStringListEncoderImpl(out));
}

std::unique_ptr<StringListDecoder> NewStringListDecoder(const absl::Cord& in) {
  return std::unique_ptr<StringListDecoder>(new CordStringListDecoderImpl(in));
}

#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

}  // namespace port
}  // namespace tensorflow
