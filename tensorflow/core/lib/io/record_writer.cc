/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/io/record_writer.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

RecordWriter::RecordWriter(WritableFile* dest) : dest_(dest) {}

RecordWriter::~RecordWriter() {}

static uint32 MaskedCrc(const char* data, size_t n) {
  return crc32c::Mask(crc32c::Value(data, n));
}

Status RecordWriter::WriteRecord(StringPiece data) {
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[sizeof(uint64) + sizeof(uint32)];
  core::EncodeFixed64(header + 0, data.size());
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
  Status s = dest_->Append(StringPiece(header, sizeof(header)));
  if (!s.ok()) {
    return s;
  }
  s = dest_->Append(data);
  if (!s.ok()) {
    return s;
  }
  char footer[sizeof(uint32)];
  core::EncodeFixed32(footer, MaskedCrc(data.data(), data.size()));
  return dest_->Append(StringPiece(footer, sizeof(footer)));
}

}  // namespace io
}  // namespace tensorflow
