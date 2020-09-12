/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_IO_LEVELDB_H_
#define TENSORFLOW_LIB_IO_LEVELDB_H_

#include <string>

#include "tensorflow/core/platform/env.h"

#include "leveldb/env.h"

namespace tensorflow {
namespace io {
class LevelDBRandomAccessFile : public leveldb::RandomAccessFile {
 public:
  LevelDBRandomAccessFile(tensorflow::RandomAccessFile* f) : file_(f) {};

  virtual ~LevelDBRandomAccessFile() {};

  virtual leveldb::Status Read(uint64_t offset, size_t n,
                               leveldb::Slice* result,
                               char* scratch) const override {
    StringPiece r;
    Status status = file_->Read(offset, n, &r, scratch);
    *result = leveldb::Slice(r.data(), r.size());
    if (status.ok()) {
      return leveldb::Status::OK();
    }
    return leveldb::Status::IOError(status.ToString());
  }
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
};

class LevelDBWritableFile : public leveldb::WritableFile {
 public:
  LevelDBWritableFile(tensorflow::WritableFile* f) : file_(f) {}
  virtual ~LevelDBWritableFile() {}

  virtual leveldb::Status Append(const leveldb::Slice& data) override {
    Status status = file_->Append(StringPiece(data.data(), data.size()));
    if (!status.ok()) {
      return leveldb::Status::IOError(status.ToString());
    }
    return leveldb::Status::OK();
  }
  virtual leveldb::Status Close() override {
    Status status = file_->Close();
    if (!status.ok()) {
      return leveldb::Status::IOError(status.ToString());
    }
    return leveldb::Status::OK();
  }
  virtual leveldb::Status Flush() override {
    Status status = file_->Flush();
    if (!status.ok()) {
      return leveldb::Status::IOError(status.ToString());
    }
    return leveldb::Status::OK();
  }
  virtual leveldb::Status Sync() override {
    Status status = file_->Flush();
    if (!status.ok()) {
      return leveldb::Status::IOError(status.ToString());
    }
    return leveldb::Status::OK();
  }

 private:
  std::unique_ptr<tensorflow::WritableFile> file_;
};

}  // namespace io
}  // namespace tensorflow

#define LEVELDB_STATUS_TO_STATUS(s) (s.ok() ? Status::OK() : errors::Internal(s.ToString()))

#endif  // TENSORFLOW_LIB_IO_LEVELDB_H_
