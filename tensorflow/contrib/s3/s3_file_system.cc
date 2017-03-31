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

#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class S3RandomAccessFile : public RandomAccessFile {
  // The filecontents is all A's
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    for (int i = 0; i < n; ++i) {
      scratch[i] = 'A';
    }
    *result = StringPiece(scratch, n);
    return Status::OK();
  }
};

class S3FileSystem : public NullFileSystem {
 public:
  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    result->reset(new S3RandomAccessFile);
    return Status::OK();
  }
  // Always return size of 10
  Status GetFileSize(const string& fname, uint64* file_size) override {
    *file_size = 10;
    return Status::OK();
  }
};

REGISTER_FILE_SYSTEM("s3", S3FileSystem);

}  // namespace tensorflow
