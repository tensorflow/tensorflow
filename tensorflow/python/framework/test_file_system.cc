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
#include "tensorflow/core/platform/null_file_system.h"

namespace tensorflow {

class TestRandomAccessFile : public RandomAccessFile {
  // The file contents is 10 bytes of all A's
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    for (int i = 0; i < n; ++i) {
      if (offset + i >= 10) {
        n = i;
        s = errors::OutOfRange("EOF");
        break;
      }
      scratch[i] = 'A';
    }
    *result = StringPiece(scratch, n);
    return s;
  }
};

class TestFileSystem : public NullFileSystem {
 public:
  Status NewRandomAccessFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
    result->reset(new TestRandomAccessFile);
    return absl::OkStatus();
  }
  // Always return size of 10
  Status GetFileSize(const string& fname, TransactionToken* token,
                     uint64* file_size) override {
    *file_size = 10;
    return absl::OkStatus();
  }
};

REGISTER_FILE_SYSTEM("test", TestFileSystem);

}  // namespace tensorflow
