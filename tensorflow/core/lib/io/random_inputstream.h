/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_IO_RANDOM_INPUTSTREAM_H_
#define TENSORFLOW_LIB_IO_RANDOM_INPUTSTREAM_H_

#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace io {

// Wraps a RandomAccessFile in an InputStreamInterface. A given instance of
// RandomAccessInputStream is NOT safe for concurrent use by multiple threads.
class RandomAccessInputStream : public InputStreamInterface {
 public:
  // Does not take ownership of 'file'. 'file' must outlive *this.
  explicit RandomAccessInputStream(RandomAccessFile* file);

  Status ReadNBytes(int64 bytes_to_read, string* result) override;

  int64 Tell() const override;

 private:
  RandomAccessFile* file_;  // Not owned.
  int64 pos_ = 0;           // Tracks where we are in the file.
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_RANDOM_INPUTSTREAM_H_
