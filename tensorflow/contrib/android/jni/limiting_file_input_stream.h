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
#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_LIMITING_FILE_INPUT_STREAM_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_LIMITING_FILE_INPUT_STREAM_H_

#include <unistd.h>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace tensorflow {
namespace android {

// Input stream that reads a limited amount of data from an input file
// descriptor.
class LimitingFileInputStream
    : public ::google::protobuf::io::CopyingInputStream {
 public:
  // Construct a stream to read from file <fd>, returning on the first <limit>
  // bytes. If <fd> has fewer than <limit> bytes, then limit has no effect.
  LimitingFileInputStream(int fd, int limit) : fd_(fd), bytes_left_(limit) {}
  ~LimitingFileInputStream() {}

  int Read(void* buffer, int size) {
    int result;
    do {
      result = read(fd_, buffer, std::min(bytes_left_, size));
    } while (result < 0 && errno == EINTR);

    if (result < 0) {
      errno_ = errno;
    } else {
      bytes_left_ -= result;
    }
    return result;
  }

  int Skip(int count) {
    if (lseek(fd_, count, SEEK_CUR) == (off_t)-1) {
      return -1;
    }
    // Seek succeeded.
    bytes_left_ -= count;
    return count;
  }

 private:
  const int fd_;
  int bytes_left_;
  int errno_ = 0;
};

}  // namespace android
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_LIMITING_FILE_INPUT_STREAM_H_
