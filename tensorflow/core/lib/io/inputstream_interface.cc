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

#include "tensorflow/core/lib/io/inputstream_interface.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace io {

// To limit memory usage, the default implementation of SkipNBytes() only reads
// 8MB at a time.
static constexpr int64_t kMaxSkipSize = 8 * 1024 * 1024;

Status InputStreamInterface::SkipNBytes(int64_t bytes_to_skip) {
  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can't skip a negative number of bytes");
  }
  tstring unused;
  // Read kDefaultSkipSize at a time till bytes_to_skip.
  while (bytes_to_skip > 0) {
    int64_t bytes_to_read = std::min<int64_t>(kMaxSkipSize, bytes_to_skip);
    TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &unused));
    bytes_to_skip -= bytes_to_read;
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
