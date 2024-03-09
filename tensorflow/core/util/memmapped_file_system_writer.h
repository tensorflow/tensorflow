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
#ifndef TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_WRITER_H_
#define TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_WRITER_H_

#include <memory>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"
#include "tensorflow/core/util/memmapped_file_system.pb.h"

namespace tensorflow {

// A class for saving into the memmapped format that can be read by
// MemmappedFileSystem.
class MemmappedFileSystemWriter {
 public:
  MemmappedFileSystemWriter() = default;
  ~MemmappedFileSystemWriter() = default;
  Status InitializeToFile(Env* env, const string& filename);
  Status SaveTensor(const Tensor& tensor, const string& element_name);
  Status SaveProtobuf(const protobuf::MessageLite& message,
                      const string& element_name);
  // Writes out the directory of regions and closes the output file.
  Status FlushAndClose();

 private:
  Status AdjustAlignment(uint64 alignment);
  void AddToDirectoryElement(const string& element_name, uint64 length);
  MemmappedFileSystemDirectory directory_;
  // The current offset in the file, to support alignment.
  uint64 output_file_offset_ = 0;
  std::unique_ptr<WritableFile> output_file_;
  MemmappedFileSystemWriter(const MemmappedFileSystemWriter&) = delete;
  void operator=(const MemmappedFileSystemWriter&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_WRITER_H_
