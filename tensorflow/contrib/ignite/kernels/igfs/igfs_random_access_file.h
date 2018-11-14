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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_RANDOM_ACCESS_FILE_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_RANDOM_ACCESS_FILE_H_

#include "tensorflow/contrib/ignite/kernels/igfs/igfs_client.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

class IGFSRandomAccessFile : public RandomAccessFile {
 public:
  IGFSRandomAccessFile(const string &file_name, int64_t resource_id,
                       std::unique_ptr<IGFSClient> &&client);
  ~IGFSRandomAccessFile() override;
  Status Read(uint64 offset, size_t n, StringPiece *result,
              char *scratch) const override;

 private:
  const string file_name_;
  const int64_t resource_id_;
  std::unique_ptr<IGFSClient> client_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_RANDOM_ACCESS_FILE_H_
