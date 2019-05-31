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

#include "tensorflow/contrib/ignite/kernels/igfs/igfs_random_access_file.h"
#include "tensorflow/contrib/ignite/kernels/igfs/igfs_messages.h"

namespace tensorflow {

IGFSRandomAccessFile::IGFSRandomAccessFile(const string &file_name,
                                           int64_t resource_id,
                                           std::unique_ptr<IGFSClient> &&client)
    : file_name_(file_name),
      resource_id_(resource_id),
      client_(std::move(client)) {}

IGFSRandomAccessFile::~IGFSRandomAccessFile() {
  CtrlResponse<CloseResponse> close_response = {false};
  Status status = client_->Close(&close_response, resource_id_);

  if (!status.ok()) LOG(ERROR) << status.ToString();
}

Status IGFSRandomAccessFile::Read(uint64 offset, size_t n, StringPiece *result,
                                  char *scratch) const {
  ReadBlockCtrlResponse response = ReadBlockCtrlResponse((uint8_t *)scratch);
  TF_RETURN_IF_ERROR(client_->ReadBlock(&response, resource_id_, offset, n));

  std::streamsize sz = response.res.GetSuccessfullyRead();
  if (sz == 0) return errors::OutOfRange("End of file");

  *result = StringPiece(scratch, sz);

  return Status::OK();
}

}  // namespace tensorflow
