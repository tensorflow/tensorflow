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

#include "igfs_writable_file.h"
#include "igfs_messages.h"

namespace tensorflow {

IGFSWritableFile::IGFSWritableFile(const std::string &file_name,
                                   int64_t resource_id,
                                   std::shared_ptr<IGFSClient> client)
    : file_name_(file_name), resource_id_(resource_id), client_(client) {}

IGFSWritableFile::~IGFSWritableFile() {
  if (resource_id_ >= 0) {
    CtrlResponse<CloseResponse> close_response = {false};

    Status status = client_->Close(&close_response, resource_id_);
    if (!status.ok()) LOG(ERROR) << status.ToString();
  }
}

Status IGFSWritableFile::Append(const StringPiece &data) {
  return client_->WriteBlock(resource_id_, (uint8_t *)data.data(), data.size());
}

Status IGFSWritableFile::Close() {
  int64_t resource_to_be_closed = resource_id_;
  resource_id_ = -1;

  CtrlResponse<CloseResponse> close_response = {false};
  return client_->Close(&close_response, resource_to_be_closed);
}

Status IGFSWritableFile::Flush() { return Status::OK(); }

Status IGFSWritableFile::Sync() { return Status::OK(); }

}  // namespace tensorflow