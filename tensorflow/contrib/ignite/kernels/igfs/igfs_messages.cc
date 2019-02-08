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

#include "tensorflow/contrib/ignite/kernels/igfs/igfs_messages.h"

namespace tensorflow {

Status IGFSPath::Read(ExtendedTCPClient *client) {
  return client->ReadNullableString(&path);
}

Status IGFSFile::Read(ExtendedTCPClient *client) {
  int32_t block_size;
  int64_t group_block_size;
  std::map<string, string> properties = {};
  int64_t access_time;

  bool has_path;
  TF_RETURN_IF_ERROR(client->ReadBool(&has_path));
  if (has_path) {
    IGFSPath path = {};
    TF_RETURN_IF_ERROR(path.Read(client));
  }

  TF_RETURN_IF_ERROR(client->ReadInt(&block_size));
  TF_RETURN_IF_ERROR(client->ReadLong(&group_block_size));
  TF_RETURN_IF_ERROR(client->ReadLong(&length));
  TF_RETURN_IF_ERROR(client->ReadStringMap(&properties));
  TF_RETURN_IF_ERROR(client->ReadLong(&access_time));
  TF_RETURN_IF_ERROR(client->ReadLong(&modification_time));
  TF_RETURN_IF_ERROR(client->ReadByte(&flags));

  return Status::OK();
}

Request::Request(int32_t command_id) : command_id_(command_id) {}

Status Request::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(client->WriteByte(0));
  TF_RETURN_IF_ERROR(client->FillWithZerosUntil(8));
  TF_RETURN_IF_ERROR(client->WriteInt(command_id_));
  TF_RETURN_IF_ERROR(client->FillWithZerosUntil(24));

  return Status::OK();
}

Status Response::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->Ignore(1));
  TF_RETURN_IF_ERROR(client->SkipToPos(8));
  TF_RETURN_IF_ERROR(client->ReadInt(&req_id));
  TF_RETURN_IF_ERROR(client->SkipToPos(24));
  TF_RETURN_IF_ERROR(client->ReadInt(&res_type));

  bool has_error;
  TF_RETURN_IF_ERROR(client->ReadBool(&has_error));

  if (has_error) {
    int32_t error_code;
    string error_msg;
    TF_RETURN_IF_ERROR(client->ReadString(&error_msg));
    TF_RETURN_IF_ERROR(client->ReadInt(&error_code));

    return errors::Unknown("Error [code=", error_code, ", message=\"",
                           error_msg, "\"]");
  }

  TF_RETURN_IF_ERROR(client->SkipToPos(header_size_ + 5));
  TF_RETURN_IF_ERROR(client->ReadInt(&length));
  TF_RETURN_IF_ERROR(client->SkipToPos(header_size_ + response_header_size_));

  return Status::OK();
}

PathCtrlRequest::PathCtrlRequest(int32_t command_id_, const string &user_name,
                                 const string &path,
                                 const string &destination_path, bool flag,
                                 bool collocate,
                                 const std::map<string, string> &properties)
    : Request(command_id_),
      user_name_(user_name),
      path_(path),
      destination_path_(destination_path),
      flag_(flag),
      collocate_(collocate),
      props_(properties) {}

Status PathCtrlRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(Request::Write(client));

  TF_RETURN_IF_ERROR(client->WriteString(user_name_));
  TF_RETURN_IF_ERROR(WritePath(client, path_));
  TF_RETURN_IF_ERROR(WritePath(client, destination_path_));
  TF_RETURN_IF_ERROR(client->WriteBool(flag_));
  TF_RETURN_IF_ERROR(client->WriteBool(collocate_));
  TF_RETURN_IF_ERROR(client->WriteStringMap(props_));

  return Status::OK();
}

Status PathCtrlRequest::WritePath(ExtendedTCPClient *client,
                                  const string &path) const {
  TF_RETURN_IF_ERROR(client->WriteBool(!path.empty()));
  if (!path.empty()) TF_RETURN_IF_ERROR(client->WriteString(path));

  return Status::OK();
}

Status StreamCtrlRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(client->WriteByte(0));
  TF_RETURN_IF_ERROR(client->FillWithZerosUntil(8));
  TF_RETURN_IF_ERROR(client->WriteInt(command_id_));
  TF_RETURN_IF_ERROR(client->WriteLong(stream_id_));
  TF_RETURN_IF_ERROR(client->WriteInt(length_));

  return Status::OK();
}

StreamCtrlRequest::StreamCtrlRequest(int32_t command_id_, int64_t stream_id,
                                     int32_t length)
    : Request(command_id_), stream_id_(stream_id), length_(length) {}

DeleteRequest::DeleteRequest(const string &user_name, const string &path,
                             bool flag)
    : PathCtrlRequest(DELETE_ID, user_name, path, {}, flag, true, {}) {}

Status DeleteResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadBool(&exists));

  return Status::OK();
}

ExistsRequest::ExistsRequest(const string &user_name, const string &path)
    : PathCtrlRequest(EXISTS_ID, user_name, path, {}, false, true, {}) {}

Status ExistsResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadBool(&exists));

  return Status::OK();
}

HandshakeRequest::HandshakeRequest(const string &fs_name, const string &log_dir)
    : Request(HANDSHAKE_ID), fs_name_(fs_name), log_dir_(log_dir) {}

Status HandshakeRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(Request::Write(client));

  TF_RETURN_IF_ERROR(client->WriteString(fs_name_));
  TF_RETURN_IF_ERROR(client->WriteString(log_dir_));

  return Status::OK();
}

Status HandshakeResponse::Read(ExtendedTCPClient *client) {
  int64_t block_size;
  bool sampling;

  TF_RETURN_IF_ERROR(client->ReadNullableString(&fs_name));
  TF_RETURN_IF_ERROR(client->ReadLong(&block_size));

  bool has_sampling_;
  TF_RETURN_IF_ERROR(client->ReadBool(&has_sampling_));

  if (has_sampling_) {
    TF_RETURN_IF_ERROR(client->ReadBool(&sampling));
  }

  return Status::OK();
}

ListRequest::ListRequest(int32_t command_id_, const string &user_name,
                         const string &path)
    : PathCtrlRequest(command_id_, user_name, path, {}, false, true, {}) {}

ListFilesRequest::ListFilesRequest(const string &user_name, const string &path)
    : ListRequest(LIST_FILES_ID, user_name, path) {}

ListPathsRequest::ListPathsRequest(const string &user_name, const string &path)
    : ListRequest(LIST_PATHS_ID, user_name, path) {}

OpenCreateRequest::OpenCreateRequest(const string &user_name,
                                     const string &path)
    : PathCtrlRequest(OPEN_CREATE_ID, user_name, path, {}, false, true, {}) {}

Status OpenCreateRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(PathCtrlRequest::Write(client));

  TF_RETURN_IF_ERROR(client->WriteInt(replication_));
  TF_RETURN_IF_ERROR(client->WriteLong(blockSize_));

  return Status::OK();
}

Status OpenCreateResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadLong(&stream_id));

  return Status::OK();
}

OpenAppendRequest::OpenAppendRequest(const string &user_name,
                                     const string &path)
    : PathCtrlRequest(OPEN_APPEND_ID, user_name, path, {}, false, true, {}) {}

Status OpenAppendRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(PathCtrlRequest::Write(client));

  return Status::OK();
}

Status OpenAppendResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadLong(&stream_id));

  return Status::OK();
}

OpenReadRequest::OpenReadRequest(const string &user_name, const string &path,
                                 bool flag,
                                 int32_t sequential_reads_before_prefetch)
    : PathCtrlRequest(OPEN_READ_ID, user_name, path, {}, flag, true, {}),
      sequential_reads_before_prefetch_(sequential_reads_before_prefetch) {}

OpenReadRequest::OpenReadRequest(const string &user_name, const string &path)
    : OpenReadRequest(user_name, path, false, 0) {}

Status OpenReadRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(PathCtrlRequest::Write(client));

  if (flag_) {
    TF_RETURN_IF_ERROR(client->WriteInt(sequential_reads_before_prefetch_));
  }

  return Status::OK();
}

Status OpenReadResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadLong(&stream_id));
  TF_RETURN_IF_ERROR(client->ReadLong(&length));

  return Status::OK();
}

InfoRequest::InfoRequest(const string &user_name, const string &path)
    : PathCtrlRequest(INFO_ID, user_name, path, {}, false, true, {}) {}

Status InfoResponse::Read(ExtendedTCPClient *client) {
  file_info = IGFSFile();
  TF_RETURN_IF_ERROR(file_info.Read(client));

  return Status::OK();
}

MakeDirectoriesRequest::MakeDirectoriesRequest(const string &user_name,
                                               const string &path)
    : PathCtrlRequest(MKDIR_ID, user_name, path, {}, false, true, {}) {}

Status MakeDirectoriesResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadBool(&successful));

  return Status::OK();
}

CloseRequest::CloseRequest(int64_t streamId)
    : StreamCtrlRequest(CLOSE_ID, streamId, 0) {}

Status CloseResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadBool(&successful));

  return Status::OK();
}

ReadBlockRequest::ReadBlockRequest(int64_t stream_id, int64_t pos,
                                   int32_t length)
    : StreamCtrlRequest(READ_BLOCK_ID, stream_id, length), pos(pos) {}

Status ReadBlockRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(StreamCtrlRequest::Write(client));

  TF_RETURN_IF_ERROR(client->WriteLong(pos));

  return Status::OK();
}

Status ReadBlockResponse::Read(ExtendedTCPClient *client, int32_t length,
                               uint8_t *dst) {
  TF_RETURN_IF_ERROR(client->ReadData(dst, length));
  successfully_read = length;

  return Status::OK();
}

Status ReadBlockResponse::Read(ExtendedTCPClient *client) {
  return Status::OK();
}

std::streamsize ReadBlockResponse::GetSuccessfullyRead() {
  return successfully_read;
}

ReadBlockCtrlResponse::ReadBlockCtrlResponse(uint8_t *dst)
    : CtrlResponse(false), dst(dst) {}

Status ReadBlockCtrlResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(Response::Read(client));

  res = ReadBlockResponse();
  TF_RETURN_IF_ERROR(res.Read(client, length, dst));

  return Status::OK();
}

WriteBlockRequest::WriteBlockRequest(int64_t stream_id, const uint8_t *data,
                                     int32_t length)
    : StreamCtrlRequest(WRITE_BLOCK_ID, stream_id, length), data(data) {}

Status WriteBlockRequest::Write(ExtendedTCPClient *client) const {
  TF_RETURN_IF_ERROR(StreamCtrlRequest::Write(client));
  TF_RETURN_IF_ERROR(client->WriteData((uint8_t *)data, length_));

  return Status::OK();
}

RenameRequest::RenameRequest(const string &user_name, const string &path,
                             const string &destination_path)
    : PathCtrlRequest(RENAME_ID, user_name, path, destination_path, false, true,
                      {}) {}

Status RenameResponse::Read(ExtendedTCPClient *client) {
  TF_RETURN_IF_ERROR(client->ReadBool(&successful));

  return Status::OK();
}

}  // namespace tensorflow
