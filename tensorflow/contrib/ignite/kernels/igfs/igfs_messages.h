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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_MESSAGES_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_MESSAGES_H_

#include "tensorflow/contrib/ignite/kernels/igfs/igfs_extended_tcp_client.h"

namespace tensorflow {

enum CommandId {
  HANDSHAKE_ID = 0,
  EXISTS_ID = 2,
  INFO_ID = 3,
  RENAME_ID = 6,
  DELETE_ID = 7,
  MKDIR_ID = 8,
  LIST_PATHS_ID = 9,
  LIST_FILES_ID = 10,
  OPEN_READ_ID = 13,
  OPEN_APPEND_ID = 14,
  OPEN_CREATE_ID = 15,
  CLOSE_ID = 16,
  READ_BLOCK_ID = 17,
  WRITE_BLOCK_ID = 18,
};

class IGFSPath {
 public:
  Status Read(ExtendedTCPClient *client);

  string path;
};

class IGFSFile {
 public:
  Status Read(ExtendedTCPClient *client);

  int64_t length;
  int64_t modification_time;
  uint8_t flags;
};

class Request {
 public:
  Request(int32_t command_id);
  virtual Status Write(ExtendedTCPClient *client) const;

 protected:
  const int32_t command_id_;
};

class Response {
 public:
  virtual Status Read(ExtendedTCPClient *client);

  int32_t res_type;
  int32_t req_id;
  int32_t length;

 protected:
  static const int32_t header_size_ = 24;
  static const int32_t response_header_size_ = 9;
};

class PathCtrlRequest : public Request {
 public:
  PathCtrlRequest(int32_t command_id, const string &user_name,
                  const string &path, const string &destination_path, bool flag,
                  bool collocate, const std::map<string, string> &properties);
  Status Write(ExtendedTCPClient *client) const override;

 protected:
  Status WritePath(ExtendedTCPClient *client, const string &path) const;

  const string user_name_;
  const string path_;
  const string destination_path_;
  const bool flag_;
  const bool collocate_;
  const std::map<string, string> props_;
};

class StreamCtrlRequest : public Request {
 public:
  StreamCtrlRequest(int32_t command_id, int64_t stream_id, int32_t length);
  Status Write(ExtendedTCPClient *client) const override;

 protected:
  int64_t stream_id_;
  int32_t length_;
};

template <class R>
class CtrlResponse : public Response {
 public:
  CtrlResponse(bool optional) : optional_(optional) {}
  Status Read(ExtendedTCPClient *client) override {
    TF_RETURN_IF_ERROR(Response::Read(client));

    if (optional_) {
      TF_RETURN_IF_ERROR(client->ReadBool(&has_content));

      if (!has_content) return Status::OK();
    }

    res = R();
    has_content = true;
    TF_RETURN_IF_ERROR(res.Read(client));

    return Status::OK();
  }

  R res;
  bool has_content;

 private:
  bool optional_;
};

template <class T>
class ListResponse {
 public:
  Status Read(ExtendedTCPClient *client) {
    int32_t len;
    TF_RETURN_IF_ERROR(client->ReadInt(&len));

    entries.clear();

    for (int32_t i = 0; i < len; i++) {
      T f = {};
      TF_RETURN_IF_ERROR(f.Read(client));
      entries.push_back(f);
    }

    return Status::OK();
  }

  std::vector<T> entries;
};

class DeleteRequest : public PathCtrlRequest {
 public:
  DeleteRequest(const string &user_name, const string &path, bool flag);
};

class DeleteResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  bool exists;
};

class ExistsRequest : public PathCtrlRequest {
 public:
  explicit ExistsRequest(const string &user_name, const string &path);
};

class ExistsResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  bool exists;
};

class HandshakeRequest : public Request {
 public:
  HandshakeRequest(const string &fs_name, const string &log_dir);
  Status Write(ExtendedTCPClient *client) const override;

 private:
  string fs_name_;
  string log_dir_;
};

class HandshakeResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  string fs_name;
};

class ListRequest : public PathCtrlRequest {
 public:
  explicit ListRequest(int32_t command_id, const string &user_name,
                       const string &path);
};

class ListFilesRequest : public ListRequest {
 public:
  ListFilesRequest(const string &user_name, const string &path);
};

class ListFilesResponse : public ListResponse<IGFSFile> {};

class ListPathsRequest : public ListRequest {
 public:
  ListPathsRequest(const string &user_name, const string &path);
};

class ListPathsResponse : public ListResponse<IGFSPath> {};

class OpenCreateRequest : public PathCtrlRequest {
 public:
  OpenCreateRequest(const string &user_name, const string &path);
  Status Write(ExtendedTCPClient *client) const override;

 private:
  int32_t replication_;
  int64_t blockSize_;
};

class OpenCreateResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  int64_t stream_id;
};

class OpenAppendRequest : public PathCtrlRequest {
 public:
  explicit OpenAppendRequest(const string &user_name, const string &path);
  Status Write(ExtendedTCPClient *client) const override;
};

class OpenAppendResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  int64_t stream_id;
};

class OpenReadRequest : public PathCtrlRequest {
 public:
  OpenReadRequest(const string &user_name, const string &path, bool flag,
                  int32_t seqReadsBeforePrefetch);
  OpenReadRequest(const string &user_name, const string &path);
  Status Write(ExtendedTCPClient *client) const override;

 protected:
  /** Sequential reads before prefetch. */
  int32_t sequential_reads_before_prefetch_;
};

class OpenReadResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  int64_t stream_id;
  int64_t length;
};

class InfoRequest : public PathCtrlRequest {
 public:
  InfoRequest(const string &user_name, const string &path);
};

class InfoResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  IGFSFile file_info;
};

class MakeDirectoriesRequest : public PathCtrlRequest {
 public:
  MakeDirectoriesRequest(const string &userName, const string &path);
};

class MakeDirectoriesResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  bool successful;
};

/** Stream control requests. **/

class CloseRequest : public StreamCtrlRequest {
 public:
  explicit CloseRequest(int64_t stream_id);
};

class CloseResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  bool successful;
};

class ReadBlockRequest : public StreamCtrlRequest {
 public:
  ReadBlockRequest(int64_t stream_id, int64_t pos, int32_t length);
  Status Write(ExtendedTCPClient *client) const override;

 private:
  int64_t pos;
};

class ReadBlockResponse {
 public:
  Status Read(ExtendedTCPClient *client, int32_t length, uint8_t *dst);
  Status Read(ExtendedTCPClient *client);
  std::streamsize GetSuccessfullyRead();

 private:
  int32_t length;
  std::streamsize successfully_read;
};

class ReadBlockCtrlResponse : public CtrlResponse<ReadBlockResponse> {
 public:
  ReadBlockCtrlResponse(uint8_t *dst);
  Status Read(ExtendedTCPClient *client) override;

 private:
  uint8_t *dst;
};

class WriteBlockRequest : public StreamCtrlRequest {
 public:
  WriteBlockRequest(int64_t stream_id, const uint8_t *data, int32_t length);
  Status Write(ExtendedTCPClient *client) const override;

 private:
  const uint8_t *data;
};

class RenameRequest : public PathCtrlRequest {
 public:
  RenameRequest(const string &user_name, const string &path,
                const string &destination_path);
};

class RenameResponse {
 public:
  Status Read(ExtendedTCPClient *client);

  bool successful;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_IGFS_IGFS_MESSAGES_H_
