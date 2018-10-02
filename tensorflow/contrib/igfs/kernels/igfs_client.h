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

#ifndef TENSORFLOW_CONTRIB_IGFS_KERNELS_IGFS_CLIENT_H_
#define TENSORFLOW_CONTRIB_IGFS_KERNELS_IGFS_CLIENT_H_

#include "tensorflow/contrib/igfs/kernels/igfs_messages.h"

namespace tensorflow {

class IGFSClient {
 public:
  IGFSClient(const string &host, int port, const string &fs_name,
             const string &user_name);
  ~IGFSClient();

  inline Status Handshake(CtrlResponse<HandshakeResponse> *res) {
    return SendRequestGetResponse(HandshakeRequest(fs_name_, {}), res);
  }

  inline Status ListFiles(CtrlResponse<ListFilesResponse> *res,
                          const string &path) {
    return SendRequestGetResponse(ListFilesRequest(user_name_, path), res);
  }

  inline Status ListPaths(CtrlResponse<ListPathsResponse> *res,
                          const string &path) {
    return SendRequestGetResponse(ListPathsRequest(user_name_, path), res);
  }

  inline Status Info(CtrlResponse<InfoResponse> *res, const string &path) {
    return SendRequestGetResponse(InfoRequest(user_name_, path), res);
  }

  inline Status OpenCreate(CtrlResponse<OpenCreateResponse> *res,
                           const string &path) {
    return SendRequestGetResponse(OpenCreateRequest(user_name_, path), res);
  }

  inline Status OpenAppend(CtrlResponse<OpenAppendResponse> *res,
                           const string &path) {
    return SendRequestGetResponse(OpenAppendRequest(user_name_, path), res);
  }

  inline Status OpenRead(CtrlResponse<OpenReadResponse> *res,
                         const string &path) {
    return SendRequestGetResponse(OpenReadRequest(user_name_, path), res);
  }

  inline Status Exists(CtrlResponse<ExistsResponse> *res, const string &path) {
    return SendRequestGetResponse(ExistsRequest(user_name_, path), res);
  }

  inline Status MkDir(CtrlResponse<MakeDirectoriesResponse> *res,
                      const string &path) {
    return SendRequestGetResponse(MakeDirectoriesRequest(user_name_, path),
                                  res);
  }

  inline Status Delete(CtrlResponse<DeleteResponse> *res, const string &path,
                       bool recursive) {
    return SendRequestGetResponse(DeleteRequest(user_name_, path, recursive),
                                  res);
  }

  inline Status WriteBlock(int64_t stream_id, const uint8_t *data,
                           int32_t len) {
    return SendRequestGetResponse(WriteBlockRequest(stream_id, data, len),
                                  nullptr);
  }

  inline Status ReadBlock(ReadBlockCtrlResponse *res, int64_t stream_id,
                          int64_t pos, int32_t length) {
    return SendRequestGetResponse(ReadBlockRequest(stream_id, pos, length),
                                  res);
  }

  inline Status Close(CtrlResponse<CloseResponse> *res, int64_t stream_id) {
    return SendRequestGetResponse(CloseRequest(stream_id), res);
  }

  inline Status Rename(CtrlResponse<RenameResponse> *res, const string &source,
                       const string &dest) {
    return SendRequestGetResponse(RenameRequest(user_name_, source, dest), res);
  }

 private:
  Status SendRequestGetResponse(const Request &request, Response *response);

  const string fs_name_;
  const string user_name_;
  ExtendedTCPClient client_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGFS_KERNELS_IGFS_CLIENT_H_
