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

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"

#include "tensorflow/contrib/ignite/kernels/igfs/igfs.h"
#include "tensorflow/contrib/ignite/kernels/igfs/igfs_client.h"
#include "tensorflow/contrib/ignite/kernels/igfs/igfs_random_access_file.h"
#include "tensorflow/contrib/ignite/kernels/igfs/igfs_writable_file.h"

namespace tensorflow {

static string GetEnvOrElse(const string &env, string default_value) {
  const char *env_c_str = env.c_str();
  return getenv(env_c_str) != nullptr ? getenv(env_c_str) : default_value;
}

static string MakeRelative(const string &a, const string &b) {
  string max = a;
  string min = b;
  bool first = b.size() > a.size();

  if (first) {
    max = b;
    min = a;
  }

  auto r = mismatch(min.begin(), min.end(), max.begin());
  return string((first ? r.first : r.second), first ? min.end() : max.end());
}

string IGFS::TranslateName(const string &name) const {
  StringPiece scheme, namenode, path;
  io::ParseURI(name, &scheme, &namenode, &path);
  return string(path.data(), path.length());
}

IGFS::IGFS()
    : host_(GetEnvOrElse("IGFS_HOST", "localhost")),
      port_([] {
        int port;
        if (strings::safe_strto32(GetEnvOrElse("IGFS_PORT", "10500").c_str(),
                                  &port)) {
          return port;
        } else {
          LOG(WARNING)
              << "IGFS_PORT environment variable had an invalid value: "
              << getenv("IGFS_PORT") << "\nUsing default port 10500.";
          return 10500;
        }
      }()),
      fs_name_(GetEnvOrElse("IGFS_FS_NAME", "default_fs")) {
  LOG(INFO) << "IGFS created [host=" << host_ << ", port=" << port_
            << ", fs_name=" << fs_name_ << "]";
}

IGFS::~IGFS() {
  LOG(INFO) << "IGFS destroyed [host=" << host_ << ", port=" << port_
            << ", fs_name=" << fs_name_ << "]";
}

Status IGFS::NewRandomAccessFile(const string &file_name,
                                 std::unique_ptr<RandomAccessFile> *result) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<OpenReadResponse> open_read_response(true);
  TF_RETURN_IF_ERROR(client->OpenRead(&open_read_response, path));

  int64 resource_id = open_read_response.res.stream_id;
  result->reset(new IGFSRandomAccessFile(path, resource_id, std::move(client)));

  LOG(INFO) << "New random access file completed successfully [file_name="
            << file_name << "]";

  return Status::OK();
}

Status IGFS::NewWritableFile(const string &file_name,
                             std::unique_ptr<WritableFile> *result) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<ExistsResponse> exists_response(false);
  TF_RETURN_IF_ERROR(client->Exists(&exists_response, path));

  if (exists_response.res.exists) {
    CtrlResponse<DeleteResponse> del_response(false);
    TF_RETURN_IF_ERROR(client->Delete(&del_response, path, false));
  }

  CtrlResponse<OpenCreateResponse> open_create_resp(false);
  TF_RETURN_IF_ERROR(client->OpenCreate(&open_create_resp, path));

  int64 resource_id = open_create_resp.res.stream_id;
  result->reset(new IGFSWritableFile(path, resource_id, std::move(client)));

  LOG(INFO) << "New writable file completed successfully [file_name="
            << file_name << "]";

  return Status::OK();
}

Status IGFS::NewAppendableFile(const string &file_name,
                               std::unique_ptr<WritableFile> *result) {
  std::unique_ptr<IGFSClient> client = CreateClient();

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<ExistsResponse> exists_response(false);
  TF_RETURN_IF_ERROR(client->Exists(&exists_response, file_name));

  if (exists_response.res.exists) {
    CtrlResponse<DeleteResponse> del_response(false);
    TF_RETURN_IF_ERROR(client->Delete(&del_response, file_name, false));
  }

  CtrlResponse<OpenAppendResponse> open_append_resp(false);
  TF_RETURN_IF_ERROR(client->OpenAppend(&open_append_resp, file_name));

  result->reset(new IGFSWritableFile(TranslateName(file_name),
                                     open_append_resp.res.stream_id,
                                     std::move(client)));

  LOG(INFO) << "New appendable file completed successfully [file_name="
            << file_name << "]";

  return Status::OK();
}

Status IGFS::NewReadOnlyMemoryRegionFromFile(
    const string &file_name, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  return errors::Unimplemented("IGFS does not support ReadOnlyMemoryRegion");
}

Status IGFS::FileExists(const string &file_name) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  const string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<ExistsResponse> exists_response(false);
  TF_RETURN_IF_ERROR(client->Exists(&exists_response, path));

  if (!exists_response.res.exists)
    return errors::NotFound("File ", path, " not found");

  LOG(INFO) << "File exists completed successfully [file_name=" << file_name
            << "]";

  return Status::OK();
}

Status IGFS::GetChildren(const string &file_name, std::vector<string> *result) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);
  path = path + "/";

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<ListPathsResponse> list_paths_response(false);
  TF_RETURN_IF_ERROR(client->ListPaths(&list_paths_response, path));

  *result = std::vector<string>();
  std::vector<IGFSPath> entries = list_paths_response.res.entries;

  for (IGFSPath &value : entries)
    result->push_back(MakeRelative(value.path, path));

  LOG(INFO) << "Get children completed successfully [file_name=" << file_name
            << "]";

  return Status::OK();
}

Status IGFS::GetMatchingPaths(const string &pattern,
                              std::vector<string> *results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status IGFS::DeleteFile(const string &file_name) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<DeleteResponse> del_response(false);
  TF_RETURN_IF_ERROR(client->Delete(&del_response, path, false));

  if (!del_response.res.exists)
    return errors::NotFound("File ", path, " not found");

  LOG(INFO) << "Delete file completed successfully [file_name=" << file_name
            << "]";

  return Status::OK();
}

Status IGFS::CreateDir(const string &file_name) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  const string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<MakeDirectoriesResponse> mkdir_response(false);
  TF_RETURN_IF_ERROR(client->MkDir(&mkdir_response, path));

  if (!mkdir_response.res.successful)
    return errors::Unknown("Can't create directory ", path);

  LOG(INFO) << "Create dir completed successful [file_name=" << file_name
            << "]";

  return Status::OK();
}

Status IGFS::DeleteDir(const string &file_name) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<ListFilesResponse> list_files_response(false);
  TF_RETURN_IF_ERROR(client->ListFiles(&list_files_response, path));

  if (!list_files_response.res.entries.empty()) {
    return errors::FailedPrecondition("Can't delete a non-empty directory");
  } else {
    CtrlResponse<DeleteResponse> del_response(false);
    TF_RETURN_IF_ERROR(client->Delete(&del_response, path, true));
  }

  LOG(INFO) << "Delete dir completed successful [file_name=" << file_name
            << "]";

  return Status::OK();
}

Status IGFS::GetFileSize(const string &file_name, uint64 *size) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<InfoResponse> info_response(false);
  TF_RETURN_IF_ERROR(client->Info(&info_response, path));

  *size = info_response.res.file_info.length;

  LOG(INFO) << "Get file size completed successful [file_name=" << file_name
            << "]";

  return Status::OK();
}

Status IGFS::RenameFile(const string &src, const string &dst) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string src_path = TranslateName(src);
  string dst_path = TranslateName(dst);

  if (FileExists(dst).ok()) DeleteFile(dst);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<RenameResponse> rename_response(false);
  TF_RETURN_IF_ERROR(client->Rename(&rename_response, src_path, dst_path));

  if (!rename_response.res.successful)
    return errors::NotFound("File ", src_path, " not found");

  LOG(INFO) << "Rename file completed successful [src=" << src
            << ", dst=" << dst << "]";

  return Status::OK();
}

Status IGFS::Stat(const string &file_name, FileStatistics *stats) {
  std::unique_ptr<IGFSClient> client = CreateClient();
  string path = TranslateName(file_name);

  CtrlResponse<HandshakeResponse> handshake_response(true);
  TF_RETURN_IF_ERROR(client->Handshake(&handshake_response));

  CtrlResponse<InfoResponse> info_response(false);
  TF_RETURN_IF_ERROR(client->Info(&info_response, path));

  IGFSFile info = info_response.res.file_info;

  *stats = FileStatistics(info.length, info.modification_time * 1000000,
                          (info.flags & 0x1) != 0);

  LOG(INFO) << "Stat completed successful [file_name=" << file_name << "]";

  return Status::OK();
}

std::unique_ptr<IGFSClient> IGFS::CreateClient() const {
  return std::unique_ptr<IGFSClient>(
      new IGFSClient(host_, port_, fs_name_, ""));
}

}  // namespace tensorflow
