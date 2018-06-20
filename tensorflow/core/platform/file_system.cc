/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <sys/stat.h>
#include <algorithm>
#include <deque>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/platform.h"

namespace tensorflow {

FileSystem::~FileSystem() {}

string FileSystem::TranslateName(const string& name) const {
  // If the name is empty, CleanPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;
  return io::CleanPath(name);
}

Status FileSystem::IsDirectory(const string& name) {
  // Check if path exists.
  TF_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
}

void FileSystem::FlushCaches() {}

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

FileSystemRegistry::~FileSystemRegistry() {}

bool FileSystem::FilesExist(const std::vector<string>& files,
                            std::vector<Status>* status) {
  bool result = true;
  for (const auto& file : files) {
    Status s = FileExists(file);
    result &= s.ok();
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      // Return early since there is no need to check other files.
      return false;
    }
  }
  return result;
}

Status FileSystem::DeleteRecursively(const string& dirname,
                                     int64* undeleted_files,
                                     int64* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }
  std::deque<string> dir_q;      // Queue for the BFS
  std::vector<string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    Status s = GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const string& child : children) {
      const string child_path = io::JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        ret.Update(del_status);
        if (!del_status.ok()) {
          (*undeleted_files)++;
        }
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}

Status FileSystem::RecursivelyCreateDir(const string& dirname) {
  StringPiece scheme, host, remaining_dir;
  io::ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!remaining_dir.empty()) {
    Status status = FileExists(io::CreateURI(scheme, host, remaining_dir));
    if (status.ok()) {
      break;
    }
    if (status.code() != error::Code::NOT_FOUND) {
      return status;
    }
    // Basename returns "" for / ending dirs.
    if (!str_util::EndsWith(remaining_dir, "/")) {
      sub_dirs.push_back(io::Basename(remaining_dir));
    }
    remaining_dir = io::Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path = std::string(remaining_dir);
  for (const StringPiece sub_dir : sub_dirs) {
    built_path = io::JoinPath(built_path, sub_dir);
    Status status = CreateDir(io::CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK();
}

Status FileSystem::CopyFile(const string& src, const string& target) {
  return FileSystemCopyFile(this, src, this, target);
}

}  // namespace tensorflow
