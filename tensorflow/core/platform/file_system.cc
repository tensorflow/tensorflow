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
#include <deque>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

FileSystem::~FileSystem() {}

string FileSystem::TranslateName(const string& name) const {
  return io::CleanPath(name);
}

Status FileSystem::IsDirectory(const string& name) {
  // Check if path exists.
  if (!FileExists(name)) {
    return Status(tensorflow::error::NOT_FOUND, "Path not found");
  }
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
}

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

FileSystemRegistry::~FileSystemRegistry() {}

void ParseURI(StringPiece remaining, StringPiece* scheme, StringPiece* host,
              StringPiece* path) {
  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  if (!strings::Scanner(remaining)
           .One(strings::Scanner::LETTER)
           .Many(strings::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral("://")
           .GetResult(&remaining, scheme)) {
    // If there's no scheme, assume the entire string is a path.
    scheme->clear();
    host->clear();
    *path = remaining;
    return;
  }

  // 1. Parse host
  if (!strings::Scanner(remaining).ScanUntil('/').GetResult(&remaining, host)) {
    // No path, so the rest of the URI is the host.
    *host = remaining;
    path->clear();
    return;
  }

  // 2. The rest is the path
  *path = remaining;
}

string CreateURI(StringPiece scheme, StringPiece host, StringPiece path) {
  if (scheme.empty()) {
    return path.ToString();
  }
  return strings::StrCat(scheme, "://", host, path);
}

Status FileSystem::GetMatchingPaths(const string& pattern,
                                    std::vector<string>* results) {
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  const string& fixed_prefix =
      pattern.substr(0, pattern.find_first_of("*?[\\"));
  std::vector<string> all_files;
  string dir = io::Dirname(fixed_prefix).ToString();
  if (dir.empty()) dir = ".";

  // Setup a BFS to explore everything under dir.
  std::deque<string> dir_q;
  dir_q.push_back(dir);
  Status ret;  // Status to return.
  while (!dir_q.empty()) {
    string current_dir = dir_q.front();
    dir_q.pop_front();
    std::vector<string> children;
    Status s = GetChildren(current_dir, &children);
    ret.Update(s);
    for (const string& child : children) {
      const string child_path = io::JoinPath(current_dir, child);
      // If the child is a directory add it to the queue.
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      }
      all_files.push_back(child_path);
    }
  }

  // Match all obtained files to the input pattern.
  for (const auto& f : all_files) {
    if (Env::Default()->MatchPath(f, pattern)) {
      results->push_back(f);
    }
  }
  return ret;
}

Status FileSystem::DeleteRecursively(const string& dirname,
                                     int64* undeleted_files,
                                     int64* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  if (!FileExists(dirname)) {
    (*undeleted_dirs)++;
    return Status(error::NOT_FOUND, "Directory doesn't exist");
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
  ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!FileExists(CreateURI(scheme, host, remaining_dir)) &&
         !remaining_dir.empty()) {
    // Basename returns "" for / ending dirs.
    if (!remaining_dir.ends_with("/")) {
      sub_dirs.push_back(io::Basename(remaining_dir));
    }
    remaining_dir = io::Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path = remaining_dir.ToString();
  for (const StringPiece sub_dir : sub_dirs) {
    built_path = io::JoinPath(built_path, sub_dir);
    TF_RETURN_IF_ERROR(CreateDir(CreateURI(scheme, host, built_path)));
  }
  return Status::OK();
}

}  // namespace tensorflow
