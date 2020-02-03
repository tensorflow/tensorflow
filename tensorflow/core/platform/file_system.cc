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

#include "tensorflow/core/platform/file_system.h"

#include <sys/stat.h>

#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

string FileSystem::TranslateName(const string& name) const {
  // If the name is empty, CleanPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;

  // Otherwise, properly separate the URI components and clean the path one
  StringPiece scheme, host, path;
  this->ParseURI(name, &scheme, &host, &path);

  // If `path` becomes empty, return `/` (`file://` should be `/`), not `.`.
  if (path.empty()) return "/";

  return this->CleanPath(path);
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

  // If given path to a single file, we should just delete it.
  if (!IsDirectory(dirname).ok()) {
    Status delete_root_status = DeleteFile(dirname);
    if (!delete_root_status.ok()) (*undeleted_files)++;
    return delete_root_status;
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
      const string child_path = this->JoinPath(dir, child);
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
  this->ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!remaining_dir.empty()) {
    std::string current_entry = this->CreateURI(scheme, host, remaining_dir);
    Status exists_status = FileExists(current_entry);
    if (exists_status.ok()) {
      // FileExists cannot differentiate between existence of a file or a
      // directory, hence we need an additional test as we must not assume that
      // a path to a file is a path to a parent directory.
      Status directory_status = IsDirectory(current_entry);
      if (directory_status.ok()) {
        break;  // We need to start creating directories from here.
      } else if (directory_status.code() == tensorflow::error::UNIMPLEMENTED) {
        return directory_status;
      } else {
        return errors::FailedPrecondition(remaining_dir, " is not a directory");
      }
    }
    if (exists_status.code() != error::Code::NOT_FOUND) {
      return exists_status;
    }
    // Basename returns "" for / ending dirs.
    if (!str_util::EndsWith(remaining_dir, "/")) {
      sub_dirs.push_back(this->Basename(remaining_dir));
    }
    remaining_dir = this->Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path(remaining_dir);
  for (const StringPiece sub_dir : sub_dirs) {
    built_path = this->JoinPath(built_path, sub_dir);
    Status status = CreateDir(this->CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK();
}

Status FileSystem::CopyFile(const string& src, const string& target) {
  return FileSystemCopyFile(this, src, this, target);
}

char FileSystem::Separator() const { return '/'; }

string FileSystem::JoinPathImpl(std::initializer_list<StringPiece> paths) {
  string result;

  for (StringPiece path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (this->IsAbsolutePath(path)) {
        strings::StrAppend(&result, path.substr(1));
      } else {
        strings::StrAppend(&result, path);
      }
    } else {
      if (this->IsAbsolutePath(path)) {
        strings::StrAppend(&result, path);
      } else {
        strings::StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

std::pair<StringPiece, StringPiece> FileSystem::SplitPath(
    StringPiece uri) const {
  StringPiece scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  size_t pos = path.rfind(this->Separator());

  // Our code assumes it is written for linux too many times. So, for windows
  // also check for '/'
#ifdef PLATFORM_WINDOWS
  size_t pos2 = path.rfind('/');
  // Pick the max value that is not string::npos.
  if (pos == string::npos) {
    pos = pos2;
  } else {
    if (pos2 != string::npos) {
      pos = pos > pos2 ? pos : pos2;
    }
  }
#endif

  // Handle the case with no SEP in 'path'.
  if (pos == StringPiece::npos)
    return std::make_pair(StringPiece(uri.begin(), host.end() - uri.begin()),
                          path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(
        StringPiece(uri.begin(), path.begin() + 1 - uri.begin()),
        StringPiece(path.data() + 1, path.size() - 1));

  return std::make_pair(
      StringPiece(uri.begin(), path.begin() + pos - uri.begin()),
      StringPiece(path.data() + pos + 1, path.size() - (pos + 1)));
}

bool FileSystem::IsAbsolutePath(StringPiece path) const {
  return !path.empty() && path[0] == '/';
}

StringPiece FileSystem::Dirname(StringPiece path) const {
  return this->SplitPath(path).first;
}

StringPiece FileSystem::Basename(StringPiece path) const {
  return this->SplitPath(path).second;
}

StringPiece FileSystem::Extension(StringPiece path) const {
  StringPiece basename = this->Basename(path);

  int pos = basename.rfind('.');
  if (pos == StringPiece::npos) {
    return StringPiece(path.data() + path.size(), 0);
  } else {
    return StringPiece(path.data() + pos + 1, path.size() - (pos + 1));
  }
}

string FileSystem::CleanPath(StringPiece unclean_path) const {
  string path(unclean_path);
  const char* src = path.c_str();
  string::iterator dst = path.begin();

  // Check for absolute path and determine initial backtrack limit.
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  string::const_iterator backtrack_limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != backtrack_limit) {
          // We can backtrack the previous part
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          // We can never backtrack over a copied "../" part so set new limit.
          backtrack_limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/') {
        *dst++ = *src++;
      }
      if (*src) {
        *dst++ = *src++;
      }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/') {
      ++src;
    }
  }

  // Calculate and check the length of the cleaned path.
  string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    // Remove trailing '/' except if it is root path ("/" ==> path_length := 1)
    if (path_length > 1 && path[path_length - 1] == '/') {
      --path_length;
    }
    path.resize(path_length);
  } else {
    // The cleaned path is empty; assign "." as per the spec.
    path.assign(1, '.');
  }
  return path;
}

void FileSystem::ParseURI(StringPiece remaining, StringPiece* scheme,
                          StringPiece* host, StringPiece* path) const {
  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  // Keep URI pattern in tensorboard/backend/server.py updated accordingly
  if (!strings::Scanner(remaining)
           .One(strings::Scanner::LETTER)
           .Many(strings::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral("://")
           .GetResult(&remaining, scheme)) {
    // If there's no scheme, assume the entire string is a path.
    *scheme = StringPiece(remaining.begin(), 0);
    *host = StringPiece(remaining.begin(), 0);
    *path = remaining;
    return;
  }

  // 1. Parse host
  if (!strings::Scanner(remaining).ScanUntil('/').GetResult(&remaining, host)) {
    // No path, so the rest of the URI is the host.
    *host = remaining;
    *path = StringPiece(remaining.end(), 0);
    return;
  }

  // 2. The rest is the path
  *path = remaining;
}

string FileSystem::CreateURI(StringPiece scheme, StringPiece host,
                             StringPiece path) const {
  if (scheme.empty()) {
    return string(path);
  }
  return strings::StrCat(scheme, "://", host, path);
}

}  // namespace tensorflow
