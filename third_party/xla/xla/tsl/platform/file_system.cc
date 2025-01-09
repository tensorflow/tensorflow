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

#include "xla/tsl/platform/file_system.h"

#include <sys/stat.h>

#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "tsl/platform/status.h"

#if defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM) || \
    defined(PLATFORM_GOOGLE)
#include <fnmatch.h>
#else
#include "tsl/platform/regexp.h"
#endif  // defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM) || \
        // defined(PLATFORM_GOOGLE)

#include "xla/tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/scanner.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/strcat.h"

namespace tsl {

bool FileSystem::Match(const string& filename, const string& pattern) {
#if defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM) || \
    defined(PLATFORM_GOOGLE)
  // We avoid relying on RE2 on mobile platforms, because it incurs a
  // significant binary size increase.
  // For POSIX platforms, there is no need to depend on RE2 if `fnmatch` can be
  // used safely.
  return fnmatch(pattern.c_str(), filename.c_str(), FNM_PATHNAME) == 0;
#else
  string regexp(pattern);
  regexp = str_util::StringReplace(regexp, "*", "[^/]*", true);
  regexp = str_util::StringReplace(regexp, "?", ".", true);
  regexp = str_util::StringReplace(regexp, "(", "\\(", true);
  regexp = str_util::StringReplace(regexp, ")", "\\)", true);
  return RE2::FullMatch(filename, regexp);
#endif  // defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM) || \
        // defined(PLATFORM_GOOGLE)
}

string FileSystem::TranslateName(const string& name) const {
  // If the name is empty, CleanPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;

  // Otherwise, properly separate the URI components and clean the path one
  absl::string_view scheme, host, path;
  this->ParseURI(name, &scheme, &host, &path);

  // If `path` becomes empty, return `/` (`file://` should be `/`), not `.`.
  if (path.empty()) return "/";

  return this->CleanPath(path);
}

absl::Status FileSystem::IsDirectory(const string& name,
                                     TransactionToken* token) {
  // Check if path exists.
  // TODO(sami):Forward token to other methods once migration is complete.
  TF_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return absl::OkStatus();
  }
  return absl::Status(absl::StatusCode::kFailedPrecondition, "Not a directory");
}

absl::Status FileSystem::HasAtomicMove(const string& path,
                                       bool* has_atomic_move) {
  *has_atomic_move = true;
  return absl::OkStatus();
}

absl::Status FileSystem::CanCreateTempFile(const std::string& fname,
                                           bool* can_create_temp_file) {
  *can_create_temp_file = true;
  return absl::OkStatus();
}

void FileSystem::FlushCaches(TransactionToken* token) {}

bool FileSystem::FilesExist(const std::vector<string>& files,
                            TransactionToken* token,
                            std::vector<absl::Status>* status) {
  bool result = true;
  for (const auto& file : files) {
    absl::Status s = FileExists(file);
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

absl::Status FileSystem::DeleteRecursively(const string& dirname,
                                           TransactionToken* token,
                                           int64_t* undeleted_files,
                                           int64_t* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  absl::Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }

  // If given path to a single file, we should just delete it.
  if (!IsDirectory(dirname).ok()) {
    absl::Status delete_root_status = DeleteFile(dirname);
    if (!delete_root_status.ok()) (*undeleted_files)++;
    return delete_root_status;
  }

  std::deque<string> dir_q;      // Queue for the BFS
  std::vector<string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  absl::Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    absl::Status s = GetChildren(dir, &children);
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
        absl::Status del_status = DeleteFile(child_path);
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
    absl::Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}

absl::Status FileSystem::RecursivelyCreateDir(const string& dirname,
                                              TransactionToken* token) {
  absl::string_view scheme, host, remaining_dir;
  this->ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<absl::string_view> sub_dirs;
  while (!remaining_dir.empty()) {
    std::string current_entry = this->CreateURI(scheme, host, remaining_dir);
    absl::Status exists_status = FileExists(current_entry);
    if (exists_status.ok()) {
      // FileExists cannot differentiate between existence of a file or a
      // directory, hence we need an additional test as we must not assume that
      // a path to a file is a path to a parent directory.
      absl::Status directory_status = IsDirectory(current_entry);
      if (directory_status.ok()) {
        break;  // We need to start creating directories from here.
      } else if (directory_status.code() == absl::StatusCode::kUnimplemented) {
        return directory_status;
      } else {
        return errors::FailedPrecondition(remaining_dir, " is not a directory");
      }
    }
    if (exists_status.code() != error::Code::NOT_FOUND) {
      return exists_status;
    }
    // Basename returns "" for / ending dirs.
    if (!absl::EndsWith(remaining_dir, "/")) {
      sub_dirs.push_back(this->Basename(remaining_dir));
    }
    remaining_dir = this->Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path(remaining_dir);
  for (const absl::string_view sub_dir : sub_dirs) {
    built_path = this->JoinPath(built_path, sub_dir);
    absl::Status status = CreateDir(this->CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != absl::StatusCode::kAlreadyExists) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status FileSystem::CopyFile(const string& src, const string& target,
                                  TransactionToken* token) {
  return FileSystemCopyFile(this, src, this, target);
}

char FileSystem::Separator() const { return '/'; }

string FileSystem::JoinPathImpl(
    std::initializer_list<absl::string_view> paths) {
  string result;

  for (absl::string_view path : paths) {
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

std::pair<absl::string_view, absl::string_view> FileSystem::SplitPath(
    absl::string_view uri) const {
  absl::string_view scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  // We have 3 cases of results from `ParseURI`:
  //
  //   1. `path` is empty (`uri` is something like http://google.com/)
  //      Here, we don't have anything to split, so return empty components
  //
  //   2. all 3 components are non-empty (`uri` is something like
  //      http://google.com/path/to/resource)
  //      Here, all 3 components point to elements inside the same buffer as
  //      `uri`. In the given example, `scheme` contains `http://`, `host`
  //      contains `google.com/` and `path` contains `path/to/resource`.
  //      Since all 3 components point to the same buffer, we can do arithmetic
  //      such as `host.end() - uri.begin()` because we know for sure that
  //      `host` starts after `uri`.
  //
  //   3. `scheme` and `host` are empty (`uri` is local file, like /etc/passwd)
  //      Here, we split `path`, but we need to be careful with pointer
  //      arithmetic. Here we only know that `path` and `uri` represent the
  //      exact same buffer.
  //
  // To summarize, if `path` is empty there is nothing to return, in all other
  // cases we can do arithmetic involving `path` and `uri` but if
  // `host`/`scheme` are involved we need to make sure these are not empty.

  // Case 1 above
  if (path.empty()) {
    return std::make_pair(absl::string_view(), absl::string_view());
  }

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
  if (pos == absl::string_view::npos) {
    if (host.empty()) {
      // Case 3 above, `uri` and `path` point to the same thing
      // We are returning all of the `path` as basename here.
      return std::make_pair(absl::string_view(), path);
    }

    // Safe to do this arithmetic here, we are in case 2 above
    return std::make_pair(
        absl::string_view(uri.data(), host.end() - uri.begin()), path);
  }

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0) {
    return std::make_pair(
        absl::string_view(uri.data(), path.begin() + 1 - uri.begin()),
        absl::string_view(path.data() + 1, path.size() - 1));
  }

  return std::make_pair(
      absl::string_view(uri.data(), path.begin() + pos - uri.begin()),
      absl::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}

bool FileSystem::IsAbsolutePath(absl::string_view path) const {
  return !path.empty() && path[0] == '/';
}

absl::string_view FileSystem::Dirname(absl::string_view path) const {
  return this->SplitPath(path).first;
}

absl::string_view FileSystem::Basename(absl::string_view path) const {
  return this->SplitPath(path).second;
}

absl::string_view FileSystem::Extension(absl::string_view path) const {
  absl::string_view basename = this->Basename(path);

  size_t pos = basename.rfind('.');
  if (pos == absl::string_view::npos) {
    return absl::string_view(path.data() + path.size(), 0);
  } else {
    return absl::string_view(path.data() + pos + 1, path.size() - (pos + 1));
  }
}

string FileSystem::CleanPath(absl::string_view unclean_path) const {
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

void FileSystem::ParseURI(absl::string_view remaining,
                          absl::string_view* scheme, absl::string_view* host,
                          absl::string_view* path) const {
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
    *scheme = absl::string_view();
    *host = absl::string_view();
    *path = remaining;
    return;
  }

  // 1. Parse host
  if (!strings::Scanner(remaining).ScanUntil('/').GetResult(&remaining, host)) {
    // No path, so the rest of the URI is the host.
    *host = remaining;
    *path = absl::string_view();
    return;
  }

  // 2. The rest is the path
  *path = remaining;
}

string FileSystem::CreateURI(absl::string_view scheme, absl::string_view host,
                             absl::string_view path) const {
  if (scheme.empty()) {
    return string(path);
  }
  return strings::StrCat(scheme, "://", host, path);
}

std::string FileSystem::DecodeTransaction(const TransactionToken* token) {
  // TODO(sami): Switch using StrCat when void* is supported
  if (token) {
    std::stringstream oss;
    oss << "Token= " << token->token << ", Owner=" << token->owner;
    return oss.str();
  }
  return "No Transaction";
}

}  // namespace tsl
