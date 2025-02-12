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

#include "tsl/platform/path.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/scanner.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/strcat.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {
namespace io {
namespace internal {
namespace {

const char kPathSep[] = "/";
}  // namespace

string JoinPathImpl(std::initializer_list<absl::string_view> paths) {
  string result;

  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
      continue;
    }

    if (IsAbsolutePath(path)) path = path.substr(1);

    if (result[result.size() - 1] == kPathSep[0]) {
      strings::StrAppend(&result, path);
    } else {
      strings::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

// Return the parts of the URI, split on the final "/" in the path. If there is
// no "/" in the path, the first part of the output is the scheme and host, and
// the second is the path. If the only "/" in the path is the first character,
// it is included in the first part of the output.
std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view uri) {
  absl::string_view scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  auto pos = path.rfind('/');
#ifdef PLATFORM_WINDOWS
  if (pos == StringPiece::npos) pos = path.rfind('\\');
#endif
  // Handle the case with no '/' in 'path'.
  if (pos == absl::string_view::npos)
    return std::make_pair(
        absl::string_view(uri.data(), host.end() - uri.begin()), path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(
        absl::string_view(uri.data(), path.begin() + 1 - uri.begin()),
        absl::string_view(path.data() + 1, path.size() - 1));

  return std::make_pair(
      absl::string_view(uri.data(), path.begin() + pos - uri.begin()),
      absl::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}

// Return the parts of the basename of path, split on the final ".".
// If there is no "." in the basename or "." is the final character in the
// basename, the second value will be empty.
std::pair<absl::string_view, absl::string_view> SplitBasename(
    absl::string_view path) {
  path = Basename(path);

  auto pos = path.rfind('.');
  if (pos == absl::string_view::npos)
    return std::make_pair(path,
                          absl::string_view(path.data() + path.size(), 0));
  return std::make_pair(
      absl::string_view(path.data(), pos),
      absl::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}

}  // namespace internal

bool IsAbsolutePath(absl::string_view path) {
  return !path.empty() && path[0] == '/';
}

absl::string_view Dirname(absl::string_view path) {
  return internal::SplitPath(path).first;
}

absl::string_view Basename(absl::string_view path) {
  return internal::SplitPath(path).second;
}

absl::string_view Extension(absl::string_view path) {
  return internal::SplitBasename(path).second;
}

absl::string_view BasenamePrefix(absl::string_view path) {
  return internal::SplitBasename(path).first;
}

string CleanPath(absl::string_view unclean_path) {
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

void ParseURI(absl::string_view uri, absl::string_view* scheme,
              absl::string_view* host, absl::string_view* path) {
  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  // Keep URI pattern in TensorBoard's `_parse_event_files_spec` updated
  // accordingly
  if (!strings::Scanner(uri)
           .One(strings::Scanner::LETTER)
           .Many(strings::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral("://")
           .GetResult(&uri, scheme)) {
    // If there's no scheme, assume the entire string is a path.
    *scheme = absl::string_view(uri.data(), 0);
    *host = absl::string_view(uri.data(), 0);
    *path = uri;
    return;
  }

  // 1. Parse host
  if (!strings::Scanner(uri).ScanUntil('/').GetResult(&uri, host)) {
    // No path, so the rest of the URI is the host.
    *host = uri;
    *path = absl::string_view();  // empty path
    return;
  }

  // 2. The rest is the path
  *path = uri;
}

string CreateURI(absl::string_view scheme, absl::string_view host,
                 absl::string_view path) {
  if (scheme.empty()) {
    return string(path);
  }
  return strings::StrCat(scheme, "://", host, path);
}

// Returns a unique number every time it is called.
int64_t UniqueId() {
  static mutex mu(LINKER_INITIALIZED);
  static int64_t id = 0;
  mutex_lock l(mu);
  return ++id;
}

string CommonPathPrefix(absl::Span<const string> paths) {
  if (paths.empty()) return "";
  size_t min_filename_size =
      absl::c_min_element(paths, [](const string& a, const string& b) {
        return a.size() < b.size();
      })->size();
  if (min_filename_size == 0) return "";

  size_t common_prefix_size = [&] {
    for (size_t prefix_size = 0; prefix_size < min_filename_size;
         prefix_size++) {
      char c = paths[0][prefix_size];
      for (int f = 1; f < paths.size(); f++) {
        if (paths[f][prefix_size] != c) {
          return prefix_size;
        }
      }
    }
    return min_filename_size;
  }();

  size_t rpos = absl::string_view(paths[0])
                    .substr(0, common_prefix_size)
                    .rfind(internal::kPathSep);
  return rpos == std::string::npos
             ? ""
             : std::string(absl::string_view(paths[0]).substr(0, rpos + 1));
}

string GetTempFilename(const string& extension) {
#if defined(__ANDROID__)
  LOG(FATAL) << "GetTempFilename is not implemented in this platform.";
#elif defined(PLATFORM_WINDOWS)
  char temp_dir[_MAX_PATH];
  DWORD retval;
  retval = GetTempPath(_MAX_PATH, temp_dir);
  if (retval > _MAX_PATH || retval == 0) {
    LOG(FATAL) << "Cannot get the directory for temporary files.";
  }

  char temp_file_name[_MAX_PATH];
  retval = GetTempFileName(temp_dir, "", UniqueId(), temp_file_name);
  if (retval > _MAX_PATH || retval == 0) {
    LOG(FATAL) << "Cannot get a temporary file in: " << temp_dir;
  }

  string full_tmp_file_name(temp_file_name);
  full_tmp_file_name.append(extension);
  return full_tmp_file_name;
#else
  for (const char* dir : std::vector<const char*>(
           {getenv("TEST_TMPDIR"), getenv("TMPDIR"), getenv("TMP"), "/tmp"})) {
    if (!dir || !dir[0]) {
      continue;
    }
    struct stat statbuf;
    if (!stat(dir, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      // UniqueId is added here because mkstemps is not as thread safe as it
      // looks. https://github.com/tensorflow/tensorflow/issues/5804 shows
      // the problem.
      string tmp_filepath;
      int fd;
      if (extension.length()) {
        tmp_filepath = io::JoinPath(
            dir, strings::StrCat("tmp_file_tensorflow_", UniqueId(), "_XXXXXX.",
                                 extension));
        fd = mkstemps(&tmp_filepath[0], extension.length() + 1);
      } else {
        tmp_filepath = io::JoinPath(
            dir,
            strings::StrCat("tmp_file_tensorflow_", UniqueId(), "_XXXXXX"));
        fd = mkstemp(&tmp_filepath[0]);
      }
      if (fd < 0) {
        LOG(FATAL) << "Failed to create temp file.";
      } else {
        if (close(fd) < 0) {
          LOG(ERROR) << "close() failed: " << strerror(errno);
        }
        return tmp_filepath;
      }
    }
  }
  LOG(FATAL) << "No temp directory found.";
  std::abort();
#endif
}

namespace {
// This is private to the file, because it's possibly too limited to be useful
// externally.
bool StartsWithSegment(absl::string_view path, absl::string_view segment) {
  return absl::StartsWith(path, segment) &&
         (path.size() == segment.size() ||
          path.at(segment.size()) == internal::kPathSep[0]);
}
}  // namespace

bool GetTestWorkspaceDir(string* dir) {
  const char* srcdir = getenv("TEST_SRCDIR");
  if (srcdir == nullptr) {
    return false;
  }
  const char* workspace = getenv("TEST_WORKSPACE");
  if (workspace == nullptr) {
    return false;
  }
  if (dir != nullptr) {
    *dir = tsl::io::JoinPath(srcdir, workspace);
  }
  return true;
}

bool GetTestUndeclaredOutputsDir(string* dir) {
  const char* outputs_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputs_dir == nullptr) {
    return false;
  }
  if (dir != nullptr) {
    *dir = outputs_dir;
  }
  return true;
}

bool ResolveTestPrefixes(absl::string_view path, string& resolved_path) {
  constexpr absl::string_view kTestWorkspaceSegment = "TEST_WORKSPACE";
  constexpr absl::string_view kOutputDirSegment = "TEST_UNDECLARED_OUTPUTS_DIR";

  if (StartsWithSegment(path, kTestWorkspaceSegment)) {
    if (!GetTestWorkspaceDir(&resolved_path)) {
      return false;
    }
    resolved_path += path.substr(kTestWorkspaceSegment.size());
    return true;
  } else if (StartsWithSegment(path, kOutputDirSegment)) {
    if (!GetTestUndeclaredOutputsDir(&resolved_path)) {
      return false;
    }
    resolved_path += path.substr(kOutputDirSegment.size());
    return true;
  } else {
    resolved_path = path;
    return true;
  }
}

[[maybe_unused]] std::string& AppendDotExeIfWindows(std::string& path) {
#ifdef PLATFORM_WINDOWS
  path.append(".exe");
#endif  // PLATFORM_WINDOWS
  return path;
}

}  // namespace io
}  // namespace tsl
