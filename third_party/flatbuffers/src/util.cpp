/*
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
// Dont't remove `format off`, it prevent reordering of win-includes.
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifdef _MSC_VER
#    include <crtdbg.h>
#  endif
#  include <windows.h>  // Must be included before <direct.h>
#  include <direct.h>
#  include <winbase.h>
#  undef interface  // This is also important because of reasons
#else
#  include <limits.h>
#endif
// clang-format on

#include "flatbuffers/base.h"
#include "flatbuffers/util.h"

#include <sys/stat.h>
#include <clocale>
#include <fstream>

namespace flatbuffers {

bool FileExistsRaw(const char *name) {
  std::ifstream ifs(name);
  return ifs.good();
}

bool LoadFileRaw(const char *name, bool binary, std::string *buf) {
  if (DirExists(name)) return false;
  std::ifstream ifs(name, binary ? std::ifstream::binary : std::ifstream::in);
  if (!ifs.is_open()) return false;
  if (binary) {
    // The fastest way to read a file into a string.
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg();
    (*buf).resize(static_cast<size_t>(size));
    ifs.seekg(0, std::ios::beg);
    ifs.read(&(*buf)[0], (*buf).size());
  } else {
    // This is slower, but works correctly on all platforms for text files.
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *buf = oss.str();
  }
  return !ifs.bad();
}

static LoadFileFunction g_load_file_function = LoadFileRaw;
static FileExistsFunction g_file_exists_function = FileExistsRaw;

bool LoadFile(const char *name, bool binary, std::string *buf) {
  FLATBUFFERS_ASSERT(g_load_file_function);
  return g_load_file_function(name, binary, buf);
}

bool FileExists(const char *name) {
  FLATBUFFERS_ASSERT(g_file_exists_function);
  return g_file_exists_function(name);
}

bool DirExists(const char *name) {
  // clang-format off

  #ifdef _WIN32
    #define flatbuffers_stat _stat
    #define FLATBUFFERS_S_IFDIR _S_IFDIR
  #else
    #define flatbuffers_stat stat
    #define FLATBUFFERS_S_IFDIR S_IFDIR
  #endif
  // clang-format on
  struct flatbuffers_stat file_info;
  if (flatbuffers_stat(name, &file_info) != 0) return false;
  return (file_info.st_mode & FLATBUFFERS_S_IFDIR) != 0;
}

LoadFileFunction SetLoadFileFunction(LoadFileFunction load_file_function) {
  LoadFileFunction previous_function = g_load_file_function;
  g_load_file_function = load_file_function ? load_file_function : LoadFileRaw;
  return previous_function;
}

FileExistsFunction SetFileExistsFunction(
    FileExistsFunction file_exists_function) {
  FileExistsFunction previous_function = g_file_exists_function;
  g_file_exists_function =
      file_exists_function ? file_exists_function : FileExistsRaw;
  return previous_function;
}

bool SaveFile(const char *name, const char *buf, size_t len, bool binary) {
  std::ofstream ofs(name, binary ? std::ofstream::binary : std::ofstream::out);
  if (!ofs.is_open()) return false;
  ofs.write(buf, len);
  return !ofs.bad();
}

// We internally store paths in posix format ('/'). Paths supplied
// by the user should go through PosixPath to ensure correct behavior
// on Windows when paths are string-compared.

static const char kPathSeparatorWindows = '\\';
static const char *PathSeparatorSet = "\\/";  // Intentionally no ':'

std::string StripExtension(const std::string &filepath) {
  size_t i = filepath.find_last_of('.');
  return i != std::string::npos ? filepath.substr(0, i) : filepath;
}

std::string GetExtension(const std::string &filepath) {
  size_t i = filepath.find_last_of('.');
  return i != std::string::npos ? filepath.substr(i + 1) : "";
}

std::string StripPath(const std::string &filepath) {
  size_t i = filepath.find_last_of(PathSeparatorSet);
  return i != std::string::npos ? filepath.substr(i + 1) : filepath;
}

std::string StripFileName(const std::string &filepath) {
  size_t i = filepath.find_last_of(PathSeparatorSet);
  return i != std::string::npos ? filepath.substr(0, i) : "";
}

std::string ConCatPathFileName(const std::string &path,
                               const std::string &filename) {
  std::string filepath = path;
  if (filepath.length()) {
    char &filepath_last_character = string_back(filepath);
    if (filepath_last_character == kPathSeparatorWindows) {
      filepath_last_character = kPathSeparator;
    } else if (filepath_last_character != kPathSeparator) {
      filepath += kPathSeparator;
    }
  }
  filepath += filename;
  // Ignore './' at the start of filepath.
  if (filepath[0] == '.' && filepath[1] == kPathSeparator) {
    filepath.erase(0, 2);
  }
  return filepath;
}

std::string PosixPath(const char *path) {
  std::string p = path;
  std::replace(p.begin(), p.end(), '\\', '/');
  return p;
}

void EnsureDirExists(const std::string &filepath) {
  auto parent = StripFileName(filepath);
  if (parent.length()) EnsureDirExists(parent);
    // clang-format off

  #ifdef _WIN32
    (void)_mkdir(filepath.c_str());
  #else
    mkdir(filepath.c_str(), S_IRWXU|S_IRGRP|S_IXGRP);
  #endif
  // clang-format on
}

std::string AbsolutePath(const std::string &filepath) {
  // clang-format off

  #ifdef FLATBUFFERS_NO_ABSOLUTE_PATH_RESOLUTION
    return filepath;
  #else
    #ifdef _WIN32
      char abs_path[MAX_PATH];
      return GetFullPathNameA(filepath.c_str(), MAX_PATH, abs_path, nullptr)
    #else
      char abs_path[PATH_MAX];
      return realpath(filepath.c_str(), abs_path)
    #endif
      ? abs_path
      : filepath;
  #endif // FLATBUFFERS_NO_ABSOLUTE_PATH_RESOLUTION
  // clang-format on
}

// Locale-independent code.
#if defined(FLATBUFFERS_LOCALE_INDEPENDENT) && \
    (FLATBUFFERS_LOCALE_INDEPENDENT > 0)

// clang-format off
// Allocate locale instance at startup of application.
ClassicLocale ClassicLocale::instance_;

#ifdef _MSC_VER
  ClassicLocale::ClassicLocale()
    : locale_(_create_locale(LC_ALL, "C")) {}
  ClassicLocale::~ClassicLocale() { _free_locale(locale_); }
#else
  ClassicLocale::ClassicLocale()
    : locale_(newlocale(LC_ALL, "C", nullptr)) {}
  ClassicLocale::~ClassicLocale() { freelocale(locale_); }
#endif
// clang-format on

#endif  // !FLATBUFFERS_LOCALE_INDEPENDENT

std::string RemoveStringQuotes(const std::string &s) {
  auto ch = *s.c_str();
  return ((s.size() >= 2) && (ch == '\"' || ch == '\'') &&
          (ch == string_back(s)))
             ? s.substr(1, s.length() - 2)
             : s;
}

bool SetGlobalTestLocale(const char *locale_name, std::string *_value) {
  const auto the_locale = setlocale(LC_ALL, locale_name);
  if (!the_locale) return false;
  if (_value) *_value = std::string(the_locale);
  return true;
}

bool ReadEnvironmentVariable(const char *var_name, std::string *_value) {
  #ifdef _MSC_VER
  __pragma(warning(disable : 4996)); // _CRT_SECURE_NO_WARNINGS
  #endif
  auto env_str = std::getenv(var_name);
  if (!env_str) return false;
  if (_value) *_value = std::string(env_str);
  return true;
}

void SetupDefaultCRTReportMode() {
  // clang-format off

  #ifdef _MSC_VER
    // By default, send all reports to STDOUT to prevent CI hangs.
    // Enable assert report box [Abort|Retry|Ignore] if a debugger is present.
    const int dbg_mode = (_CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG) |
                         (IsDebuggerPresent() ? _CRTDBG_MODE_WNDW : 0);
    (void)dbg_mode; // release mode fix
    // CrtDebug reports to _CRT_WARN channel.
    _CrtSetReportMode(_CRT_WARN, dbg_mode);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    // The assert from <assert.h> reports to _CRT_ERROR channel
    _CrtSetReportMode(_CRT_ERROR, dbg_mode);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    // Internal CRT assert channel?
    _CrtSetReportMode(_CRT_ASSERT, dbg_mode);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
  #endif

  // clang-format on
}

}  // namespace flatbuffers
