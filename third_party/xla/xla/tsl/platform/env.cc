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

#include "xla/tsl/platform/env.h"

#include <sys/stat.h>

#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/stringprintf.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(__FreeBSD__)
#include <sys/sysctl.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#undef DeleteFile
#undef CopyFile
#include "xla/tsl/platform/windows/wide_char.h"
#define PATH_MAX MAX_PATH
#else
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace tsl {

// 128KB copy buffer
constexpr size_t kCopyFileBufferSize = 128 * 1024;

class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  absl::Status Register(const std::string& scheme, Factory factory) override;
  absl::Status Register(const std::string& scheme,
                        std::unique_ptr<FileSystem> filesystem) override;
  FileSystem* Lookup(const std::string& scheme) override;
  absl::Status GetRegisteredFileSystemSchemes(
      std::vector<std::string>* schemes) override;

 private:
  mutable mutex mu_;
  mutable std::unordered_map<std::string, std::unique_ptr<FileSystem>> registry_
      TF_GUARDED_BY(mu_);
};

absl::Status FileSystemRegistryImpl::Register(
    const std::string& scheme, FileSystemRegistry::Factory factory) {
  mutex_lock lock(mu_);
  if (!registry_.emplace(scheme, std::unique_ptr<FileSystem>(factory()))
           .second) {
    return errors::AlreadyExists("File factory for ", scheme,
                                 " already registered");
  }
  return absl::OkStatus();
}

absl::Status FileSystemRegistryImpl::Register(
    const std::string& scheme, std::unique_ptr<FileSystem> filesystem) {
  mutex_lock lock(mu_);
  if (!registry_.emplace(scheme, std::move(filesystem)).second) {
    return errors::AlreadyExists("File system for ", scheme,
                                 " already registered");
  }
  return absl::OkStatus();
}

FileSystem* FileSystemRegistryImpl::Lookup(const std::string& scheme) {
  mutex_lock lock(mu_);
  const auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

absl::Status FileSystemRegistryImpl::GetRegisteredFileSystemSchemes(
    std::vector<std::string>* schemes) {
  mutex_lock lock(mu_);
  for (const auto& e : registry_) {
    schemes->push_back(e.first);
  }
  return absl::OkStatus();
}

Env::Env() : file_system_registry_(new FileSystemRegistryImpl) {}

absl::Status Env::GetFileSystemForFile(const std::string& fname,
                                       FileSystem** result) {
  absl::string_view scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  FileSystem* file_system = file_system_registry_->Lookup(std::string(scheme));
  if (!file_system) {
    if (scheme.empty()) {
      scheme = "[local]";
    }

    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not implemented (file: '", fname, "')");
  }
  *result = file_system;
  return absl::OkStatus();
}

absl::Status Env::GetRegisteredFileSystemSchemes(
    std::vector<std::string>* schemes) {
  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

absl::Status Env::RegisterFileSystem(const std::string& scheme,
                                     FileSystemRegistry::Factory factory) {
  return file_system_registry_->Register(scheme, std::move(factory));
}

absl::Status Env::RegisterFileSystem(const std::string& scheme,
                                     std::unique_ptr<FileSystem> filesystem) {
  return file_system_registry_->Register(scheme, std::move(filesystem));
}

absl::Status Env::SetOption(const std::string& scheme, const std::string& key,
                            const std::string& value) {
  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, value);
}

absl::Status Env::SetOption(const std::string& scheme, const std::string& key,
                            const std::vector<string>& values) {
  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, values);
}

absl::Status Env::SetOption(const std::string& scheme, const std::string& key,
                            const std::vector<int64_t>& values) {
  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, values);
}

absl::Status Env::SetOption(const std::string& scheme, const std::string& key,
                            const std::vector<double>& values) {
  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, values);
}

absl::Status Env::FlushFileSystemCaches() {
  std::vector<string> schemes;
  TF_RETURN_IF_ERROR(GetRegisteredFileSystemSchemes(&schemes));
  for (const string& scheme : schemes) {
    FileSystem* fs = nullptr;
    TF_RETURN_IF_ERROR(
        GetFileSystemForFile(io::CreateURI(scheme, "", ""), &fs));
    fs->FlushCaches();
  }
  return absl::OkStatus();
}

absl::Status Env::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewRandomAccessFile(fname, result);
}

absl::Status Env::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewReadOnlyMemoryRegionFromFile(fname, result);
}

absl::Status Env::NewWritableFile(const string& fname,
                                  std::unique_ptr<WritableFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewWritableFile(fname, result);
}

absl::Status Env::NewAppendableFile(const string& fname,
                                    std::unique_ptr<WritableFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewAppendableFile(fname, result);
}

absl::Status Env::FileExists(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->FileExists(fname);
}

bool Env::FilesExist(const std::vector<string>& files,
                     std::vector<absl::Status>* status) {
  std::unordered_map<string, std::vector<string>> files_per_fs;
  for (const auto& file : files) {
    absl::string_view scheme, host, path;
    io::ParseURI(file, &scheme, &host, &path);
    files_per_fs[string(scheme)].push_back(file);
  }

  std::unordered_map<string, absl::Status> per_file_status;
  bool result = true;
  for (auto itr : files_per_fs) {
    FileSystem* file_system = file_system_registry_->Lookup(itr.first);
    bool fs_result;
    std::vector<absl::Status> local_status;
    std::vector<absl::Status>* fs_status = status ? &local_status : nullptr;
    if (!file_system) {
      fs_result = false;
      if (fs_status) {
        absl::Status s = errors::Unimplemented("File system scheme '",
                                               itr.first, "' not implemented");
        local_status.resize(itr.second.size(), s);
      }
    } else {
      fs_result = file_system->FilesExist(itr.second, fs_status);
    }
    if (fs_status) {
      result &= fs_result;
      for (size_t i = 0; i < itr.second.size(); ++i) {
        per_file_status[itr.second[i]] = fs_status->at(i);
      }
    } else if (!fs_result) {
      // Return early
      return false;
    }
  }

  if (status) {
    for (const auto& file : files) {
      status->push_back(per_file_status[file]);
    }
  }

  return result;
}

absl::Status Env::GetChildren(const string& dir, std::vector<string>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dir, &fs));
  return fs->GetChildren(dir, result);
}

absl::Status Env::GetMatchingPaths(const string& pattern,
                                   std::vector<string>* results) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(pattern, &fs));
  return fs->GetMatchingPaths(pattern, results);
}

absl::Status Env::DeleteFile(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->DeleteFile(fname);
}

absl::Status Env::RecursivelyCreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->RecursivelyCreateDir(dirname);
}

absl::Status Env::CreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->CreateDir(dirname);
}

absl::Status Env::DeleteDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteDir(dirname);
}

absl::Status Env::Stat(const string& fname, FileStatistics* stat) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->Stat(fname, stat);
}

absl::Status Env::IsDirectory(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->IsDirectory(fname);
}

absl::Status Env::HasAtomicMove(const string& path, bool* has_atomic_move) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(path, &fs));
  return fs->HasAtomicMove(path, has_atomic_move);
}

absl::Status Env::CanCreateTempFile(const string& fname,
                                    bool* can_create_temp_file) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->CanCreateTempFile(fname, can_create_temp_file);
}

absl::Status Env::DeleteRecursively(const string& dirname,
                                    int64_t* undeleted_files,
                                    int64_t* undeleted_dirs) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteRecursively(dirname, undeleted_files, undeleted_dirs);
}

absl::Status Env::GetFileSize(const string& fname, uint64* file_size) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->GetFileSize(fname, file_size);
}

absl::Status Env::RenameFile(const string& src, const string& target) {
  FileSystem* src_fs;
  FileSystem* target_fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  TF_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs != target_fs) {
    return errors::Unimplemented("Renaming ", src, " to ", target,
                                 " not implemented");
  }
  return src_fs->RenameFile(src, target);
}

absl::Status Env::CopyFile(const string& src, const string& target) {
  FileSystem* src_fs;
  FileSystem* target_fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  TF_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs == target_fs) {
    return src_fs->CopyFile(src, target);
  }
  return FileSystemCopyFile(src_fs, src, target_fs, target);
}

string Env::GetExecutablePath() {
  char exe_path[PATH_MAX] = {0};
#ifdef __APPLE__
  uint32_t buffer_size(0U);
  _NSGetExecutablePath(nullptr, &buffer_size);
  std::vector<char> unresolved_path(buffer_size);
  _NSGetExecutablePath(unresolved_path.data(), &buffer_size);
  CHECK(realpath(unresolved_path.data(), exe_path));
#elif defined(__FreeBSD__)
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
  size_t exe_path_size = PATH_MAX;

  if (sysctl(mib, 4, exe_path, &exe_path_size, NULL, 0) != 0) {
    // Resolution of path failed
    return "";
  }
#elif defined(PLATFORM_WINDOWS)
  HMODULE hModule = GetModuleHandleW(NULL);
  WCHAR wc_file_path[MAX_PATH] = {0};
  GetModuleFileNameW(hModule, wc_file_path, MAX_PATH);
  string file_path = WideCharToUtf8(wc_file_path);
  std::copy(file_path.begin(), file_path.end(), exe_path);
#else
  char buf[PATH_MAX] = {0};
  int path_length = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  CHECK_NE(-1, path_length);

  if (strstr(buf, "python") != nullptr) {
    // Discard the path of the python binary, and any flags.
    int fd = open("/proc/self/cmdline", O_RDONLY);
    CHECK_NE(-1, fd);
    int cmd_length = read(fd, buf, PATH_MAX - 1);
    CHECK_NE(-1, cmd_length);
    close(fd);
    int token_pos = 0;
    for (bool token_is_first_or_flag = true; token_is_first_or_flag;) {
      // Get token length, including null
      int token_len = strlen(&buf[token_pos]) + 1;
      token_is_first_or_flag = false;
      // Check if we can skip without overshooting
      if (token_pos + token_len < cmd_length) {
        token_pos += token_len;
        token_is_first_or_flag = (buf[token_pos] == '-');  // token is a flag
      }
    }
    snprintf(exe_path, sizeof(exe_path), "%s", &buf[token_pos]);
  } else {
    snprintf(exe_path, sizeof(exe_path), "%s", buf);
  }

#endif
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  return exe_path;
}

bool Env::LocalTempFilename(string* filename) {
  std::vector<string> dirs;
  GetLocalTempDirectories(&dirs);

  // Try each directory, as they might be full, have inappropriate
  // permissions or have different problems at times.
  for (const string& dir : dirs) {
    *filename = io::JoinPath(dir, "tempfile-");
    if (CreateUniqueFileName(filename, "")) {
      return true;
    }
  }
  return false;
}

bool Env::CreateUniqueFileName(string* prefix, const string& suffix) {
  int64_t tid = GetCurrentThreadId();
  int32_t pid = GetProcessId();
  long long now_microsec = NowMicros();  // NOLINT

  absl::StrAppendFormat(prefix, "%s-%x-%d-%llx", port::Hostname(), tid, pid,
                        now_microsec);

  if (!suffix.empty()) {
    *prefix += suffix;
  }
  if (FileExists(*prefix).ok()) {
    prefix->clear();
    return false;
  } else {
    return true;
  }
}

int32 Env::GetProcessId() {
#ifdef PLATFORM_WINDOWS
  return static_cast<int32>(GetCurrentProcessId());
#else
  return static_cast<int32>(getpid());
#endif
}

Thread::~Thread() {}

EnvWrapper::~EnvWrapper() {}

absl::Status ReadFileToString(Env* env, const string& fname, string* data) {
  uint64 file_size;
  absl::Status s = env->GetFileSize(fname, &file_size);
  if (!s.ok()) {
    return s;
  }
  std::unique_ptr<RandomAccessFile> file;
  s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  data->resize(file_size);
  char* p = &*data->begin();
  absl::string_view result;
  s = file->Read(0, file_size, &result, p);
  if (!s.ok()) {
    data->clear();
  } else if (result.size() != file_size) {
    s = errors::Aborted("File ", fname, " changed while reading: ", file_size,
                        " vs. ", result.size());
    data->clear();
  } else if (result.data() == p) {
    // Data is already in the correct location
  } else {
    memmove(p, result.data(), result.size());
  }
  return s;
}

absl::Status WriteStringToFile(Env* env, const string& fname,
                               const absl::string_view& data) {
  std::unique_ptr<WritableFile> file;
  absl::Status s = env->NewWritableFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok()) {
    s = file->Close();
  }
  return s;
}

absl::Status FileSystemCopyFile(FileSystem* src_fs, const string& src,
                                FileSystem* target_fs, const string& target) {
  std::unique_ptr<RandomAccessFile> src_file;
  TF_RETURN_IF_ERROR(src_fs->NewRandomAccessFile(src, &src_file));

  // When `target` points to a directory, we need to create a file within.
  string target_name;
  if (target_fs->IsDirectory(target).ok()) {
    target_name = io::JoinPath(target, io::Basename(src));
  } else {
    target_name = target;
  }

  std::unique_ptr<WritableFile> target_file;
  TF_RETURN_IF_ERROR(target_fs->NewWritableFile(target_name, &target_file));

  uint64 offset = 0;
  std::unique_ptr<char[]> scratch(new char[kCopyFileBufferSize]);
  absl::Status s = absl::OkStatus();
  while (s.ok()) {
    absl::string_view result;
    s = src_file->Read(offset, kCopyFileBufferSize, &result, scratch.get());
    if (!(s.ok() || s.code() == error::OUT_OF_RANGE)) {
      return s;
    }
    TF_RETURN_IF_ERROR(target_file->Append(result));
    offset += result.size();
  }
  return target_file->Close();
}

// A ZeroCopyInputStream on a RandomAccessFile.
namespace {
class FileStream : public protobuf::io::ZeroCopyInputStream {
 public:
  explicit FileStream(RandomAccessFile* file) : file_(file), pos_(0) {}

  void BackUp(int count) override { pos_ -= count; }
  bool Skip(int count) override {
    pos_ += count;
    return true;
  }
  int64_t ByteCount() const override { return pos_; }
  absl::Status status() const { return status_; }

  bool Next(const void** data, int* size) override {
    absl::string_view result;
    absl::Status s = file_->Read(pos_, kBufSize, &result, scratch_);
    if (result.empty()) {
      status_ = s;
      return false;
    }
    pos_ += result.size();
    *data = result.data();
    *size = result.size();
    return true;
  }

 private:
  static constexpr int kBufSize = 512 << 10;

  RandomAccessFile* file_;
  int64_t pos_;
  absl::Status status_;
  char scratch_[kBufSize];
};

}  // namespace

absl::Status WriteBinaryProto(Env* env, const string& fname,
                              const protobuf::MessageLite& proto) {
  string serialized;
  proto.AppendToString(&serialized);
  return WriteStringToFile(env, fname, serialized);
}

absl::Status ReadBinaryProto(Env* env, const string& fname,
                             protobuf::MessageLite* proto) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));
  protobuf::io::CodedInputStream coded_stream(stream.get());

  if (!proto->ParseFromCodedStream(&coded_stream) ||
      !coded_stream.ConsumedEntireMessage()) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as binary proto");
  }
  return absl::OkStatus();
}

absl::Status WriteTextProto(Env* env, const string& fname,
                            const protobuf::Message& proto) {
  string serialized;
  if (!protobuf::TextFormat::PrintToString(proto, &serialized)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  return WriteStringToFile(env, fname, serialized);
}

absl::Status ReadTextProto(Env* env, const string& fname,
                           protobuf::Message* proto) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  if (!protobuf::TextFormat::Parse(stream.get(), proto)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as text proto");
  }
  return absl::OkStatus();
}

absl::Status ReadTextOrBinaryProto(Env* env, const string& fname,
                                   protobuf::Message* proto) {
  if (ReadTextProto(env, fname, proto).ok()) {
    return absl::OkStatus();
  }
  return ReadBinaryProto(env, fname, proto);
}

absl::Status ReadTextOrBinaryProto(Env* env, const string& fname,
                                   protobuf::MessageLite* proto) {
  return ReadBinaryProto(env, fname, proto);
}

}  // namespace tsl
