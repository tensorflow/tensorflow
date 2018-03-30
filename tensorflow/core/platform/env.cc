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
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(__FreeBSD__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#include "tensorflow/core/platform/windows/windows_file_system.h"
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#endif

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// 128KB copy buffer
constexpr size_t kCopyFileBufferSize = 128 * 1024;

class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  Status Register(const string& scheme, Factory factory) override;
  FileSystem* Lookup(const string& scheme) override;
  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override;

 private:
  mutable mutex mu_;
  mutable std::unordered_map<string, std::unique_ptr<FileSystem>> registry_
      GUARDED_BY(mu_);
};

Status FileSystemRegistryImpl::Register(const string& scheme,
                                        FileSystemRegistry::Factory factory) {
  mutex_lock lock(mu_);
  if (!registry_.emplace(string(scheme), std::unique_ptr<FileSystem>(factory()))
           .second) {
    return errors::AlreadyExists("File factory for ", scheme,
                                 " already registered");
  }
  return Status::OK();
}

FileSystem* FileSystemRegistryImpl::Lookup(const string& scheme) {
  mutex_lock lock(mu_);
  const auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

Status FileSystemRegistryImpl::GetRegisteredFileSystemSchemes(
    std::vector<string>* schemes) {
  mutex_lock lock(mu_);
  for (const auto& e : registry_) {
    schemes->push_back(e.first);
  }
  return Status::OK();
}

Env::Env() : file_system_registry_(new FileSystemRegistryImpl) {}

Status Env::GetFileSystemForFile(const string& fname, FileSystem** result) {
  StringPiece scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  FileSystem* file_system = file_system_registry_->Lookup(scheme.ToString());
  if (!file_system) {
    if (scheme.empty()) {
      scheme = "[local]";
    }

    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not implemented (file: '", fname, "')");
  }
  *result = file_system;
  return Status::OK();
}

Status Env::GetRegisteredFileSystemSchemes(std::vector<string>* schemes) {
  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

Status Env::RegisterFileSystem(const string& scheme,
                               FileSystemRegistry::Factory factory) {
  return file_system_registry_->Register(scheme, std::move(factory));
}

Status Env::FlushFileSystemCaches() {
  std::vector<string> schemes;
  TF_RETURN_IF_ERROR(GetRegisteredFileSystemSchemes(&schemes));
  for (const string& scheme : schemes) {
    FileSystem* fs = nullptr;
    TF_RETURN_IF_ERROR(
        GetFileSystemForFile(io::CreateURI(scheme, "", ""), &fs));
    fs->FlushCaches();
  }
  return Status::OK();
}

Status Env::NewRandomAccessFile(const string& fname,
                                std::unique_ptr<RandomAccessFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewRandomAccessFile(fname, result);
}

Status Env::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewReadOnlyMemoryRegionFromFile(fname, result);
}

Status Env::NewWritableFile(const string& fname,
                            std::unique_ptr<WritableFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewWritableFile(fname, result);
}

Status Env::NewAppendableFile(const string& fname,
                              std::unique_ptr<WritableFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewAppendableFile(fname, result);
}

Status Env::FileExists(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->FileExists(fname);
}

bool Env::FilesExist(const std::vector<string>& files,
                     std::vector<Status>* status) {
  std::unordered_map<string, std::vector<string>> files_per_fs;
  for (const auto& file : files) {
    StringPiece scheme, host, path;
    io::ParseURI(file, &scheme, &host, &path);
    files_per_fs[scheme.ToString()].push_back(file);
  }

  std::unordered_map<string, Status> per_file_status;
  bool result = true;
  for (auto itr : files_per_fs) {
    FileSystem* file_system = file_system_registry_->Lookup(itr.first);
    bool fs_result;
    std::vector<Status> local_status;
    std::vector<Status>* fs_status = status ? &local_status : nullptr;
    if (!file_system) {
      fs_result = false;
      if (fs_status) {
        Status s = errors::Unimplemented("File system scheme '", itr.first,
                                         "' not implemented");
        local_status.resize(itr.second.size(), s);
      }
    } else {
      fs_result = file_system->FilesExist(itr.second, fs_status);
    }
    if (fs_status) {
      result &= fs_result;
      for (int i = 0; i < itr.second.size(); ++i) {
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

Status Env::GetChildren(const string& dir, std::vector<string>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dir, &fs));
  return fs->GetChildren(dir, result);
}

Status Env::GetMatchingPaths(const string& pattern,
                             std::vector<string>* results) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(pattern, &fs));
  return fs->GetMatchingPaths(pattern, results);
}

Status Env::DeleteFile(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->DeleteFile(fname);
}

Status Env::RecursivelyCreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->RecursivelyCreateDir(dirname);
}

Status Env::CreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->CreateDir(dirname);
}

Status Env::DeleteDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteDir(dirname);
}

Status Env::Stat(const string& fname, FileStatistics* stat) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->Stat(fname, stat);
}

Status Env::IsDirectory(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->IsDirectory(fname);
}

Status Env::DeleteRecursively(const string& dirname, int64* undeleted_files,
                              int64* undeleted_dirs) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteRecursively(dirname, undeleted_files, undeleted_dirs);
}

Status Env::GetFileSize(const string& fname, uint64* file_size) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->GetFileSize(fname, file_size);
}

Status Env::RenameFile(const string& src, const string& target) {
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

Status Env::CopyFile(const string& src, const string& target) {
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
  char unresolved_path[buffer_size];
  _NSGetExecutablePath(unresolved_path, &buffer_size);
  CHECK(realpath(unresolved_path, exe_path));
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
  string file_path = WindowsFileSystem::WideCharToUtf8(wc_file_path);
  std::copy(file_path.begin(), file_path.end(), exe_path);
#else
  CHECK_NE(-1, readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
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
#ifdef __APPLE__
  uint64_t tid64;
  pthread_threadid_np(nullptr, &tid64);
  int32 tid = static_cast<int32>(tid64);
  int32 pid = static_cast<int32>(getpid());
#elif defined(__FreeBSD__)
  // Has to be casted to long first, else this error appears:
  // static_cast from 'pthread_t' (aka 'pthread *') to 'int32' (aka 'int')
  // is not allowed
  int32 tid = static_cast<int32>(static_cast<int64>(pthread_self()));
  int32 pid = static_cast<int32>(getpid());
#elif defined(PLATFORM_WINDOWS)
  int32 tid = static_cast<int32>(GetCurrentThreadId());
  int32 pid = static_cast<int32>(GetCurrentProcessId());
#else
  int32 tid = static_cast<int32>(pthread_self());
  int32 pid = static_cast<int32>(getpid());
#endif
  uint64 now_microsec = NowMicros();

  *prefix += strings::Printf("%s-%x-%d-%llx", port::Hostname().c_str(), tid,
                             pid, now_microsec);

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

Thread::~Thread() {}

EnvWrapper::~EnvWrapper() {}

Status ReadFileToString(Env* env, const string& fname, string* data) {
  uint64 file_size;
  Status s = env->GetFileSize(fname, &file_size);
  if (!s.ok()) {
    return s;
  }
  std::unique_ptr<RandomAccessFile> file;
  s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  gtl::STLStringResizeUninitialized(data, file_size);
  char* p = gtl::string_as_array(data);
  StringPiece result;
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

Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data) {
  std::unique_ptr<WritableFile> file;
  Status s = env->NewWritableFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok()) {
    s = file->Close();
  }
  return s;
}

Status FileSystemCopyFile(FileSystem* src_fs, const string& src,
                          FileSystem* target_fs, const string& target) {
  std::unique_ptr<RandomAccessFile> src_file;
  TF_RETURN_IF_ERROR(src_fs->NewRandomAccessFile(src, &src_file));

  std::unique_ptr<WritableFile> target_file;
  TF_RETURN_IF_ERROR(target_fs->NewWritableFile(target, &target_file));

  uint64 offset = 0;
  std::unique_ptr<char[]> scratch(new char[kCopyFileBufferSize]);
  Status s = Status::OK();
  while (s.ok()) {
    StringPiece result;
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
class FileStream : public ::tensorflow::protobuf::io::ZeroCopyInputStream {
 public:
  explicit FileStream(RandomAccessFile* file) : file_(file), pos_(0) {}

  void BackUp(int count) override { pos_ -= count; }
  bool Skip(int count) override {
    pos_ += count;
    return true;
  }
  protobuf_int64 ByteCount() const override { return pos_; }
  Status status() const { return status_; }

  bool Next(const void** data, int* size) override {
    StringPiece result;
    Status s = file_->Read(pos_, kBufSize, &result, scratch_);
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
  static const int kBufSize = 512 << 10;

  RandomAccessFile* file_;
  int64 pos_;
  Status status_;
  char scratch_[kBufSize];
};

}  // namespace

Status WriteBinaryProto(Env* env, const string& fname,
                        const ::tensorflow::protobuf::MessageLite& proto) {
  string serialized;
  proto.AppendToString(&serialized);
  return WriteStringToFile(env, fname, serialized);
}

Status ReadBinaryProto(Env* env, const string& fname,
                       ::tensorflow::protobuf::MessageLite* proto) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::tensorflow::protobuf::io::CodedInputStream coded_stream(stream.get());
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);

  if (!proto->ParseFromCodedStream(&coded_stream)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as binary proto");
  }
  return Status::OK();
}

Status WriteTextProto(Env* env, const string& fname,
                      const ::tensorflow::protobuf::Message& proto) {
#if !defined(TENSORFLOW_LITE_PROTOS)
  string serialized;
  if (!::tensorflow::protobuf::TextFormat::PrintToString(proto, &serialized)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  return WriteStringToFile(env, fname, serialized);
#else
  return errors::Unimplemented("Can't write text protos with protolite.");
#endif
}

Status ReadTextProto(Env* env, const string& fname,
                     ::tensorflow::protobuf::Message* proto) {
#if !defined(TENSORFLOW_LITE_PROTOS)
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  if (!::tensorflow::protobuf::TextFormat::Parse(stream.get(), proto)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as text proto");
  }
  return Status::OK();
#else
  return errors::Unimplemented("Can't parse text protos with protolite.");
#endif
}

}  // namespace tensorflow
