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

#include <deque>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

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
  ParseURI(fname, &scheme, &host, &path);
  FileSystem* file_system = file_system_registry_->Lookup(scheme.ToString());
  if (!file_system) {
    return errors::Unimplemented("File system scheme ", scheme,
                                 " not implemented");
  }
  *result = file_system;
  return Status::OK();
}

Status Env::GetRegisteredFileSystemSchemes(std::vector<string>* schemes) {
  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

Status Env::RegisterFileSystem(const string& scheme,
                               FileSystemRegistry::Factory factory) {
  return file_system_registry_->Register(scheme, factory);
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

bool Env::FileExists(const string& fname) {
  FileSystem* fs;
  if (!GetFileSystemForFile(fname, &fs).ok()) {
    return false;
  }
  return fs->FileExists(fname);
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
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  const string& fixed_prefix =
      pattern.substr(0, pattern.find_first_of("*?[\\"));
  const string& base_dir = io::Dirname(fixed_prefix).ToString();
  const string& list_dir = base_dir.empty() ? "." : base_dir;

  std::vector<string> all_files;
  TF_RETURN_IF_ERROR(fs->GetChildrenRecursively(list_dir, &all_files));

  // Match all obtained files to the input pattern.
  for (const auto& f : all_files) {
    const string& full_path = io::JoinPath(base_dir, f);
    if (MatchPath(full_path, pattern)) {
      results->push_back(full_path);
    }
  }
  return Status::OK();
}

Status Env::DeleteFile(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->DeleteFile(fname);
}

Status Env::RecursivelyCreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  StringPiece scheme, host, remaining_dir;
  ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!fs->FileExists(CreateURI(scheme, host, remaining_dir)) &&
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
    TF_RETURN_IF_ERROR(fs->CreateDir(CreateURI(scheme, host, built_path)));
  }
  return Status::OK();
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
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));

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
    Status s = fs->GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const string& child : children) {
      const string child_path = io::JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (fs->IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = fs->DeleteFile(child_path);
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
    Status s = fs->DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return Status::OK();
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
  auto s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
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
    s = stream->status();
    if (s.ok()) {
      s = Status(error::DATA_LOSS, "Parse error");
    }
  }
  return s;
}

}  // namespace tensorflow
