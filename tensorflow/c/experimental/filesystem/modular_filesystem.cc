/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/filesystem/modular_filesystem.h"

// TODO(mihaimaruseac): After all filesystems are converted, all calls to
// methods from `FileSystem` will have to be replaced to calls to private
// methods here, as part of making this class a singleton and the only way to
// register/use filesystems.

namespace tensorflow {

Status ModularFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::NewWritableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::FileExists(const std::string& fname) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

bool ModularFileSystem::FilesExist(const std::vector<std::string>& files,
                                   std::vector<Status>* status) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return true;
}

Status ModularFileSystem::GetChildren(const std::string& dir,
                                      std::vector<std::string>* result) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::GetMatchingPaths(const std::string& pattern,
                                           std::vector<std::string>* results) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::DeleteFile(const std::string& fname) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::DeleteRecursively(const std::string& dirname,
                                            int64* undeleted_files,
                                            int64* undeleted_dirs) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::DeleteDir(const std::string& dirname) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::RecursivelyCreateDir(const std::string& dirname) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::CreateDir(const std::string& dirname) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::Stat(const std::string& fname, FileStatistics* stat) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::IsDirectory(const std::string& name) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::GetFileSize(const std::string& fname,
                                      uint64* file_size) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::RenameFile(const std::string& src,
                                     const std::string& target) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::CopyFile(const std::string& src,
                                   const std::string& target) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

std::string ModularFileSystem::TranslateName(const std::string& name) const {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return "Modular filesystem stub not implemented yet";
}

void ModularFileSystem::FlushCaches() {
  // TODO(mihaimaruseac): Implementation to come in a new change
}

Status ModularRandomAccessFile::Read(uint64 offset, size_t n,
                                     StringPiece* result, char* scratch) const {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularRandomAccessFile::Name(StringPiece* result) const {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularWritableFile::Append(StringPiece data) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularWritableFile::Close() {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularWritableFile::Flush() {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularWritableFile::Sync() {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularWritableFile::Name(StringPiece* result) const {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularWritableFile::Tell(int64* position) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

}  // namespace tensorflow
