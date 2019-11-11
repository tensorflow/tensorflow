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

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/util/ptr_util.h"

// TODO(mihaimaruseac): After all filesystems are converted, all calls to
// methods from `FileSystem` will have to be replaced to calls to private
// methods here, as part of making this class a singleton and the only way to
// register/use filesystems.

namespace tensorflow {

using UniquePtrTo_TF_Status =
    ::std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

Status ModularFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  // TODO(mihaimaruseac): Implementation to come in a new change
  return Status(error::UNIMPLEMENTED,
                "Modular filesystem stub not implemented yet");
}

Status ModularFileSystem::NewWritableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  if (ops_->new_writable_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewWritableFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = tensorflow::MakeUnique<TF_WritableFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_writable_file(filesystem_.get(), translated_name.c_str(),
                          file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = tensorflow::MakeUnique<ModularWritableFile>(
        translated_name, std::move(file), writable_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
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
  if (ops_->create_dir == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dirname, " does not support CreateDir()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->create_dir(filesystem_.get(), translated_name.c_str(),
                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
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
  if (ops_->translate_name == nullptr) return FileSystem::TranslateName(name);

  char* p = ops_->translate_name(filesystem_.get(), name.c_str());
  CHECK(p != nullptr) << "TranslateName(" << name << ") returned nullptr";

  std::string ret(p);
  free(p);
  return ret;
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
