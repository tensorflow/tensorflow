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

#include <algorithm>
#include <string>
#include <utility>

#include "tensorflow/c/experimental/filesystem/modular_filesystem_registration.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/tsl/platform/errors.h"

// TODO(b/139060984): After all filesystems are converted, all calls to
// methods from `FileSystem` will have to be replaced to calls to private
// methods here, as part of making this class a singleton and the only way to
// register/use filesystems.

namespace tensorflow {

using UniquePtrTo_TF_Status =
    ::std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

Status ModularFileSystem::NewRandomAccessFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  if (ops_->new_random_access_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewRandomAccessFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = MakeUnique<TF_RandomAccessFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_random_access_file(filesystem_.get(), translated_name.c_str(),
                               file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularRandomAccessFile>(
        translated_name, std::move(file), random_access_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::NewWritableFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  if (ops_->new_writable_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewWritableFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = MakeUnique<TF_WritableFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_writable_file(filesystem_.get(), translated_name.c_str(),
                          file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularWritableFile>(translated_name, std::move(file),
                                              writable_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::NewAppendableFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  if (ops_->new_appendable_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support NewAppendableFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto file = MakeUnique<TF_WritableFile>();
  std::string translated_name = TranslateName(fname);
  ops_->new_appendable_file(filesystem_.get(), translated_name.c_str(),
                            file.get(), plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularWritableFile>(translated_name, std::move(file),
                                              writable_file_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  if (ops_->new_read_only_memory_region_from_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname,
        " does not support NewReadOnlyMemoryRegionFromFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  auto region = MakeUnique<TF_ReadOnlyMemoryRegion>();
  std::string translated_name = TranslateName(fname);
  ops_->new_read_only_memory_region_from_file(
      filesystem_.get(), translated_name.c_str(), region.get(),
      plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK)
    *result = MakeUnique<ModularReadOnlyMemoryRegion>(
        std::move(region), read_only_memory_region_ops_.get());

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::FileExists(const std::string& fname,
                                     TransactionToken* token) {
  if (ops_->path_exists == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support FileExists()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  const std::string translated_name = TranslateName(fname);
  ops_->path_exists(filesystem_.get(), translated_name.c_str(),
                    plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

bool ModularFileSystem::FilesExist(const std::vector<std::string>& files,
                                   TransactionToken* token,
                                   std::vector<Status>* status) {
  if (ops_->paths_exist == nullptr)
    return FileSystem::FilesExist(files, token, status);

  std::vector<char*> translated_names;
  translated_names.reserve(files.size());
  for (int i = 0; i < files.size(); i++)
    translated_names.push_back(strdup(TranslateName(files[i]).c_str()));

  bool result;
  if (status == nullptr) {
    result = ops_->paths_exist(filesystem_.get(), translated_names.data(),
                               files.size(), nullptr);
  } else {
    std::vector<TF_Status*> plugin_status;
    plugin_status.reserve(files.size());
    for (int i = 0; i < files.size(); i++)
      plugin_status.push_back(TF_NewStatus());
    result = ops_->paths_exist(filesystem_.get(), translated_names.data(),
                               files.size(), plugin_status.data());
    for (int i = 0; i < files.size(); i++) {
      status->push_back(StatusFromTF_Status(plugin_status[i]));
      TF_DeleteStatus(plugin_status[i]);
    }
  }

  for (int i = 0; i < files.size(); i++) free(translated_names[i]);

  return result;
}

Status ModularFileSystem::GetChildren(const std::string& dir,
                                      TransactionToken* token,
                                      std::vector<std::string>* result) {
  if (ops_->get_children == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dir, " does not support GetChildren()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dir);
  // Note that `children` is allocated by the plugin and freed by core
  // TensorFlow, so we need to use `plugin_memory_free_` here.
  char** children = nullptr;
  const int num_children =
      ops_->get_children(filesystem_.get(), translated_name.c_str(), &children,
                         plugin_status.get());
  if (num_children >= 0) {
    for (int i = 0; i < num_children; i++) {
      result->push_back(std::string(children[i]));
      plugin_memory_free_(children[i]);
    }
    plugin_memory_free_(children);
  }

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::GetMatchingPaths(const std::string& pattern,
                                           TransactionToken* token,
                                           std::vector<std::string>* result) {
  if (ops_->get_matching_paths == nullptr)
    return internal::GetMatchingPaths(this, Env::Default(), pattern, result);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  // Note that `matches` is allocated by the plugin and freed by core
  // TensorFlow, so we need to use `plugin_memory_free_` here.
  char** matches = nullptr;
  const int num_matches = ops_->get_matching_paths(
      filesystem_.get(), pattern.c_str(), &matches, plugin_status.get());
  if (num_matches >= 0) {
    for (int i = 0; i < num_matches; i++) {
      result->push_back(std::string(matches[i]));
      plugin_memory_free_(matches[i]);
    }
    plugin_memory_free_(matches);
  }

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::DeleteFile(const std::string& fname,
                                     TransactionToken* token) {
  if (ops_->delete_file == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support DeleteFile()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(fname);
  ops_->delete_file(filesystem_.get(), translated_name.c_str(),
                    plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::DeleteRecursively(const std::string& dirname,
                                            TransactionToken* token,
                                            int64_t* undeleted_files,
                                            int64_t* undeleted_dirs) {
  if (undeleted_files == nullptr || undeleted_dirs == nullptr)
    return errors::FailedPrecondition(
        "DeleteRecursively must not be called with `undeleted_files` or "
        "`undeleted_dirs` set to NULL");

  if (ops_->delete_recursively == nullptr)
    return FileSystem::DeleteRecursively(dirname, token, undeleted_files,
                                         undeleted_dirs);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  uint64_t plugin_undeleted_files, plugin_undeleted_dirs;
  ops_->delete_recursively(filesystem_.get(), translated_name.c_str(),
                           &plugin_undeleted_files, &plugin_undeleted_dirs,
                           plugin_status.get());
  *undeleted_files = plugin_undeleted_files;
  *undeleted_dirs = plugin_undeleted_dirs;
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::DeleteDir(const std::string& dirname,
                                    TransactionToken* token) {
  if (ops_->delete_dir == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dirname, " does not support DeleteDir()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->delete_dir(filesystem_.get(), translated_name.c_str(),
                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::RecursivelyCreateDir(const std::string& dirname,
                                               TransactionToken* token) {
  if (ops_->recursively_create_dir == nullptr)
    return FileSystem::RecursivelyCreateDir(dirname, token);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->recursively_create_dir(filesystem_.get(), translated_name.c_str(),
                               plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::CreateDir(const std::string& dirname,
                                    TransactionToken* token) {
  if (ops_->create_dir == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", dirname, " does not support CreateDir()"));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(dirname);
  ops_->create_dir(filesystem_.get(), translated_name.c_str(),
                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::Stat(const std::string& fname,
                               TransactionToken* token, FileStatistics* stat) {
  if (ops_->stat == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Filesystem for ", fname, " does not support Stat()"));

  if (stat == nullptr)
    return errors::InvalidArgument("FileStatistics pointer must not be NULL");

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(fname);
  TF_FileStatistics stats;
  ops_->stat(filesystem_.get(), translated_name.c_str(), &stats,
             plugin_status.get());

  if (TF_GetCode(plugin_status.get()) == TF_OK) {
    stat->length = stats.length;
    stat->mtime_nsec = stats.mtime_nsec;
    stat->is_directory = stats.is_directory;
  }

  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::IsDirectory(const std::string& name,
                                      TransactionToken* token) {
  if (ops_->is_directory == nullptr)
    return FileSystem::IsDirectory(name, token);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(name);
  ops_->is_directory(filesystem_.get(), translated_name.c_str(),
                     plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::GetFileSize(const std::string& fname,
                                      TransactionToken* token,
                                      uint64* file_size) {
  if (ops_->get_file_size == nullptr) {
    FileStatistics stat;
    Status status = Stat(fname, &stat);
    if (!status.ok()) return status;
    if (stat.is_directory)
      return errors::FailedPrecondition("Called GetFileSize on a directory");

    *file_size = stat.length;
    return status;
  }

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_name = TranslateName(fname);
  *file_size = ops_->get_file_size(filesystem_.get(), translated_name.c_str(),
                                   plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::RenameFile(const std::string& src,
                                     const std::string& target,
                                     TransactionToken* token) {
  if (ops_->rename_file == nullptr) {
    Status status = CopyFile(src, target);
    if (status.ok()) status = DeleteFile(src);
    return status;
  }

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_src = TranslateName(src);
  std::string translated_target = TranslateName(target);
  ops_->rename_file(filesystem_.get(), translated_src.c_str(),
                    translated_target.c_str(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::CopyFile(const std::string& src,
                                   const std::string& target,
                                   TransactionToken* token) {
  if (ops_->copy_file == nullptr)
    return FileSystem::CopyFile(src, target, token);

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  std::string translated_src = TranslateName(src);
  std::string translated_target = TranslateName(target);
  ops_->copy_file(filesystem_.get(), translated_src.c_str(),
                  translated_target.c_str(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

std::string ModularFileSystem::TranslateName(const std::string& name) const {
  if (ops_->translate_name == nullptr) return FileSystem::TranslateName(name);

  char* p = ops_->translate_name(filesystem_.get(), name.c_str());
  CHECK(p != nullptr) << "TranslateName(" << name << ") returned nullptr";

  std::string ret(p);
  // Since `p` is allocated by plugin, free it using plugin's method.
  plugin_memory_free_(p);
  return ret;
}

void ModularFileSystem::FlushCaches(TransactionToken* token) {
  if (ops_->flush_caches != nullptr) ops_->flush_caches(filesystem_.get());
}

Status ModularFileSystem::SetOption(const std::string& name,
                                    const std::vector<string>& values) {
  if (ops_->set_filesystem_configuration == nullptr) {
    return errors::Unimplemented(
        "Filesystem does not support SetConfiguration()");
  }
  if (values.empty()) {
    return errors::InvalidArgument(
        "SetConfiguration() needs number of values > 0");
  }

  TF_Filesystem_Option option;
  memset(&option, 0, sizeof(option));
  option.name = const_cast<char*>(name.c_str());
  TF_Filesystem_Option_Value option_value;
  memset(&option_value, 0, sizeof(option_value));
  option_value.type_tag = TF_Filesystem_Option_Type_Buffer;
  option_value.num_values = values.size();
  std::vector<TF_Filesystem_Option_Value_Union> option_values(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    memset(&option_values[i], 0, sizeof(option_values[i]));
    option_values[i].buffer_val.buf = const_cast<char*>(values[i].c_str());
    option_values[i].buffer_val.buf_length = values[i].size();
  }
  option_value.values = &option_values[0];
  option.value = &option_value;
  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->set_filesystem_configuration(filesystem_.get(), &option, 1,
                                     plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::SetOption(const std::string& name,
                                    const std::vector<int64_t>& values) {
  if (ops_->set_filesystem_configuration == nullptr) {
    return errors::Unimplemented(
        "Filesystem does not support SetConfiguration()");
  }
  if (values.empty()) {
    return errors::InvalidArgument(
        "SetConfiguration() needs number of values > 0");
  }

  TF_Filesystem_Option option;
  memset(&option, 0, sizeof(option));
  option.name = const_cast<char*>(name.c_str());
  TF_Filesystem_Option_Value option_value;
  memset(&option_value, 0, sizeof(option_value));
  option_value.type_tag = TF_Filesystem_Option_Type_Int;
  option_value.num_values = values.size();
  std::vector<TF_Filesystem_Option_Value_Union> option_values(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    memset(&option_values[i], 0, sizeof(option_values[i]));
    option_values[i].int_val = values[i];
  }
  option_value.values = &option_values[0];
  option.value = &option_value;
  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->set_filesystem_configuration(filesystem_.get(), &option, 1,
                                     plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularFileSystem::SetOption(const std::string& name,
                                    const std::vector<double>& values) {
  if (ops_->set_filesystem_configuration == nullptr) {
    return errors::Unimplemented(
        "Filesystem does not support SetConfiguration()");
  }
  if (values.empty()) {
    return errors::InvalidArgument(
        "SetConfiguration() needs number of values > 0");
  }

  TF_Filesystem_Option option;
  memset(&option, 0, sizeof(option));
  option.name = const_cast<char*>(name.c_str());
  TF_Filesystem_Option_Value option_value;
  memset(&option_value, 0, sizeof(option_value));
  option_value.type_tag = TF_Filesystem_Option_Type_Real;
  option_value.num_values = values.size();
  std::vector<TF_Filesystem_Option_Value_Union> option_values(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    memset(&option_values[i], 0, sizeof(option_values[i]));
    option_values[i].real_val = values[i];
  }
  option_value.values = &option_values[0];
  option.value = &option_value;
  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->set_filesystem_configuration(filesystem_.get(), &option, 1,
                                     plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularRandomAccessFile::Read(uint64 offset, size_t n,
                                     StringPiece* result, char* scratch) const {
  if (ops_->read == nullptr)
    return errors::Unimplemented(
        tensorflow::strings::StrCat("Read() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  int64_t read =
      ops_->read(file_.get(), offset, n, scratch, plugin_status.get());
  if (read > 0) *result = StringPiece(scratch, read);
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularRandomAccessFile::Name(StringPiece* result) const {
  *result = filename_;
  return OkStatus();
}

Status ModularWritableFile::Append(StringPiece data) {
  if (ops_->append == nullptr)
    return errors::Unimplemented(tensorflow::strings::StrCat(
        "Append() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->append(file_.get(), data.data(), data.size(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Close() {
  if (ops_->close == nullptr)
    return errors::Unimplemented(
        tensorflow::strings::StrCat("Close() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->close(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Flush() {
  if (ops_->flush == nullptr) return OkStatus();

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->flush(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Sync() {
  if (ops_->sync == nullptr) return Flush();

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  ops_->sync(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status ModularWritableFile::Name(StringPiece* result) const {
  *result = filename_;
  return OkStatus();
}

Status ModularWritableFile::Tell(int64_t* position) {
  if (ops_->tell == nullptr)
    return errors::Unimplemented(
        tensorflow::strings::StrCat("Tell() not implemented for ", filename_));

  UniquePtrTo_TF_Status plugin_status(TF_NewStatus(), TF_DeleteStatus);
  *position = ops_->tell(file_.get(), plugin_status.get());
  return StatusFromTF_Status(plugin_status.get());
}

Status RegisterFilesystemPlugin(const std::string& dso_path) {
  // Step 1: Load plugin
  Env* env = Env::Default();
  void* dso_handle;
  TF_RETURN_IF_ERROR(env->LoadDynamicLibrary(dso_path.c_str(), &dso_handle));

  // Step 2: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "TF_InitPlugin", &dso_symbol),
      "Failed to load TF_InitPlugin symbol for DSO: ", dso_path);

  // Step 3: Call `TF_InitPlugin`
  TF_FilesystemPluginInfo info;
  memset(&info, 0, sizeof(info));
  auto TF_InitPlugin =
      reinterpret_cast<int (*)(TF_FilesystemPluginInfo*)>(dso_symbol);
  TF_InitPlugin(&info);

  // Step 4: Do the actual registration
  return filesystem_registration::RegisterFilesystemPluginImpl(&info);
}

}  // namespace tensorflow
