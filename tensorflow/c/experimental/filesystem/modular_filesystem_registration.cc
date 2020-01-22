/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/modular_filesystem_registration.h"

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/modular_filesystem.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

// Checks that all schemes provided by a plugin are valid.
// TODO(mihaimaruseac): More validation could be done here, based on supported
// charset, maximum length, etc. Punting it for later.
static Status ValidateScheme(const char* scheme) {
  if (scheme == nullptr)
    return errors::InvalidArgument(
        "Attempted to register filesystem with `nullptr` URI scheme");
  return Status::OK();
}

// Checks if the plugin and core ABI numbers match.
//
// If the numbers don't match, plugin cannot be loaded.
static Status CheckABI(int pluginABI, int coreABI, StringPiece where) {
  if (pluginABI != coreABI)
    return errors::FailedPrecondition(
        strings::StrCat("Plugin ABI (", pluginABI, ") for ", where,
                        " operations doesn't match expected core ABI (",
                        coreABI, "). Plugin cannot be loaded."));
  return Status::OK();
}

// Checks if the plugin and core ABI numbers match, for all operations.
//
// If the numbers don't match, plugin cannot be loaded.
//
// Uses the simpler `CheckABI(int, int, StringPiece)`.
static Status ValidateABI(const TF_FilesystemPluginInfo* info) {
  TF_RETURN_IF_ERROR(
      CheckABI(info->filesystem_ops_abi, TF_FILESYSTEM_OPS_ABI, "filesystem"));

  if (info->random_access_file_ops != nullptr)
    TF_RETURN_IF_ERROR(CheckABI(info->random_access_file_ops_abi,
                                TF_RANDOM_ACCESS_FILE_OPS_ABI,
                                "random access file"));

  if (info->writable_file_ops != nullptr)
    TF_RETURN_IF_ERROR(CheckABI(info->writable_file_ops_abi,
                                TF_WRITABLE_FILE_OPS_ABI, "writable file"));

  if (info->read_only_memory_region_ops != nullptr)
    TF_RETURN_IF_ERROR(CheckABI(info->read_only_memory_region_ops_abi,
                                TF_READ_ONLY_MEMORY_REGION_OPS_ABI,
                                "read only memory region"));

  return Status::OK();
}

// Checks if the plugin and core API numbers match, logging mismatches.
static void CheckAPI(int plugin_API, int core_API, StringPiece where) {
  if (plugin_API != core_API) {
    VLOG(0) << "Plugin API (" << plugin_API << ") for " << where
            << " operations doesn't match expected core API (" << core_API
            << "). Plugin will be loaded but functionality might be missing.";
  }
}

// Checks if the plugin and core API numbers match, for all operations.
//
// Uses the simpler `CheckAPIHelper(int, int, StringPiece)`.
static void ValidateAPI(const TF_FilesystemPluginInfo* info) {
  CheckAPI(info->filesystem_ops_api, TF_FILESYSTEM_OPS_API, "filesystem");

  if (info->random_access_file_ops != nullptr)
    CheckAPI(info->random_access_file_ops_api, TF_RANDOM_ACCESS_FILE_OPS_API,
             "random access file");

  if (info->writable_file_ops != nullptr)
    CheckAPI(info->writable_file_ops_api, TF_WRITABLE_FILE_OPS_API,
             "writable file");

  if (info->read_only_memory_region_ops != nullptr)
    CheckAPI(info->read_only_memory_region_ops_api,
             TF_READ_ONLY_MEMORY_REGION_OPS_API, "read only memory region");
}

// Validates the filesystem operations supplied by the plugin.
static Status ValidateHelper(const TF_FilesystemOps* ops) {
  if (ops == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without operations");

  if (ops->init == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `init` operation");

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `cleanup` operation");

  return Status::OK();
}

// Validates the random access file operations supplied by the plugin.
static Status ValidateHelper(const TF_RandomAccessFileOps* ops) {
  if (ops == nullptr) {
    // We allow filesystems where files can only be written to (from TF code)
    return Status::OK();
  }

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `cleanup` operation on random "
        "access files");

  return Status::OK();
}

// Validates the writable file operations supplied by the plugin.
static Status ValidateHelper(const TF_WritableFileOps* ops) {
  if (ops == nullptr) {
    // We allow read-only filesystems
    return Status::OK();
  }

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `cleanup` operation on writable "
        "files");

  return Status::OK();
}

// Validates the read only memory region operations given by the plugin.
static Status ValidateHelper(const TF_ReadOnlyMemoryRegionOps* ops) {
  if (ops == nullptr) {
    // read only memory region support is always optional
    return Status::OK();
  }

  if (ops->cleanup == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `cleanup` operation on read "
        "only memory regions");

  if (ops->data == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `data` operation on read only "
        "memory regions");

  if (ops->length == nullptr)
    return errors::FailedPrecondition(
        "Trying to register filesystem without `length` operation on read only "
        "memory regions");

  return Status::OK();
}

// Validates the operations supplied by the plugin.
//
// Uses the 4 simpler `ValidateHelper(const TF_...*)` to validate each
// individual function table and then checks that the function table for a
// specific file type exists if the plugin offers support for creating that
// type of files.
static Status ValidateOperations(const TF_FilesystemPluginInfo* info) {
  TF_RETURN_IF_ERROR(ValidateHelper(info->filesystem_ops));
  TF_RETURN_IF_ERROR(ValidateHelper(info->random_access_file_ops));
  TF_RETURN_IF_ERROR(ValidateHelper(info->writable_file_ops));
  TF_RETURN_IF_ERROR(ValidateHelper(info->read_only_memory_region_ops));

  if (info->filesystem_ops->new_random_access_file != nullptr &&
      info->random_access_file_ops == nullptr)
    return errors::FailedPrecondition(
        "Filesystem allows creation of random access files but no "
        "operations on them have been supplied.");

  if ((info->filesystem_ops->new_writable_file != nullptr ||
       info->filesystem_ops->new_appendable_file != nullptr) &&
      info->writable_file_ops == nullptr)
    return errors::FailedPrecondition(
        "Filesystem allows creation of writable files but no "
        "operations on them have been supplied.");

  if (info->filesystem_ops->new_read_only_memory_region_from_file != nullptr &&
      info->read_only_memory_region_ops == nullptr)
    return errors::FailedPrecondition(
        "Filesystem allows creation of readonly memory regions but no "
        "operations on them have been supplied.");

  return Status::OK();
}

// Copies a function table from plugin memory space to core memory space.
//
// This has three benefits:
//   * allows having newer plugins than the current core TensorFlow: the
//     additional entries in the plugin's table are just discarded;
//   * allows having older plugins than the current core TensorFlow (though
//     we are still warning users): the entries that core TensorFlow expects
//     but plugins didn't provide will be set to `nullptr` values and core
//     TensorFlow will know to not call these on behalf of users;
//   * increased security as plugins will not be able to alter function table
//     after loading up. Thus, malicious plugins can't alter functionality to
//     probe for gadgets inside core TensorFlow. We can even protect the area
//     of memory where the copies reside to not allow any more writes to it
//     after all copies are created.
template <typename T>
static std::unique_ptr<const T> CopyToCore(const T* plugin_ops,
                                           size_t plugin_size) {
  if (plugin_ops == nullptr) return nullptr;

  size_t copy_size = std::min(plugin_size, sizeof(T));
  auto core_ops = tensorflow::MakeUnique<T>();
  memset(core_ops.get(), 0, sizeof(T));
  memcpy(core_ops.get(), plugin_ops, copy_size);
  return core_ops;
}

// Registers one filesystem from the plugin.
static Status RegisterFileSystem(const TF_FilesystemPluginInfo* info) {
  // Step 1: Copy all the function tables to core TensorFlow memory space
  auto core_filesystem_ops = CopyToCore<TF_FilesystemOps>(
      info->filesystem_ops, info->filesystem_ops_size);
  auto core_random_access_file_ops = CopyToCore<TF_RandomAccessFileOps>(
      info->random_access_file_ops, info->random_access_file_ops_size);
  auto core_writable_file_ops = CopyToCore<TF_WritableFileOps>(
      info->writable_file_ops, info->writable_file_ops_size);
  auto core_read_only_memory_region_ops =
      CopyToCore<TF_ReadOnlyMemoryRegionOps>(
          info->read_only_memory_region_ops,
          info->read_only_memory_region_ops_size);

  // Step 2: Initialize the opaque filesystem structure
  auto filesystem = tensorflow::MakeUnique<TF_Filesystem>();
  TF_Status* c_status = TF_NewStatus();
  Status status = Status::OK();
  core_filesystem_ops->init(filesystem.get(), c_status);
  status = Status(c_status->status);
  TF_DeleteStatus(c_status);
  if (!status.ok()) return status;

  // Step 3: Actual registration
  return Env::Default()->RegisterFileSystem(
      info->scheme, tensorflow::MakeUnique<tensorflow::ModularFileSystem>(
                        std::move(filesystem), std::move(core_filesystem_ops),
                        std::move(core_random_access_file_ops),
                        std::move(core_writable_file_ops),
                        std::move(core_read_only_memory_region_ops)));
}

// Registers all filesystems, if plugin is providing valid information.
//
// Extracted to a separate function so that pointers inside `info` are freed
// by the caller regardless of whether validation/registration failed or not.
static Status ValidateAndRegisterFilesystems(
    const TF_FilesystemPluginInfo* info) {
  TF_RETURN_IF_ERROR(ValidateScheme(info->scheme));
  TF_RETURN_IF_ERROR(ValidateABI(info));
  ValidateAPI(info);  // we just warn on API number mismatch
  TF_RETURN_IF_ERROR(ValidateOperations(info));
  TF_RETURN_IF_ERROR(RegisterFileSystem(info));
  return Status::OK();
}

// Alocates memory in plugin DSO.
//
// Provided by core TensorFlow so that it can free this memory after DSO is
// loaded and filesystem information has been used to register the filesystem.
static void* basic_allocator(size_t size) { return calloc(1, size); }

namespace filesystem_registration {

Status RegisterFilesystemPluginImpl(const std::string& dso_path) {
  // Step 1: Load plugin
  Env* env = Env::Default();
  void* dso_handle;
  TF_RETURN_IF_ERROR(env->LoadLibrary(dso_path.c_str(), &dso_handle));

  // Step 2: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "TF_InitPlugin", &dso_symbol));

  // Step 3: Call `TF_InitPlugin`
  TF_FilesystemPluginInfo* info = nullptr;
  auto TF_InitPlugin = reinterpret_cast<int (*)(
      decltype(&basic_allocator), TF_FilesystemPluginInfo**)>(dso_symbol);
  int num_schemes = TF_InitPlugin(&basic_allocator, &info);
  if (num_schemes < 0 || info == nullptr)
    return errors::InvalidArgument("DSO returned invalid filesystem data");

  // Step 4: Validate and register all filesystems
  // Try to register as many filesystems as possible.
  // Free memory once we no longer need it
  Status status;
  for (int i = 0; i < num_schemes; i++) {
    status.Update(ValidateAndRegisterFilesystems(&info[i]));
    free(info[i].scheme);
    free(info[i].filesystem_ops);
    free(info[i].random_access_file_ops);
    free(info[i].writable_file_ops);
    free(info[i].read_only_memory_region_ops);
  }
  free(info);
  return status;
}

}  // namespace filesystem_registration

}  // namespace tensorflow
