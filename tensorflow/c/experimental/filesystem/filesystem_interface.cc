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
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"

#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/util/ptr_util.h"

/// This translation unit is linked in core TensorFlow and provides the
/// functionality needed for plugin registration to check ABI/API compatibility,
/// to ensure required methods are present, to ensure plugins are not allowed to
/// change functionality after being loaded and to register the filesystems
/// provided by a plugin. Consult the header file for more information about
/// how this is achieved.

namespace tensorflow {
namespace {

// Checks if the plugin and core ABI numbers match, filling in `status`.
//
// If the numbers don't match, plugin cannot be loaded.
static bool CheckABIHelper(int pluginABI, int coreABI, StringPiece where,
                           TF_Status* status) {
  if (pluginABI != coreABI) {
    TF_SetStatus(
        status, TF_FAILED_PRECONDITION,
        strings::StrCat("Plugin ABI (", pluginABI, ") for ", where,
                        " operations doesn't match expected core ABI (",
                        coreABI, "). Plugin cannot be loaded.")
            .c_str());
    return false;
  }

  return true;
}

// Checks if the plugin and core ABI numbers match, for all operations.
//
// If the numbers don't match, plugin cannot be loaded.
//
// Uses the simpler `CheckABIHelper(int, int, StringPiece, TF_Status*)`
static bool CheckABI(
    int pluginFilesystemOpsABI,
    const TF_RandomAccessFileOps* pluginRandomAccessFileOps,
    int pluginRandomAccessFileOpsABI,
    const TF_WritableFileOps* pluginWritableFileOps,
    int pluginWritableFileOpsABI,
    const TF_ReadOnlyMemoryRegionOps* pluginReadOnlyMemoryRegionOps,
    int pluginReadOnlyMemoryRegionOpsABI, TF_Status* status) {
  if (!CheckABIHelper(pluginFilesystemOpsABI, TF_FILESYSTEM_OPS_ABI,
                      "filesystem", status))
    return false;

  if (pluginRandomAccessFileOps != nullptr &&
      !CheckABIHelper(pluginRandomAccessFileOpsABI,
                      TF_RANDOM_ACCESS_FILE_OPS_ABI, "random access file",
                      status))
    return false;

  if (pluginWritableFileOps != nullptr &&
      !CheckABIHelper(pluginWritableFileOpsABI, TF_WRITABLE_FILE_OPS_ABI,
                      "writable file", status))
    return false;

  if (pluginReadOnlyMemoryRegionOps != nullptr &&
      !CheckABIHelper(pluginReadOnlyMemoryRegionOpsABI,
                      TF_READ_ONLY_MEMORY_REGION_OPS_ABI,
                      "read only memory region", status))
    return false;

  return true;
}

// Checks if the plugin and core API numbers match, logging mismatches.
static void CheckAPIHelper(int pluginAPI, int coreAPI, StringPiece where) {
  if (pluginAPI != coreAPI) {
    VLOG(0) << "Plugin API (" << pluginAPI << ") for " << where
            << " operations doesn't match expected core API (" << coreAPI
            << "). Plugin will be loaded but functionality might be missing.";
  }
}

// Checks if the plugin and core API numbers match, for all operations.
//
// Uses the simpler `CheckAPIHelper(int, int, StringPiece)`.
static void CheckAPI(
    int pluginFilesystemOpsAPI,
    const TF_RandomAccessFileOps* pluginRandomAccessFileOps,
    int pluginRandomAccessFileOpsAPI,
    const TF_WritableFileOps* pluginWritableFileOps,
    int pluginWritableFileOpsAPI,
    const TF_ReadOnlyMemoryRegionOps* pluginReadOnlyMemoryRegionOps,
    int pluginReadOnlyMemoryRegionOpsAPI) {
  CheckAPIHelper(pluginFilesystemOpsAPI, TF_FILESYSTEM_OPS_API, "filesystem");

  if (pluginRandomAccessFileOps != nullptr)
    CheckAPIHelper(pluginRandomAccessFileOpsAPI, TF_RANDOM_ACCESS_FILE_OPS_API,
                   "random access file");

  if (pluginWritableFileOps != nullptr)
    CheckAPIHelper(pluginWritableFileOpsAPI, TF_WRITABLE_FILE_OPS_API,
                   "writable file");

  if (pluginReadOnlyMemoryRegionOps != nullptr)
    CheckAPIHelper(pluginReadOnlyMemoryRegionOpsAPI,
                   TF_READ_ONLY_MEMORY_REGION_OPS_API,
                   "read only memory region");
}

// Validates the filesystem operations supplied by the plugin.
static bool ValidateHelper(const TF_FilesystemOps* ops, TF_Status* status) {
  if (ops == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without operations");
    return false;
  }

  if (ops->init == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `init` operation");
    return false;
  }

  if (ops->cleanup == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `cleanup` operation");
    return false;
  }

  return true;
}

// Validates the random access file operations supplied by the plugin.
static bool ValidateHelper(const TF_RandomAccessFileOps* ops,
                           TF_Status* status) {
  if (ops == nullptr) {
    // We allow filesystems where files can only be written to (from TF code)
    return true;
  }

  if (ops->cleanup == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `cleanup` operation on "
                 "random access files");
    return false;
  }

  return true;
}

// Validates the writable file operations supplied by the plugin.
static bool ValidateHelper(const TF_WritableFileOps* ops, TF_Status* status) {
  if (ops == nullptr) {
    // We allow read-only filesystems
    return true;
  }

  if (ops->cleanup == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `cleanup` operation on "
                 "writable files");
    return false;
  }

  return true;
}

// Validates the read only memory region operations given by the plugin.
static bool ValidateHelper(const TF_ReadOnlyMemoryRegionOps* ops,
                           TF_Status* status) {
  if (ops == nullptr) {
    // read only memory region support is always optional
    return true;
  }

  if (ops->cleanup == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `cleanup` operation on "
                 "read only memory regions");
    return false;
  }

  if (ops->data == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `data` operation on "
                 "read only memory regions");
    return false;
  }

  if (ops->length == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Trying to register filesystem without `length` operation on "
                 "read only memory regions");
    return false;
  }

  return true;
}

// Validates the operations supplied by the plugin.
//
// Uses the 4 simpler `ValidateHelper(const TF_..., TF_Status*)` to validate
// each individual function table and then checks that the function table for a
// specific file type exists if the plugin offers support for creating that
// type of files.
static bool Validate(
    const TF_FilesystemOps* pluginFilesystemOps,
    const TF_RandomAccessFileOps* pluginRandomAccessFileOps,
    const TF_WritableFileOps* pluginWritableFileOps,
    const TF_ReadOnlyMemoryRegionOps* pluginReadOnlyMemoryRegionOps,
    TF_Status* status) {
  if (!ValidateHelper(pluginFilesystemOps, status)) return false;
  if (!ValidateHelper(pluginRandomAccessFileOps, status)) return false;
  if (!ValidateHelper(pluginWritableFileOps, status)) return false;
  if (!ValidateHelper(pluginReadOnlyMemoryRegionOps, status)) return false;

  if (pluginFilesystemOps->new_random_access_file != nullptr &&
      pluginRandomAccessFileOps == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Filesystem allows creation of random access files but no "
                 "operations on them have been supplied.");
    return false;
  }

  if ((pluginFilesystemOps->new_writable_file != nullptr ||
       pluginFilesystemOps->new_appendable_file != nullptr) &&
      pluginWritableFileOps == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Filesystem allows creation of writable files but no "
                 "operations on them have been supplied.");
    return false;
  }

  if (pluginFilesystemOps->new_read_only_memory_region_from_file != nullptr &&
      pluginReadOnlyMemoryRegionOps == nullptr) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Filesystem allows creation of readonly memory regions but no "
                 "operations on them have been supplied.");
    return false;
  }

  return true;
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
static std::unique_ptr<const T> CopyToCore(const T* pluginOps,
                                           size_t pluginSize) {
  if (pluginOps == nullptr) return nullptr;

  size_t copySize = sizeof(T);
  if (pluginSize < copySize) {
    copySize = pluginSize;
  }

  auto coreOps = tensorflow::MakeUnique<T>();
  memcpy(const_cast<T*>(coreOps.get()), pluginOps, copySize);
  return coreOps;
}

}  // namespace
}  // namespace tensorflow

void RegisterFilesystemPlugin(
    int pluginFilesystemOpsABI, int pluginFilesystemOpsAPI,
    size_t pluginFilesystemOpsSize, int pluginRandomAccessFileOpsABI,
    int pluginRandomAccessFileOpsAPI, size_t pluginRandomAccessFileOpsSize,
    int pluginWritableFileOpsABI, int pluginWritableFileOpsAPI,
    size_t pluginWritableFileOpsSize, int pluginReadOnlyMemoryRegionOpsABI,
    int pluginReadOnlyMemoryRegionOpsAPI,
    size_t pluginReadOnlyMemoryRegionOpsSize, const char* scheme,
    const TF_FilesystemOps* pluginFilesystemOps,
    const TF_RandomAccessFileOps* pluginRandomAccessFileOps,
    const TF_WritableFileOps* pluginWritableFileOps,
    const TF_ReadOnlyMemoryRegionOps* pluginReadOnlyMemoryRegionOps,
    TF_Status* status) {
  if (scheme == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "`scheme` argument must not be `nullptr`.");
    return;
  }

  // ABI numbers must match exactly for plugin to be loaded
  if (!tensorflow::CheckABI(pluginFilesystemOpsABI, pluginRandomAccessFileOps,
                            pluginRandomAccessFileOpsABI, pluginWritableFileOps,
                            pluginWritableFileOpsABI,
                            pluginReadOnlyMemoryRegionOps,
                            pluginReadOnlyMemoryRegionOpsABI, status)) {
    return;
  }

  // API numbers should match but mismatch doesn't block plugin load
  tensorflow::CheckAPI(pluginFilesystemOpsAPI, pluginRandomAccessFileOps,
                       pluginRandomAccessFileOpsAPI, pluginWritableFileOps,
                       pluginWritableFileOpsAPI, pluginReadOnlyMemoryRegionOps,
                       pluginReadOnlyMemoryRegionOpsAPI);

  // Plugin can only be loaded if all supplied ops are valid
  if (!tensorflow::Validate(pluginFilesystemOps, pluginRandomAccessFileOps,
                            pluginWritableFileOps,
                            pluginReadOnlyMemoryRegionOps, status)) {
    return;
  }

  // Copy all the function tables to core TensorFlow memory space
  auto coreFilesystemOps = tensorflow::CopyToCore<TF_FilesystemOps>(
      pluginFilesystemOps, pluginFilesystemOpsSize);
  auto coreRandomAccessFileOps = tensorflow::CopyToCore<TF_RandomAccessFileOps>(
      pluginRandomAccessFileOps, pluginRandomAccessFileOpsSize);
  auto coreWritableFileOps = tensorflow::CopyToCore<TF_WritableFileOps>(
      pluginWritableFileOps, pluginWritableFileOpsSize);
  auto coreReadOnlyMemoryRegionOps =
      tensorflow::CopyToCore<TF_ReadOnlyMemoryRegionOps>(
          pluginReadOnlyMemoryRegionOps, pluginReadOnlyMemoryRegionOpsSize);

  // Initialize the opaque filesystem structure
  auto filesystem = tensorflow::MakeUnique<TF_Filesystem>();
  coreFilesystemOps->init(filesystem.get(), status);
  if (!status->status.ok()) {
    coreFilesystemOps->cleanup(filesystem.get());
    return;
  }

  // Register new filesystem
  // TODO(mihaimaruseac): Will come in a subsequent CL
}
