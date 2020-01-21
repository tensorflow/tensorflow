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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_FILESYSTEM_INTERFACE_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_FILESYSTEM_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/tf_file_statistics.h"
#include "tensorflow/c/tf_status.h"

/// This is the interop header between core TensorFlow and modular filesystem
/// plugins (see initial RFC https://github.com/tensorflow/community/pull/101).
///
/// Both core TensorFlow and every plugin will use this header. The associated
/// `.cc` file is only used by core TensorFlow to implement checking needed for
/// plugin registration and ensuring API and ABI compatibility. Plugin authors
/// don't need to read the `.cc` file but they should consult every section of
/// this file to ensure a compliant plugin can be built and that the plugin can
/// be used without recompilation in the widest range of TensorFlow versions.
///
/// The header is divided into sections, as follows:
///   1. Opaque plugin private data structures and wrappers for type safety;
///   2. Function tables for plugin functionality;
///   3. Versioning metadata;
///   4. Plugin registration API and the DSO entry point.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// SECTION 1. Opaque data structures to hold plugin specific data
/// ----------------------------------------------------------------------------
///
/// The following data structures incorporate a `void*` that is opaque to
/// TensorFlow but can be used by each filesystem plugin to represent internal
/// data.
///
/// We prefer to have these structures instead of passing `void*` into
/// method signatures to have some type of type safety: for example, operations
/// that are only valid on random access files have a `TF_RandomAccessFile`
/// argument.
///
/// Lifetime: The wrapper data structures are owned by core TensorFlow. The data
/// pointed to by the `void*` members is always owned by the plugin. The plugin
/// will provide functions to call to allocate and deallocate this data (see
/// next section) and core TensorFlow ensures to call these at the proper time.
///
/// Plugins will never receive a `TF_*` pointer that is `nullptr`. Core
/// TensorFlow will never touch the `void*` wrapped by these structures, except
/// to initialize it as `nullptr`.

typedef struct TF_RandomAccessFile {
  void* plugin_file;
} TF_RandomAccessFile;

typedef struct TF_WritableFile {
  void* plugin_file;
} TF_WritableFile;

typedef struct TF_ReadOnlyMemoryRegion {
  void* plugin_memory_region;
} TF_ReadOnlyMemoryRegion;

typedef struct TF_Filesystem {
  void* plugin_filesystem;
} TF_Filesystem;

/// SECTION 2. Function tables for functionality provided by plugins
/// ----------------------------------------------------------------------------
///
/// The following data structures represent the function tables for operations
/// that plugins provide (some are mandatory, some are optional, with or without
/// a default implementation).
///
/// Each plugin implements the operations that are supported and TensorFlow will
/// properly handle the cases when an operation is not supported (i.e., return
/// the corresponding `Status` value).
///
/// REQUIRED OPERATIONS: All required operations are marked as such, including
/// operations which are conditionally required. If the presence of an operation
/// `foo` requires operation `bar` to be present, this is specified in `foo`. If
/// the entire set of operations in a table is not provided, use `nullptr` for
/// the struct pointer (e.g., when a file type is not supported).
///
/// DEFAULT IMPLEMENTATIONS: Some operations have default implementations that
/// TensorFlow uses in case the plugin doesn't supply its own version. An
/// operation `foo` might have a default implementation which uses `bar` and
/// `foobar`. If the plugin supplies `bar` and `foobar`, TensorFlow can use the
/// default implementation of `foo`.
///
/// During plugin loading, plugins will call the registration function provided
/// by this interface, supplying values for each of these structures. Core
/// TensorFlow checks that the plugin supplies all mandatory operations and
/// then copies these tables to a different memory location, marking the new
/// operation tables as read-only. Once a plugin is loaded, none of these
/// operation pointers may change.
///
/// There are 4 function tables: one for each of the 3 file objects in
/// TensorFlow (i.e., `RandomAccessFile`, `WritableFile`,
/// `ReadOnlyMemoryRegion`) and one for all the operations a `Filesystem`
/// implements. Each of them is in a 1-to-1 correspondence with the wrapper
/// structures from the first section: these tables only contain function
/// pointers that operate on the corresponding data. Thus, the first argument of
/// each of these functions is a pointer to the paired struct and this argument
/// can be used to track state in between calls (from an object oriented point
/// of view, this can be viewed as a "vtable" for a "class" -- that is the
/// corresponding struct above --; the first argument is in place of `this`).
///
/// Except where noted otherwise, all pointer arguments are owned by core
/// TensorFlow and are guaranteed to not be `nullptr`.
///
/// All path-like arguments are null terminated `char*` strings. Plugins can
/// assume that before any function using path arguments is invoked, the path is
/// made canonical by calling the function provided by `translate_name` or a
/// default implementation of that (supplied by core TensorFlow).
///
/// The only time the pointer to the `TF_*` structures from section 1 is not
/// marked `const` in these functions is when these function are either
/// allocating or deallocating the plugin specific data. That is, in the 4
/// `cleanup` functions (one for each data structure), the `init` function for
/// `TF_Filesystem` and the `new_*` methods of `TF_FilesystemOps` to initialize
/// the 3 types of files. In all other cases, there is no need to modify the
/// address of the opaque data pointer, hence the wrapper pointer is marked
/// `const`.
///
/// For consistency, the arguments on all these functions follow the same
/// pattern: first we have the opaque pointer argument ("this" above), then the
/// input arguments, then the in-out arguments (if any) and we finish the
/// argument list with the out arguments. We only use the return type for an out
/// parameter if that is a plain C type, as this ensures ABI compatibility
/// (returning structures has issues in case compiler options affect
/// optimizations such as RVO). If a status needs to be returned from these
/// methods, the last argument is always a `TF_Status *` (or an array of such
/// pointers) owned by core TensorFlow and guaranteed to not be `nullptr`.
///
/// To ensure ABI and API compatibility, we have out-of-bounds data that is used
/// by both core TensorFlow and the plugin at load time. We don't include this
/// data in the structures here to prevent cases when padding/packing enabled by
/// different compiler options breaks compatibility. For more details about how
/// this is used, please consult next sections. Here we just wrap these tables
/// in lint warnings so that changes here cause changes to the versioning data
/// as well. Here is a short summary of what changes are allowed:
///   * adding a new method at the end of a table is allowed at any time;
///   * any other change to these tables is only allowed on a major TensorFlow
///     version change (e.g., from 2.x to 3.0). This is provided as an escape
///     hatch to allow cleaning up these tables. Since any of these changes
///     break ABI compatibility and cause all plugins to be recompiled, these
///     type of changes should be extremely rare.
///
/// Next section will detail this as well as some corner cases that are out of
/// scope for now.

// LINT.IfChange
typedef struct TF_RandomAccessFileOps {
  /// Releases resources associated with `*file`.
  ///
  /// Requires that `*file` is not used in any concurrent or subsequent
  /// operations.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  void (*cleanup)(TF_RandomAccessFile* file);

  /// Reads up to `n` bytes from `*file` starting at `offset`.
  ///
  /// The output is in `buffer`, core TensorFlow owns the buffer and guarantees
  /// that at least `n` bytes are available.
  ///
  /// Returns number of bytes read or -1 in case of error. Because of this
  /// constraint and the fact that `ssize_t` is not defined in `stdint.h`/C++
  /// standard, the return type is `int64_t`.
  ///
  /// This is thread safe.
  ///
  /// Note: the `buffer` argument is NOT a null terminated string!
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if exactly `n` bytes have been read.
  ///   * Must set `status` to `TF_OUT_OF_RANGE` if fewer than `n` bytes have
  ///     been read due to EOF.
  ///   * Must return -1 for any other error and must set `status` to any
  ///     other value to provide more information about the error.
  int64_t (*read)(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                  char* buffer, TF_Status* status);
} TF_RandomAccessFileOps;
// LINT.ThenChange(:random_access_file_ops_version)

// LINT.IfChange
typedef struct TF_WritableFileOps {
  /// Releases resources associated with `*file`.
  ///
  /// Requires that `*file` is not used in any concurrent or subsequent
  /// operations.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  void (*cleanup)(TF_WritableFile* file);

  /// Appends `buffer` of size `n` to `*file`.
  ///
  /// Core TensorFlow owns `buffer` and guarantees at least `n` bytes of storage
  /// that can be used to write data.
  ///
  /// Note: the `buffer` argument is NOT a null terminated string!
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if exactly `n` bytes have been written.
  ///   * Must set `status` to `TF_RESOURCE_EXHAUSTED` if fewer than `n` bytes
  ///     have been written, potentially due to quota/disk space.
  ///   * Might use any other error value for `status` to signal other errors.
  void (*append)(const TF_WritableFile* file, const char* buffer, size_t n,
                 TF_Status* status);

  /// Returns the current write position in `*file`.
  ///
  /// Plugins should ensure that the implementation is idempotent, 2 identical
  /// calls result in the same answer.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` and return current position if no error.
  ///   * Must set `status` to any other value and return -1 in case of error.
  int64_t (*tell)(const TF_WritableFile* file, TF_Status* status);

  /// Flushes `*file` and syncs contents to filesystem.
  ///
  /// This call might not block, and when it returns the contents might not have
  /// been fully persisted.
  ///
  /// DEFAULT IMPLEMENTATION: No op.
  void (*flush)(const TF_WritableFile* file, TF_Status* status);

  /// Syncs contents of `*file` with the filesystem.
  ///
  /// This call should block until filesystem confirms that all buffers have
  /// been flushed and persisted.
  ///
  /// DEFAULT IMPLEMENTATION: No op.
  void (*sync)(const TF_WritableFile* file, TF_Status* status);

  /// Closes `*file`.
  ///
  /// Flushes all buffers and deallocates all resources.
  ///
  /// Calling `close` must not result in calling `cleanup`.
  ///
  /// Core TensorFlow will never call `close` twice.
  void (*close)(const TF_WritableFile* file, TF_Status* status);
} TF_WritableFileOps;
// LINT.ThenChange(:writable_file_ops_version)

// LINT.IfChange
typedef struct TF_ReadOnlyMemoryRegionOps {
  /// Releases resources associated with `*region`.
  ///
  /// Requires that `*region` is not used in any concurrent or subsequent
  /// operations.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  void (*cleanup)(TF_ReadOnlyMemoryRegion* region);

  /// Returns a pointer to the memory region.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  const void* (*data)(const TF_ReadOnlyMemoryRegion* region);

  /// Returns the length of the memory region in bytes.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  uint64_t (*length)(const TF_ReadOnlyMemoryRegion* region);
} TF_ReadOnlyMemoryRegionOps;
// LINT.ThenChange(:read_only_memory_region_ops_version)

// LINT.IfChange
typedef struct TF_FilesystemOps {
  /// Acquires all resources used by the filesystem.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  void (*init)(TF_Filesystem* filesystem, TF_Status* status);

  /// Releases all resources used by the filesystem
  ///
  /// NOTE: TensorFlow does not unload DSOs. Thus, the only way a filesystem
  /// won't be registered anymore is if this function gets called by core
  /// TensorFlow and the `TF_Filesystem*` object is destroyed. However, due to
  /// registration being done in a static instance of `Env`, the destructor of
  /// `FileSystem` is never called (see
  /// https://github.com/tensorflow/tensorflow/issues/27535). In turn, this
  /// function will never be called. There are plans to refactor registration
  /// and fix this.
  ///
  /// TODO(mihaimaruseac): After all filesystems are converted, revisit note.
  ///
  /// This operation must be provided. See "REQUIRED OPERATIONS" above.
  void (*cleanup)(TF_Filesystem* filesystem);

  /// Creates a new random access read-only file from given `path`.
  ///
  /// After this call `file` may be concurrently accessed by multiple threads.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `file` was updated.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to an
  ///     existing file or one of the parent entries in `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` points to a
  ///     directory or if it is invalid (e.g., malformed, or has a parent entry
  ///     which is a file).
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// REQUIREMENTS: If plugins implement this, they must also provide a filled
  /// `TF_RandomAccessFileOps` table. See "REQUIRED OPERATIONS" above.
  void (*new_random_access_file)(const TF_Filesystem* filesystem,
                                 const char* path, TF_RandomAccessFile* file,
                                 TF_Status* status);

  /// Creates an object to write to a file with the specified `path`.
  ///
  /// If the file already exists, it is deleted and recreated. The `file` object
  /// must only be accessed by one thread at a time.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `file` was updated.
  ///   * Must set `status` to `TF_NOT_FOUND` if one of the parents entries in
  ///     `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` points to a
  ///     directory or if it is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// REQUIREMENTS: If plugins implement this, they must also provide a filled
  /// `TF_WritableFileOps` table. See "REQUIRED OPERATIONS" above.
  void (*new_writable_file)(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status);

  /// Creates an object to append to a file with the specified `path`.
  ///
  /// If the file doesn't exists, it is first created with empty contents.
  /// The `file` object must only be accessed by one thread at a time.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `file` was updated.
  ///   * Must set `status` to `TF_NOT_FOUND` if one of the parents entries in
  ///     `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` points to a
  ///     directory or if it is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// REQUIREMENTS: If plugins implement this, they must also provide a filled
  /// `TF_WritableFileOps` table. See "REQUIRED OPERATIONS" above.
  void (*new_appendable_file)(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status);

  /// Creates a read-only region of memory from contents of `path`.
  ///
  /// After this call `region` may be concurrently accessed by multiple threads.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `region` was updated.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to an
  ///     existing file or one of the parent entries in `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` points to a
  ///     directory or if it is invalid.
  ///   * Must set `status` to `TF_INVALID_ARGUMENT` if `path` points to an
  ///     empty file.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// REQUIREMENTS: If plugins implement this, they must also provide a filled
  /// `TF_ReadOnlyMemoryRegionOps` table. See "REQUIRED OPERATIONS" above.
  void (*new_read_only_memory_region_from_file)(const TF_Filesystem* filesystem,
                                                const char* path,
                                                TF_ReadOnlyMemoryRegion* region,
                                                TF_Status* status);

  /// Creates the directory specified by `path`, assuming parent exists.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if directory was created.
  ///   * Must set `status` to `TF_NOT_FOUND` if one of the parents entries in
  ///     `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid.
  ///   * Must set `status` to `TF_ALREADY_EXISTS` if `path` already exists.
  ///   * Might use any other error value for `status` to signal other errors.
  void (*create_dir)(const TF_Filesystem* filesystem, const char* path,
                     TF_Status* status);

  /// Creates the directory specified by `path` and all needed ancestors.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if directory was created.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid or
  ///     if it exists but is not a directory.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// NOTE: The requirements specify that `TF_ALREADY_EXISTS` is not returned if
  /// directory exists. Similarly, `TF_NOT_FOUND` is not be returned, as the
  /// missing directory entry and all its descendants will be created by the
  /// plugin.
  ///
  /// DEFAULT IMPLEMENTATION: Creates directories one by one. Needs
  /// `path_exists`, `is_directory`, and `create_dir`.
  void (*recursively_create_dir)(const TF_Filesystem* filesystem,
                                 const char* path, TF_Status* status);

  /// Deletes the file specified by `path`.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if file was deleted.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` points to a
  ///     directory or if it is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  void (*delete_file)(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status);

  /// Deletes the empty directory specified by `path`.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if directory was deleted.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` does not point
  ///     to a directory, if `path` is invalid, or if directory is not empty.
  ///   * Might use any other error value for `status` to signal other errors.
  void (*delete_dir)(const TF_Filesystem* filesystem, const char* path,
                     TF_Status* status);

  /// Deletes the directory specified by `path` and all its contents.
  ///
  /// This is accomplished by traversing directory tree rooted at `path` and
  /// deleting entries as they are encountered, from leaves to root. Each plugin
  /// is free to choose a different approach which obtains similar results.
  ///
  /// On successful deletion, `status` must be `TF_OK` and `*undeleted_files`
  /// and `*undeleted_dirs` must be 0. On unsuccessful deletion, `status` must
  /// be set to the reason why one entry couldn't be removed and the proper
  /// count must be updated. If the deletion is unsuccessful because the
  /// traversal couldn't start, `*undeleted_files` must be set to 0 and
  /// `*undeleted_dirs` must be set to 1.
  ///
  /// TODO(mihaimaruseac): After all filesystems are converted, consider
  /// invariant about `*undeleted_files` and `*undeleted_dirs`.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if directory was deleted.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// DEFAULT IMPLEMENTATION: Does a BFS traversal of tree rooted at `path`,
  /// deleting entries as needed. Needs `path_exists`, `get_children`,
  /// `is_directory`, `delete_file`, and `delete_dir`.
  void (*delete_recursively)(const TF_Filesystem* filesystem, const char* path,
                             uint64_t* undeleted_files,
                             uint64_t* undeleted_dirs, TF_Status* status);

  /// Renames the file given by `src` to that in `dst`.
  ///
  /// Replaces `dst` if it exists. In case of error, both `src` and `dst` keep
  /// the same state as before the call.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if rename was completed.
  ///   * Must set `status` to `TF_NOT_FOUND` if one of the parents entries in
  ///     either `src` or `dst` doesn't exist or if the specified `src` path
  ///     doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if either `src` or
  ///     `dst` is a directory or if either of them is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// DEFAULT IMPLEMENTATION: Copies file and deletes original. Needs
  /// `copy_file`. and `delete_file`.
  void (*rename_file)(const TF_Filesystem* filesystem, const char* src,
                      const char* dst, TF_Status* status);

  /// Copies the file given by `src` to that in `dst`.
  ///
  /// Similar to `rename_file`, but both `src` and `dst` exist after this call
  /// with the same contents. In case of error, both `src` and `dst` keep the
  /// same state as before the call.
  ///
  /// If `dst` is a directory, creates a file with the same name as the source
  /// inside the target directory.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if rename was completed.
  ///   * Must set `status` to `TF_NOT_FOUND` if one of the parents entries in
  ///     either `src` or `dst` doesn't exist or if the specified `src` path
  ///     doesn't exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if either `src` or
  ///     `dst` is a directory or if either of them is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// DEFAULT IMPLEMENTATION: Reads from `src` and writes to `dst`. Needs
  /// `new_random_access_file` and `new_writable_file`.
  void (*copy_file)(const TF_Filesystem* filesystem, const char* src,
                    const char* dst, TF_Status* status);

  /// Checks if `path` exists.
  ///
  /// Note that this doesn't differentiate between files and directories.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `path` exists.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to a
  ///     filesystem entry.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  void (*path_exists)(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status);

  /// Checks if all values in `paths` exist in the filesystem.
  ///
  /// Returns `true` if and only if calling `path_exists` on each entry in
  /// `paths` would set `status` to `TF_OK`.
  ///
  /// Caller guarantees that:
  ///   * `paths` has exactly `num_files` entries.
  ///   * `statuses` is either null or an array of `num_files` non-null elements
  ///     of type `TF_Status*`.
  ///
  /// If `statuses` is not null, plugins must fill each element with detailed
  /// status for each file, as if calling `path_exists` on each one. Core
  /// TensorFlow initializes the `statuses` array and plugins must use
  /// `TF_SetStatus` to set each element instead of directly assigning.
  ///
  /// DEFAULT IMPLEMENTATION: Checks existence of every file. Needs
  /// `path_exists`.
  bool (*paths_exist)(const TF_Filesystem* filesystem, char** paths,
                      int num_files, TF_Status** statuses);

  /// Obtains statistics for the given `path`.
  ///
  /// Updates `stats` only if `status` is set to `TF_OK`.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `path` exists.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to a
  ///     filesystem entry.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  void (*stat)(const TF_Filesystem* filesystem, const char* path,
               TF_FileStatistics* stats, TF_Status* status);

  /// Checks whether the given `path` is a directory or not.
  ///
  /// If `status` is not `TF_OK`, returns `false`, otherwise returns the same
  /// as the `is_directory` member of a `TF_FileStatistics` that would be used
  /// on the equivalent call of `stat`.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `path` exists.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to a
  ///     filesystem entry.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// DEFAULT IMPLEMENTATION: Gets statistics about `path`. Needs `stat`.
  bool (*is_directory)(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status);

  /// Returns the size of the file given by `path`.
  ///
  /// If `status` is not `TF_OK`, return value is undefined. Otherwise, returns
  /// the same as `length` member of a `TF_FileStatistics` that would be used on
  /// the equivalent call of `stat`.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if `path` exists.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to a
  ///     filesystem entry.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if `path` is invalid or
  ///     points to a directory.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// DEFAULT IMPLEMENTATION: Gets statistics about `path`. Needs `stat`.
  int64_t (*get_file_size)(const TF_Filesystem* filesystem, const char* path,
                           TF_Status* status);

  /// Translates `uri` to a filename for the filesystem
  ///
  /// A filesystem is registered for a specific scheme and all of the methods
  /// should work with URIs. Hence, each filesystem needs to be able to
  /// translate from an URI to a path on the filesystem. For example, this
  /// function could translate `fs:///path/to/a/file` into `/path/to/a/file`, if
  /// implemented by a filesystem registered to handle the `fs://` scheme.
  ///
  /// A new `char*` buffer must be allocated by this method. Core TensorFlow
  /// manages the lifetime of the buffer after the call. Thus, all callers of
  /// this method must take ownership of the returned pointer.
  ///
  /// The implementation should clean up paths, including but not limited to,
  /// removing duplicate `/`s, and resolving `..` and `.`.
  ///
  /// Plugins must not return `nullptr`. Returning empty strings is allowed.
  ///
  /// This function will be called by core TensorFlow to clean up all path
  /// arguments for all other methods in the filesystem API.
  ///
  /// DEFAULT IMPLEMENTATION: Uses `io::CleanPath` and `io::ParseURI`.
  char* (*translate_name)(const TF_Filesystem* filesystem, const char* uri);

  /// Finds all entries in the directory given by `path`.
  ///
  /// The returned entries are paths relative to `path`.
  ///
  /// Plugins must allocate `entries` to hold all names that need to be returned
  /// and return the size of `entries`. Caller takes ownership of `entries`
  /// after the call.
  ///
  /// In case of error, plugins must set `status` to a value different than
  /// `TF_OK`, free memory allocated for `entries` and return -1.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if all children were returned.
  ///   * Must set `status` to `TF_NOT_FOUND` if `path` doesn't point to a
  ///     filesystem entry or if one of the parents entries in `path` doesn't
  ///     exist.
  ///   * Must set `status` to `TF_FAILED_PRECONDITION` if one of the parent
  ///     entries in `path` is not a directory, or if `path` is a file.
  ///   * Might use any other error value for `status` to signal other errors.
  int (*get_children)(const TF_Filesystem* filesystem, const char* path,
                      char*** entries, TF_Status* status);

  /// Finds all entries matching the regular expression given by `glob`.
  ///
  /// Pattern must match the entire entry name, not just a substring.
  ///
  /// pattern: { term }
  /// term:
  ///   '*': matches any sequence of non-'/' characters
  ///   '?': matches a single non-'/' character
  ///   '[' [ '^' ] { match-list } ']':
  ///        matches any single character (not) on the list
  ///   c: matches character c (c != '*', '?', '\\', '[')
  ///   '\\' c: matches character c
  /// character-range:
  ///   c: matches character c (c != '\\', '-', ']')
  ///   '\\' c: matches character c
  ///   lo '-' hi: matches character c for lo <= c <= hi
  ///
  /// Implementations must allocate `entries` to hold all names that need to be
  /// returned and return the size of `entries`. Caller takes ownership of
  /// `entries` after the call.
  ///
  /// In case of error, the implementations must set `status` to a value
  /// different than `TF_OK`, free any memory that might have been allocated for
  /// `entries` and return -1.
  ///
  /// Plugins:
  ///   * Must set `status` to `TF_OK` if all matches were returned.
  ///   * Might use any other error value for `status` to signal other errors.
  ///
  /// DEFAULT IMPLEMENTATION: Scans the directory tree (in parallel if possible)
  /// and fills `*entries`. Needs `get_children` and `is_directory`.
  int (*get_matching_paths)(const TF_Filesystem* filesystem, const char* glob,
                            char*** entries, TF_Status* status);

  /// Flushes any filesystem cache currently in memory
  ///
  /// DEFAULT IMPLEMENTATION: No op.
  void (*flush_caches)(const TF_Filesystem* filesystem);
} TF_FilesystemOps;
// LINT.ThenChange(:filesystem_ops_version)

/// SECTION 3. ABI and API compatibility
/// ----------------------------------------------------------------------------
///
/// In this section we define constants and macros to record versioning
/// information for each of the structures in section 2: ABI and API versions
/// and the number of functions in each of the function tables (which is
/// automatically determined, so ignored for the rest of this comment).
///
/// Since filesystem plugins are outside of TensorFlow's code tree, they are not
/// tied with TensorFlow releases and should have their own versioning metadata
/// in addition with the data discussed in this section. Each plugin author can
/// use a custom scheme, but it should only relate to changes in plugin code.
/// This section only touches metadata related to the versioning of this
/// interface that is shared by all possible plugins.
///
/// The API number increases whenever we break API compatibility while still
/// maintaining ABI compatibility. This happens only in the following cases:
///   1. A new method is added _at the end_ of the function table.
///   2. Preconditions or postconditions for one operation in these function
///   table change. Note that only core TensorFlow is able to impose these
///   invariants (i.e., guarantee the preconditions before calling the operation
///   and check the postconditions after the operation returns). If plugins need
///   additional invariants, they should be checked on the plugin side and the
///   `status` out variable should be updated accordingly (e.g., to include
///   plugin version information that relates to the condition change).
///
/// All other changes to the data structures (e.g., method removal, method
/// reordering, argument reordering, adding or removing arguments, changing the
/// type or the constness of a parameter, etc.) results in an ABI breakage.
/// Thus, we should not do any of these types of changes, except, potentially,
/// when we are releasing a new major version of TensorFlow. This is an escape
/// hatch, to be used rarely, preferably only to cleanup these structures.
/// Whenever we do these changes, the ABI number must be increased.
///
/// Next section will detail how this metadata is used at plugin registration to
/// only load compatible plugins and discard all others.

// LINT.IfChange(random_access_file_ops_version)
constexpr int TF_RANDOM_ACCESS_FILE_OPS_API = 0;
constexpr int TF_RANDOM_ACCESS_FILE_OPS_ABI = 0;
constexpr size_t TF_RANDOM_ACCESS_FILE_OPS_SIZE =
    sizeof(TF_RandomAccessFileOps);
// LINT.ThenChange()

// LINT.IfChange(writable_file_ops_version)
constexpr int TF_WRITABLE_FILE_OPS_API = 0;
constexpr int TF_WRITABLE_FILE_OPS_ABI = 0;
constexpr size_t TF_WRITABLE_FILE_OPS_SIZE = sizeof(TF_WritableFileOps);
// LINT.ThenChange()

// LINT.IfChange(read_only_memory_region_ops_version)
constexpr int TF_READ_ONLY_MEMORY_REGION_OPS_API = 0;
constexpr int TF_READ_ONLY_MEMORY_REGION_OPS_ABI = 0;
constexpr size_t TF_READ_ONLY_MEMORY_REGION_OPS_SIZE =
    sizeof(TF_ReadOnlyMemoryRegionOps);
// LINT.ThenChange()

// LINT.IfChange(filesystem_ops_version)
constexpr int TF_FILESYSTEM_OPS_API = 0;
constexpr int TF_FILESYSTEM_OPS_ABI = 0;
constexpr size_t TF_FILESYSTEM_OPS_SIZE = sizeof(TF_FilesystemOps);
// LINT.ThenChange()

/// SECTION 4. Plugin registration and initialization
/// ----------------------------------------------------------------------------
///
/// In this section we define the API used by core TensorFlow to initialize a
/// filesystem provided by a plugin. That is, we define the following:
///   * `TF_InitPlugin` function: must be present in the plugin shared object as
///     it will be called by core TensorFlow when the filesystem plugin is
///     loaded;
///   * `TF_FilesystemPluginInfo` struct: used to transfer information between
///     plugins and core TensorFlow about the operations provided and metadata;
///   * `TF_SetFilesystemVersionMetadata` function: must be called by plugins in
///     their `TF_InitPlugin` to record the versioning information the plugins
///     are compiled against.
///
/// The `TF_InitPlugin` function is used by plugins to set up the data
/// structures that implement this interface, as presented in Section 2. In
/// order to not have plugin shared objects call back symbols defined in core
/// TensorFlow, `TF_InitPlugin` has a `TF_FilesystemPluginInfo` argument which
/// the plugin must fill (using the `TF_SetFilesystemVersionMetadata` for the
/// metadata and setting up all the supported operations and the URI schemes
/// that are supported).

/// This structure incorporates the operations defined in Section 2 and the
/// metadata defined in section 3, allowing plugins to define different ops
/// for different URI schemes.
///
/// Every URI scheme is of the form "fs" for URIs of form "fs:///path/to/file".
/// For local filesystems (i.e., when the URI is "/path/to/file"), the scheme
/// must be "". The scheme must never be `nullptr`.
///
/// Every plugin fills this in `TF_InitPlugin`, using the alocator passed as
/// argument to allocate memory. After `TF_InitPlugin` finishes, core
/// TensorFlow uses the information present in this to initialize filesystems
/// for the URI schemes that the plugin requests.
///
/// All pointers defined in this structure point to memory allocated by the DSO
/// using an allocator provided by core TensorFlow when calling `TF_InitPlugin`.
///
/// IMPORTANT: To maintain binary compatibility, the layout of this structure
/// must not change! In the unlikely case that a new type of file needs to be
/// supported, add the new ops and metadata at the end of the structure.
typedef struct TF_FilesystemPluginInfo {
  char* scheme;
  int filesystem_ops_abi;
  int filesystem_ops_api;
  size_t filesystem_ops_size;
  TF_FilesystemOps* filesystem_ops;
  int random_access_file_ops_abi;
  int random_access_file_ops_api;
  size_t random_access_file_ops_size;
  TF_RandomAccessFileOps* random_access_file_ops;
  int writable_file_ops_abi;
  int writable_file_ops_api;
  size_t writable_file_ops_size;
  TF_WritableFileOps* writable_file_ops;
  int read_only_memory_region_ops_abi;
  int read_only_memory_region_ops_api;
  size_t read_only_memory_region_ops_size;
  TF_ReadOnlyMemoryRegionOps* read_only_memory_region_ops;
} TF_FilesystemPluginInfo;

/// Convenience function for setting the versioning metadata.
///
/// The argument is guaranteed to not be `nullptr`.
///
/// We want this to be defined in the plugin's memory space and we guarantee
/// that core TensorFlow will never call this.
static inline void TF_SetFilesystemVersionMetadata(
    TF_FilesystemPluginInfo* info) {
  info->filesystem_ops_abi = TF_FILESYSTEM_OPS_ABI;
  info->filesystem_ops_api = TF_FILESYSTEM_OPS_API;
  info->filesystem_ops_size = TF_FILESYSTEM_OPS_SIZE;
  info->random_access_file_ops_abi = TF_RANDOM_ACCESS_FILE_OPS_ABI;
  info->random_access_file_ops_api = TF_RANDOM_ACCESS_FILE_OPS_API;
  info->random_access_file_ops_size = TF_RANDOM_ACCESS_FILE_OPS_SIZE;
  info->writable_file_ops_abi = TF_WRITABLE_FILE_OPS_ABI;
  info->writable_file_ops_api = TF_WRITABLE_FILE_OPS_API;
  info->writable_file_ops_size = TF_WRITABLE_FILE_OPS_SIZE;
  info->read_only_memory_region_ops_abi = TF_READ_ONLY_MEMORY_REGION_OPS_ABI;
  info->read_only_memory_region_ops_api = TF_READ_ONLY_MEMORY_REGION_OPS_API;
  info->read_only_memory_region_ops_size = TF_READ_ONLY_MEMORY_REGION_OPS_SIZE;
}

/// Initializes a TensorFlow plugin.
///
/// Must be implemented by the plugin DSO. It is called by TensorFlow runtime.
///
/// Filesystem plugins can be loaded on demand by users via
/// `Env::LoadLibrary` or during TensorFlow's startup if they are on certain
/// paths (although this has a security risk if two plugins register for the
/// same filesystem and the malicious one loads before the legimitate one -
/// but we consider this to be something that users should care about and
/// manage themselves). In both of these cases, core TensorFlow looks for
/// the `TF_InitPlugin` symbol and calls this function.
///
/// All memory allocated by this function must be allocated via the `allocator`
/// argument.
///
/// For every filesystem URI scheme that this plugin supports, the plugin must
/// add one `TF_FilesystemPluginInfo` entry in `plugin_info`.
///
/// Returns number of entries in `plugin_info` (i.e., number of URI schemes
/// supported).
TF_CAPI_EXPORT extern int TF_InitPlugin(void* (*allocator)(size_t size),
                                        TF_FilesystemPluginInfo** plugin_info);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_FILESYSTEM_INTERFACE_H_
