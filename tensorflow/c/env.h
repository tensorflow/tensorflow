/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_ENV_H_
#define TENSORFLOW_C_ENV_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_file_statistics.h"
#include "tensorflow/c/tf_status.h"

// --------------------------------------------------------------------------
// C API for tensorflow::Env.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_WritableFileHandle TF_WritableFileHandle;
typedef struct TF_StringStream TF_StringStream;
typedef struct TF_Thread TF_Thread;

typedef struct TF_ThreadOptions {
  // Thread stack size to use (in bytes), zero implies that the system default
  // will be used.
  size_t stack_size;

  // Guard area size to use near thread stacks to use (in bytes), zero implies
  // that the system default will be used.
  size_t guard_size;

  // The NUMA node to use, -1 implies that there should be no NUMA affinity for
  // this thread.
  int numa_node;
} TF_ThreadOptions;

// Creates the specified directory. Typical status code are:
//  * TF_OK - successfully created the directory
//  * TF_ALREADY_EXISTS - directory already exists
//  * TF_PERMISSION_DENIED - dirname is not writable
TF_CAPI_EXPORT extern void TF_CreateDir(const char* dirname, TF_Status* status);

// Deletes the specified directory. Typical status codes are:
//  * TF_OK - successfully deleted the directory
//  * TF_FAILED_PRECONDITION - the directory is not empty
TF_CAPI_EXPORT extern void TF_DeleteDir(const char* dirname, TF_Status* status);

// Deletes the specified directory and all subdirectories and files underneath
// it. This is accomplished by traversing the directory tree rooted at dirname
// and deleting entries as they are encountered.
//
// If dirname itself is not readable or does not exist, *undeleted_dir_count is
// set to 1, *undeleted_file_count is set to 0 and an appropriate status (e.g.
// TF_NOT_FOUND) is returned.
//
// If dirname and all its descendants were successfully deleted, TF_OK is
// returned and both error counters are set to zero.
//
// Otherwise, while traversing the tree, undeleted_file_count and
// undeleted_dir_count are updated if an entry of the corresponding type could
// not be deleted. The returned error status represents the reason that any one
// of these entries could not be deleted.
//
// Typical status codes:
//  * TF_OK - dirname exists and we were able to delete everything underneath
//  * TF_NOT_FOUND - dirname doesn't exist
//  * TF_PERMISSION_DENIED - dirname or some descendant is not writable
//  * TF_UNIMPLEMENTED - some underlying functions (like Delete) are not
//    implemented
TF_CAPI_EXPORT extern void TF_DeleteRecursively(const char* dirname,
                                                int64_t* undeleted_file_count,
                                                int64_t* undeleted_dir_count,
                                                TF_Status* status);

// Obtains statistics for the given path. If status is TF_OK, *stats is
// updated, otherwise it is not touched.
TF_CAPI_EXPORT extern void TF_FileStat(const char* filename,
                                       TF_FileStatistics* stats,
                                       TF_Status* status);

// Creates or truncates the given filename and returns a handle to be used for
// appending data to the file. If status is TF_OK, *handle is updated and the
// caller is responsible for freeing it (see TF_CloseWritableFile).
TF_CAPI_EXPORT extern void TF_NewWritableFile(const char* filename,
                                              TF_WritableFileHandle** handle,
                                              TF_Status* status);

// Closes the given handle and frees its memory. If there was a problem closing
// the file, it is indicated by status. Memory is freed in any case.
TF_CAPI_EXPORT extern void TF_CloseWritableFile(TF_WritableFileHandle* handle,
                                                TF_Status* status);

// Syncs content of the handle to the filesystem. Blocks waiting for the
// filesystem to indicate that the content has been persisted.
TF_CAPI_EXPORT extern void TF_SyncWritableFile(TF_WritableFileHandle* handle,
                                               TF_Status* status);

// Flush local buffers to the filesystem. If the process terminates after a
// successful flush, the contents may still be persisted, since the underlying
// filesystem may eventually flush the contents.  If the OS or machine crashes
// after a successful flush, the contents may or may not be persisted, depending
// on the implementation.
TF_CAPI_EXPORT extern void TF_FlushWritableFile(TF_WritableFileHandle* handle,
                                                TF_Status* status);

// Appends the given bytes to the file. Any failure to do so is indicated in
// status.
TF_CAPI_EXPORT extern void TF_AppendWritableFile(TF_WritableFileHandle* handle,
                                                 const char* data,
                                                 size_t length,
                                                 TF_Status* status);

// Deletes the named file and indicates whether successful in *status.
TF_CAPI_EXPORT extern void TF_DeleteFile(const char* filename,
                                         TF_Status* status);

// Retrieves the next item from the given TF_StringStream and places a pointer
// to it in *result. If no more items are in the list, *result is set to NULL
// and false is returned.
//
// Ownership of the items retrieved with this function remains with the library.
// Item points are invalidated after a call to TF_StringStreamDone.
TF_CAPI_EXPORT extern bool TF_StringStreamNext(TF_StringStream* list,
                                               const char** result);

// Frees the resources associated with given string list. All pointers returned
// by TF_StringStreamNext are invalid after this call.
TF_CAPI_EXPORT extern void TF_StringStreamDone(TF_StringStream* list);

// Retrieves the list of children of the given directory. You can iterate
// through the list with TF_StringStreamNext. The caller is responsible for
// freeing the list (see TF_StringStreamDone).
TF_CAPI_EXPORT extern TF_StringStream* TF_GetChildren(const char* filename,
                                                      TF_Status* status);

// Retrieves a list of directory names on the local machine that may be used for
// temporary storage. You can iterate through the list with TF_StringStreamNext.
// The caller is responsible for freeing the list (see TF_StringStreamDone).
TF_CAPI_EXPORT extern TF_StringStream* TF_GetLocalTempDirectories(void);

// Creates a temporary file name with an extension.
// The caller is responsible for freeing the returned pointer.
TF_CAPI_EXPORT extern char* TF_GetTempFileName(const char* extension);

// Returns the number of nanoseconds since the Unix epoch.
TF_CAPI_EXPORT extern uint64_t TF_NowNanos(void);

// Returns the number of microseconds since the Unix epoch.
TF_CAPI_EXPORT extern uint64_t TF_NowMicros(void);

// Returns the number of seconds since the Unix epoch.
TF_CAPI_EXPORT extern uint64_t TF_NowSeconds(void);

// Populates a TF_ThreadOptions struct with system-default values.
TF_CAPI_EXPORT extern void TF_DefaultThreadOptions(TF_ThreadOptions* options);

// Returns a new thread that is running work_func and is identified
// (for debugging/performance-analysis) by thread_name.
//
// The given param (which may be null) is passed to work_func when the thread
// starts. In this way, data may be passed from the thread back to the caller.
//
// Caller takes ownership of the result and must call TF_JoinThread on it
// eventually.
TF_CAPI_EXPORT extern TF_Thread* TF_StartThread(const TF_ThreadOptions* options,
                                                const char* thread_name,
                                                void (*work_func)(void*),
                                                void* param);

// Waits for the given thread to finish execution, then deletes it.
TF_CAPI_EXPORT extern void TF_JoinThread(TF_Thread* thread);

// \brief Load a dynamic library.
//
// Pass "library_filename" to a platform-specific mechanism for dynamically
// loading a library. The rules for determining the exact location of the
// library are platform-specific and are not documented here.
//
// On success, place OK in status and return the newly created library handle.
// Otherwise returns nullptr and set error status.
TF_CAPI_EXPORT extern void* TF_LoadSharedLibrary(const char* library_filename,
                                                 TF_Status* status);

// \brief Get a pointer to a symbol from a dynamic library.
//
// "handle" should be a pointer returned from a previous call to
// TF_LoadLibraryFromEnv. On success, place OK in status and return a pointer to
// the located symbol. Otherwise returns nullptr and set error status.
TF_CAPI_EXPORT extern void* TF_GetSymbolFromLibrary(void* handle,
                                                    const char* symbol_name,
                                                    TF_Status* status);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_C_ENV_H_
