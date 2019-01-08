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

// Common DSO loading functionality: exposes callables that dlopen DSOs
// in either the runfiles directories

#ifndef TENSORFLOW_STREAM_EXECUTOR_DSO_LOADER_H_
#define TENSORFLOW_STREAM_EXECUTOR_DSO_LOADER_H_

#include <vector>
#include "tensorflow/stream_executor/platform/port.h"

#include "absl/strings/string_view.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"

namespace stream_executor {
namespace internal {

// Permits StreamExecutor code to dynamically load a pre-determined set of
// relevant DSOs via dlopen.
//
// Thread-safe.
class DsoLoader {
 public:
  // The following methods either load the DSO of interest and return a dlopen
  // handle or error status in the canonical namespace.

  static port::Status GetCublasDsoHandle(void** dso_handle);
  static port::Status GetCudnnDsoHandle(void** dso_handle);
  static port::Status GetCufftDsoHandle(void** dso_handle);
  static port::Status GetCurandDsoHandle(void** dso_handle);
  static port::Status GetLibcudaDsoHandle(void** dso_handle);
  static port::Status GetLibcuptiDsoHandle(void** dso_handle);

  // Registers a new binary-relative path to use as a dlopen search path.
  static void RegisterRpath(absl::string_view path);

 private:
  // Registered rpaths (singleton vector) and a mutex that guards it.
  static std::vector<string>* GetRpaths();

  // Descriptive boolean wrapper to indicate whether symbols are made available
  // to resolve in later-loaded libraries.
  enum class LoadKind { kLocal, kGlobal };

  // Loads a DSO from the given "path" (which can technically be any dlopen-able
  // name). If the load kind is global, the symbols in the loaded DSO are
  // visible to subsequent DSO loading operations.
  static port::Status GetDsoHandle(absl::string_view path, void** dso_handle,
                                   LoadKind load_kind = LoadKind::kLocal);

  // Returns the binary directory (or binary path) associated with the currently
  // executing program. If strip_executable_name is true, the executable file is
  // stripped off of the path.
  static string GetBinaryDirectory(bool strip_executable_name);

  // Invokes realpath on the original path; updates candidate and returns true
  // if it succeeds (i.e. a file exists at the path); otherwise, returns false.
  static bool TrySymbolicDereference(string* candidate);

  // Attempts to find a path to the DSO of interest, otherwise returns the
  // bare library name:
  // Arguments:
  //   library_name: the filename in tree; e.g. libOpenCL.so.1.0.0
  //   runfiles_relpath: where to look for the library relative to the runfiles
  //      root; e.g. third_party/gpus/cuda/lib64
  static string FindDsoPath(absl::string_view library_name,
                            absl::string_view runfiles_relpath);

  // Return platform dependent paths for DSOs
  static string GetCudaLibraryDirPath();
  static string GetCudaDriverLibraryPath();
  static string GetCudaCuptiLibraryPath();

  SE_DISALLOW_COPY_AND_ASSIGN(DsoLoader);
};

// Wrapper around the DsoLoader that prevents us from dlopen'ing any of the DSOs
// more than once.
class CachedDsoLoader {
 public:
  // Cached versions of the corresponding DsoLoader methods above.
  static port::StatusOr<void*> GetCublasDsoHandle();
  static port::StatusOr<void*> GetCudnnDsoHandle();
  static port::StatusOr<void*> GetCufftDsoHandle();
  static port::StatusOr<void*> GetCurandDsoHandle();
  static port::StatusOr<void*> GetLibcudaDsoHandle();
  static port::StatusOr<void*> GetLibcuptiDsoHandle();

 private:
  // Fetches a DSO handle via "load_dso" and returns the StatusOr form of the
  // result.
  static port::StatusOr<void*> FetchHandleResult(
      std::function<port::Status(void**)> load_dso);

  SE_DISALLOW_COPY_AND_ASSIGN(CachedDsoLoader);
};

}  // namespace internal
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DSO_LOADER_H_
