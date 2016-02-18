/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/stream_executor/dso_loader.h"

#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <initializer_list>
#include "tensorflow/stream_executor/platform/port.h"
#include <vector>

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/lib/str_util.h"

namespace perftools {
namespace gputools {
namespace internal {

// TensorFlow OSS configure uses the following lines to configure versions. For
// any modifications of the format, please make sure the script still works.
string GetCudaVersion() { return ""; }
string GetCudnnVersion() { return ""; }

/* static */ port::Status DsoLoader::GetCublasDsoHandle(void** dso_handle) {
  return GetDsoHandle(FindDsoPath("libcublas.so" + GetCudaVersion(),
                                  "third_party/gpus/cuda/lib64"),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetCudnnDsoHandle(void** dso_handle) {
  // libcudnn is versioned differently than the other libraries and may have a
  // different version number than other CUDA libraries.  See b/22397368 for
  // some details about the complications surrounding this.
  return GetDsoHandle(
      FindDsoPath("libcudnn.so" + GetCudnnVersion(),
                  "third_party/gpus/cuda/lib64"),
      dso_handle);
}

/* static */ port::Status DsoLoader::GetCufftDsoHandle(void** dso_handle) {
  return GetDsoHandle(FindDsoPath("libcufft.so" + GetCudaVersion(),
                                  "third_party/gpus/cuda/lib64"),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetCurandDsoHandle(void** dso_handle) {
  return GetDsoHandle(FindDsoPath("libcurand.so" + GetCudaVersion(),
                                  "third_party/gpus/cuda/lib64"),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetLibcudaDsoHandle(void** dso_handle) {
  return GetDsoHandle(
      FindDsoPath("libcuda.so.1", "third_party/gpus/cuda/driver/lib64"),
      dso_handle);
}

/* static */ port::Status DsoLoader::GetLibcuptiDsoHandle(void** dso_handle) {
  return GetDsoHandle(
      FindDsoPath("libcupti.so" + GetCudaVersion(),
                  "third_party/gpus/cuda/extras/CUPTI/lib64"),
      dso_handle);
}

/* static */ void DsoLoader::RegisterRpath(port::StringPiece path) {
  mutex_lock lock{rpath_mutex_};
  GetRpaths()->push_back(path.ToString());
}


/* static */ port::Status DsoLoader::GetDsoHandle(port::StringPiece path,
                                                  void** dso_handle,
                                                  LoadKind load_kind) {

  int dynload_flags =
      RTLD_LAZY | (load_kind == LoadKind::kLocal ? RTLD_LOCAL : RTLD_GLOBAL);
  string path_string = path.ToString();
  *dso_handle = dlopen(path_string.c_str(), dynload_flags);
  if (*dso_handle == nullptr) {
    LOG(INFO) << "Couldn't open CUDA library " << path
              << ". LD_LIBRARY_PATH: " << getenv("LD_LIBRARY_PATH");
    return port::Status(
        port::error::FAILED_PRECONDITION,
        port::StrCat("could not dlopen DSO: ", path, "; dlerror: ", dlerror()));
  }
  LOG(INFO) << "successfully opened CUDA library " << path
            << (load_kind == LoadKind::kLocal ? " locally" : " globally");
  return port::Status::OK();
}

/* static */ string DsoLoader::GetBinaryDirectory(bool strip_executable_name) {
  char exe_path[PATH_MAX] = {0};
  CHECK_ERR(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  if (strip_executable_name) {
    // The exe is the last component of the path, so remove one component.
    std::vector<string> components = port::Split(exe_path, '/');
    components.pop_back();
    return port::Join(components, "/");
  }
  return exe_path;
}

// Creates a heap-allocated vector for initial rpaths.
// Ownership is transferred to the caller.
static std::vector<string>* CreatePrimordialRpaths() {
  auto rpaths = new std::vector<string>;
  rpaths->push_back(
      "driver/driver_sh.runfiles/third_party/gpus/cuda/lib64");
  return rpaths;
}

/* static */ mutex DsoLoader::rpath_mutex_{LINKER_INITIALIZED};
/* static */ std::vector<string>* DsoLoader::GetRpaths() {
  static std::vector<string>* rpaths = CreatePrimordialRpaths();
  return rpaths;
}

/* static */ bool DsoLoader::TrySymbolicDereference(string* candidate) {
  char buf[PATH_MAX];
  char* result = realpath(candidate->c_str(), buf);
  if (result == nullptr) {
    return false;
  }
  VLOG(3) << "realpath resolved candidate path \"" << *candidate << "\" to \""
          << result << "\"";
  *candidate = result;
  return true;
}

/* static */ string DsoLoader::FindDsoPath(port::StringPiece library_name,
                                           port::StringPiece runfiles_relpath) {

  // Keep a record of the paths we attempted so we can dump out meaningful
  // diagnostics if no path is found.
  std::vector<string> attempted;

  using StringPieces = std::vector<port::StringPiece>;
  string candidate;

  // Otherwise, try binary-plus-rpath locations.
  string binary_directory =
      GetBinaryDirectory(true /* = strip_executable_name */);
  mutex_lock lock{rpath_mutex_};
  for (const string& rpath : *GetRpaths()) {
    candidate =
        port::Join(StringPieces{binary_directory, rpath, library_name}, "/");
    if (TrySymbolicDereference(&candidate)) {
      return candidate;
    }
  }
  attempted.push_back(candidate);

  return library_name.ToString();
}

// -- CachedDsoLoader

/* static */ port::StatusOr<void*> CachedDsoLoader::GetCublasDsoHandle() {
  static port::StatusOr<void*> result =
      FetchHandleResult(DsoLoader::GetCublasDsoHandle);
  return result;
}

/* static */ port::StatusOr<void*> CachedDsoLoader::GetCurandDsoHandle() {
  static port::StatusOr<void*> result =
      FetchHandleResult(DsoLoader::GetCurandDsoHandle);
  return result;
}

/* static */ port::StatusOr<void*> CachedDsoLoader::GetCudnnDsoHandle() {
  static port::StatusOr<void*> result =
      FetchHandleResult(DsoLoader::GetCudnnDsoHandle);
  return result;
}

/* static */ port::StatusOr<void*> CachedDsoLoader::GetCufftDsoHandle() {
  static port::StatusOr<void*> result =
      FetchHandleResult(DsoLoader::GetCufftDsoHandle);
  return result;
}

/* static */ port::StatusOr<void*> CachedDsoLoader::GetLibcudaDsoHandle() {
  static port::StatusOr<void*> result =
      FetchHandleResult(DsoLoader::GetLibcudaDsoHandle);
  return result;
}

/* static */ port::StatusOr<void*> CachedDsoLoader::GetLibcuptiDsoHandle() {
  static port::StatusOr<void*> result =
      FetchHandleResult(DsoLoader::GetLibcuptiDsoHandle);
  return result;
}

/* static */ port::StatusOr<void*> CachedDsoLoader::FetchHandleResult(
    std::function<port::Status(void**)> load_dso) {
  void* handle;
  auto status = load_dso(&handle);
  if (!status.ok()) {
    return status;
  }
  return handle;
}

}  // namespace internal
}  // namespace gputools
}  // namespace perftools
