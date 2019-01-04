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

// TODO(jhen): Replace hardcoded, platform specific path strings in GetXXXPath()
// with a function in e.g. cuda.h.

#include <limits.h>
#include <stdlib.h>
#include <initializer_list>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/default/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

#if !defined(PLATFORM_GOOGLE)
#include "absl/strings/string_view.h"
#include "cuda/cuda_config.h"
#endif

namespace stream_executor {
namespace internal {

string GetCudaVersion() { return TF_CUDA_VERSION; }
string GetCudnnVersion() { return TF_CUDNN_VERSION; }

/* static */ port::Status DsoLoader::GetCublasDsoHandle(void** dso_handle) {
  return GetDsoHandle(FindDsoPath(port::Env::Default()->FormatLibraryFileName(
                                      "cublas", GetCudaVersion()),
                                  GetCudaLibraryDirPath()),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetCudnnDsoHandle(void** dso_handle) {
  // libcudnn is versioned differently than the other libraries and may have a
  // different version number than other CUDA libraries.  See b/22397368 for
  // some details about the complications surrounding this.
  return GetDsoHandle(FindDsoPath(port::Env::Default()->FormatLibraryFileName(
                                      "cudnn", GetCudnnVersion()),
                                  GetCudaLibraryDirPath()),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetCufftDsoHandle(void** dso_handle) {
  return GetDsoHandle(FindDsoPath(port::Env::Default()->FormatLibraryFileName(
                                      "cufft", GetCudaVersion()),
                                  GetCudaLibraryDirPath()),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetCurandDsoHandle(void** dso_handle) {
  return GetDsoHandle(FindDsoPath(port::Env::Default()->FormatLibraryFileName(
                                      "curand", GetCudaVersion()),
                                  GetCudaLibraryDirPath()),
                      dso_handle);
}

/* static */ port::Status DsoLoader::GetLibcudaDsoHandle(void** dso_handle) {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle(
      FindDsoPath(port::Env::Default()->FormatLibraryFileName("nvcuda", ""),
                  GetCudaDriverLibraryPath()),
      dso_handle);
#else
  port::Status status = GetDsoHandle(
      FindDsoPath(port::Env::Default()->FormatLibraryFileName("cuda", "1"),
                  GetCudaDriverLibraryPath()),
      dso_handle);
#if defined(__APPLE__)
  // On Mac OS X, CUDA sometimes installs libcuda.dylib instead of
  // libcuda.1.dylib.
  return status.ok()
             ? status
             : GetDsoHandle(
                   FindDsoPath(
                       port::Env::Default()->FormatLibraryFileName("cuda", ""),
                       GetCudaDriverLibraryPath()),
                   dso_handle);
#else
  return status;
#endif
#endif
}

/* static */ port::Status DsoLoader::GetLibcuptiDsoHandle(void** dso_handle) {
#if defined(ANDROID_TEGRA)
  // On Android devices the CUDA version number is not added to the library
  // name.
  return GetDsoHandle(
      FindDsoPath(port::Env::Default()->FormatLibraryFileName("cupti", ""),
                  GetCudaCuptiLibraryPath()),
      dso_handle);
#else
  return GetDsoHandle(FindDsoPath(port::Env::Default()->FormatLibraryFileName(
                                      "cupti", GetCudaVersion()),
                                  GetCudaCuptiLibraryPath()),
                      dso_handle);
#endif
}

static mutex& GetRpathMutex() {
  static mutex* mu = new mutex;
  return *mu;
}

/* static */ void DsoLoader::RegisterRpath(absl::string_view path) {
  mutex_lock lock{GetRpathMutex()};
  GetRpaths()->emplace_back(path);
}

/* static */ port::Status DsoLoader::GetDsoHandle(absl::string_view path,
                                                  void** dso_handle,
                                                  LoadKind load_kind) {
  if (load_kind != LoadKind::kLocal) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "Only LoadKind::kLocal is currently supported");
  }
  string path_string(path);
  port::Status s =
      port::Env::Default()->LoadLibrary(path_string.c_str(), dso_handle);
  if (!s.ok()) {
#if !defined(PLATFORM_WINDOWS)
    char* ld_library_path = getenv("LD_LIBRARY_PATH");
#endif
    LOG(INFO) << "Couldn't open CUDA library " << path
#if !defined(PLATFORM_WINDOWS)
              << ". LD_LIBRARY_PATH: "
              << (ld_library_path != nullptr ? ld_library_path : "")
#endif
        ;
    return port::Status(port::error::FAILED_PRECONDITION,
                        absl::StrCat("could not dlopen DSO: ", path,
                                     "; dlerror: ", s.error_message()));
  }
  LOG(INFO) << "successfully opened CUDA library " << path << " locally";
  return port::Status::OK();
}

/* static */ string DsoLoader::GetBinaryDirectory(bool strip_executable_name) {
  string exe_path = port::Env::Default()->GetExecutablePath();
  return strip_executable_name ? string(port::Dirname(exe_path)) : exe_path;
}

// Creates a heap-allocated vector for initial rpaths.
// Ownership is transferred to the caller.
static std::vector<string>* CreatePrimordialRpaths() {
  auto rpaths = new std::vector<string>;
#if defined(__APPLE__)
  rpaths->push_back("driver/driver_sh.runfiles/local_config_cuda/cuda/lib");
#else
  rpaths->push_back("driver/driver_sh.runfiles/local_config_cuda/cuda/lib64");
#endif
  return rpaths;
}

/* static */ std::vector<string>* DsoLoader::GetRpaths() {
  static std::vector<string>* rpaths = CreatePrimordialRpaths();
  return rpaths;
}

/* static */ bool DsoLoader::TrySymbolicDereference(string* candidate) {
#if defined(PLATFORM_WINDOWS)
  return false;
#else
  char buf[PATH_MAX];
  char* result = realpath(candidate->c_str(), buf);
  if (result == nullptr) {
    return false;
  }
  VLOG(3) << "realpath resolved candidate path \"" << *candidate << "\" to \""
          << result << "\"";
  *candidate = result;
  return true;
#endif
}

/* static */ string DsoLoader::FindDsoPath(absl::string_view library_name,
                                           absl::string_view runfiles_relpath) {
  // Keep a record of the paths we attempted so we can dump out meaningful
  // diagnostics if no path is found.
  std::vector<string> attempted;

  using StringPieces = std::vector<absl::string_view>;
  string candidate;

  // Otherwise, try binary-plus-rpath locations.
  string binary_directory =
      GetBinaryDirectory(true /* = strip_executable_name */);
  mutex_lock lock{GetRpathMutex()};
  for (const string& rpath : *GetRpaths()) {
    candidate =
        port::Join(StringPieces{binary_directory, rpath, library_name}, "/");
    if (TrySymbolicDereference(&candidate)) {
      return candidate;
    }
  }
  attempted.push_back(candidate);

  return string(library_name);
}

/* static */ string DsoLoader::GetCudaLibraryDirPath() {
#if defined(__APPLE__)
  return "external/local_config_cuda/cuda/lib";
#else
  return "external/local_config_cuda/cuda/lib64";
#endif
}

/* static */ string DsoLoader::GetCudaDriverLibraryPath() {
#if defined(__APPLE__)
  return "external/local_config_cuda/cuda/driver/lib";
#elif defined(PLATFORM_WINDOWS)
  return "";
#else
  return "external/local_config_cuda/cuda/driver/lib64";
#endif
}

/* static */ string DsoLoader::GetCudaCuptiLibraryPath() {
#if defined(__APPLE__)
  return "external/local_config_cuda/cuda/extras/CUPTI/lib";
#else
  return "external/local_config_cuda/cuda/extras/CUPTI/lib64";
#endif
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
}  // namespace stream_executor
