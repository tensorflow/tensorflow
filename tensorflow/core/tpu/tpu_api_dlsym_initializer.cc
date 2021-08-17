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

#include "tensorflow/core/tpu/tpu_api_dlsym_initializer.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_api_dlsym_set_fn.h"

#if !defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_initializer_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#endif


// Reminder: Update tpu_library_loader_windows.cc if you are adding new publicly
// visible methods.

namespace tensorflow {
namespace tpu {
namespace {
#if defined(PLATFORM_GOOGLE)
Status InitializeTpuLibrary(void* library_handle) {
  return errors::Unimplemented("You must statically link in a TPU library.");
}
#else  // PLATFORM_GOOGLE
#include "tensorflow/core/tpu/tpu_library_init_fns.inc"

Status InitializeTpuLibrary(void* library_handle) {
  Status s = InitializeTpuStructFns(library_handle);

  // Retrieve arguments from environment if applicable
  std::pair<std::vector<std::string>, std::vector<const char*> > args =
      GetLibTpuInitArguments();

  // TPU platform registration must only be performed after the library is
  // loaded. We do not want to register a TPU platform in XLA without the
  // supporting library providing the necessary APIs.
  if (s.ok()) {
    void (*initialize_fn)(bool init_library, int num_args, const char** args);
    initialize_fn = reinterpret_cast<decltype(initialize_fn)>(
        dlsym(library_handle, "TfTpu_Initialize"));
    (*initialize_fn)(/*init_library=*/true, args.second.size(),
                     args.second.data());

    RegisterTpuPlatform();
  }

  return s;
}

void* CreateGcsFilesystemFn() {
  return new tensorflow::RetryingGcsFileSystem();
}

// This is a temporary fix for including GCS file system on TPU builds.
// Will be removed once b/176954917 is fully resolved with the build fix.
void InitializeCreateGcsFileSystemFnPtr() {
  int fd = shm_open(absl::StrCat("/tmp_tf_gcs_fs_pointer_", getpid()).data(),
                    O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    LOG(ERROR) << "Unable to open shared memory for GCS file system creator.";
    return;
  }

  if (ftruncate(fd, sizeof(tensorflow::FileSystem*)) == -1) {
    LOG(ERROR)
        << "Unable to allocate shared memory for GCS file system creator.";
    return;
  }

  void* (**fn)() = reinterpret_cast<void* (**)()>(mmap(
      NULL, sizeof(void* (*)()), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (fn == MAP_FAILED) {
    LOG(ERROR) << "Cannot mmap shared memory for GCS file system creator.";
    return;
  }

  *fn = &CreateGcsFilesystemFn;

  munmap(fn, sizeof(void* (*)()));
  close(fd);

  // Clean up shared memory on a clean exit.
  atexit([]() {
    shm_unlink(absl::StrCat("/tmp_tf_gcs_fs_pointer_", getpid()).data());
  });
}

bool FindAndLoadTpuLibrary() {
  const char* env_value = getenv("TPU_LIBRARY_PATH");
  const char* libtpu_path =
      env_value && strlen(env_value) > 0 ? env_value : "libtpu.so";
  LOG(INFO) << "Libtpu path is: " << libtpu_path;
  void* library = dlopen(libtpu_path, RTLD_NOW);
  if (library) {
    // We can open the shared library which means we are in a TPU environment.
    // Try to acquire exclusive access.
    if (TryAcquireTpuLock()) {
      InitializeTpuLibrary(library);
    }
  }

  InitializeCreateGcsFileSystemFnPtr();
  return true;
}

static bool tpu_library_finder = FindAndLoadTpuLibrary();
#endif  // PLATFORM_GOOGLE
}  // namespace
}  // namespace tpu
}  // namespace tensorflow
