/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_initializer_helper.h"

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/core/tpu/tpu_api_dlsym_set_fn.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

#if !defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#endif  // PLATFORM_GOOGLE

namespace tensorflow {
namespace tpu {
namespace {

static std::string GetEnvVar(const char* name) {
  // Constructing a std::string directly from nullptr is undefined behavior.
  return absl::StrCat(getenv(name));
}

bool GetEnvBool(const char* name, bool defval) {
  const char* env = getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  int int_env;
  bool has_int = absl::SimpleAtoi(env, &int_env);
  return has_int && int_env != 0;
}

}  // namespace

// This function gets pid of a process and checks if that process is using tpu.
// It is not able to check processes that are owned by another user.
bool IsTpuUsed(int64_t pid) {
  std::string path = absl::StrCat("/proc/", pid, "/fd");
  DIR* raw_fd_dir = opendir(path.c_str());
  if (!raw_fd_dir) {
    return false;
  }
  std::unique_ptr<DIR, int (*)(DIR*)> fd_dir(raw_fd_dir, closedir);
  struct dirent* ent;
  std::string line;
  std::string tpu_dev_path = "/dev/accel0";
  line.resize(tpu_dev_path.size());
  while ((ent = readdir(raw_fd_dir))) {
    if (!isdigit(*ent->d_name)) continue;
    int64_t fd = strtol(ent->d_name, nullptr, 10);
    path = absl::StrCat("/proc/", pid, "/fd/", fd);
    if (!readlink(path.c_str(), &line[0], line.size())) continue;
    if (line != tpu_dev_path) continue;
    return true;
  }
  return false;
}

// This function iterates through all the processes in /proc and logs if any
// process it was able to check is using the TPU. It does not have permission to
// processes owned by another user.
// TODO (shahrokhi) use tensorflow/core/platform/filesystem (GetChildren) for
// this.
std::string FindAndLogLibtpuProcess() {
  DIR* proc = opendir("/proc");

  if (proc == nullptr) {
    return "";
  }
  std::unique_ptr<DIR, int (*)(DIR*)> proc_dir(proc, closedir);
  struct dirent* ent;
  int64_t pid;
  while ((ent = readdir(proc))) {
    if (!isdigit(*ent->d_name)) continue;

    pid = strtol(ent->d_name, nullptr, 10);
    if (IsTpuUsed(pid)) {
      std::string error_message = "libtpu.so is already in use by process with pid " + std::to_string(pid) + ". Not attempting to load libtpu.so in this process.";
      return error_message;
   }
  }
  return "";
}

Status TryAcquireTpuLock() {
  static absl::Mutex* mu = new absl::Mutex();
  absl::MutexLock l(mu);

  static bool attempted_file_open = false;
  static bool should_load_library = false;

  if (!attempted_file_open) {
    std::string load_library_override =
        absl::StrCat(getenv("TPU_LOAD_LIBRARY"));

    if (load_library_override == "1") {
      return Status::OK();
    } else if (load_library_override == "0") {
      return errors::FailedPrecondition("load library override is not set");
    }

    // If TPU_CHIPS_PER_PROCESS_BOUNDS doesn't include all chips, we assume
    // we're using different chips in different processes and thus multiple
    // libtpu loads are ok.
    // TODO(skyewm): we could make per-chip lock files and look at
    // TPU_VISIBLE_DEVICES if we wanted to make this really precise.
    std::string chips_per_process_bounds =
        GetEnvVar("TPU_CHIPS_PER_PROCESS_BOUNDS");
    bool allow_multiple_libtpu_load =
        GetEnvBool("ALLOW_MULTIPLE_LIBTPU_LOAD", false);
    // TODO(skyewm): remove this when TPU_CHIPS_PER_HOST_BOUNDS is fully
    // deprecated
    if (chips_per_process_bounds.empty()) {
      chips_per_process_bounds = GetEnvVar("TPU_CHIPS_PER_HOST_BOUNDS");
    }
    if ((chips_per_process_bounds.empty() ||
         chips_per_process_bounds == "2,2,1") &&
        !allow_multiple_libtpu_load) {
      int fd = open("/tmp/libtpu_lockfile", O_CREAT | O_RDWR, 0644);

      // This lock is held until the process exits intentionally. The underlying
      // TPU device will be held on until it quits.
      if (lockf(fd, F_TLOCK, 0) != 0) {
	std::string error_message = FindAndLogLibtpuProcess();
        if (error_message == "") {
	  error_message = "libtpu.so already in use by another process probably  owned by another user. Run \"$ sudo lsof -w /dev/accel0\" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.";
       }
       return errors::Aborted(error_message);	
     } else {
	return Status::OK();
      }
    } else {
      VLOG(1) << "TPU_CHIPS_PER_PROCESS_BOUNDS is not empty or "
                 "ALLOW_MULTIPLE_LIBTPU_LOAD is set to True, "
                 "therefore allowing multiple libtpu.so loads.";
      return Status::OK();
    }
  }
}
#if defined(PLATFORM_GOOGLE)
Status InitializeTpuLibrary(void* library_handle) {
  return errors::Unimplemented("You must statically link in a TPU library.");
}
#else  // PLATFORM_GOOGLE
#include "tensorflow/core/tpu/tpu_library_init_fns.inc"

Status InitializeTpuLibrary(void* library_handle) {
  Status s = InitializeTpuStructFns(library_handle);

  // Retrieve arguments from environment if applicable
  std::pair<std::vector<std::string>, std::vector<const char*>> args =
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

namespace {
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
}  // namespace

Status FindAndLoadTpuLibrary() {
  const char* env_value = getenv("TPU_LIBRARY_PATH");
  const char* libtpu_path =
      env_value && strlen(env_value) > 0 ? env_value : "libtpu.so";
  LOG(INFO) << "Libtpu path is: " << libtpu_path;
  void* library = dlopen(libtpu_path, RTLD_NOW);
  if (library) {
    // We can open the shared library which means we are in a TPU environment.
    // Try to acquire exclusive access.
    Status tpu_lock_status = TryAcquireTpuLock();
    if (tpu_lock_status == Status::OK()) {
      Status initialize_library_status = InitializeTpuLibrary(library);
      if (initialize_library_status != Status::OK()){
        return initialize_library_status;
      }
    }
    else{
      return tpu_lock_status;
    }
  }

  InitializeCreateGcsFileSystemFnPtr();
  return Status::OK();
}

#endif  // PLATFORM_GOOGLE
std::pair<std::vector<std::string>, std::vector<const char*>>
GetLibTpuInitArguments() {
  // We make copies of the arguments returned by getenv because the memory
  // returned may be altered or invalidated by further calls to getenv.
  std::vector<std::string> args;
  std::vector<const char*> arg_ptrs;

  // Retrieve arguments from environment if applicable.
  char* env = getenv("LIBTPU_INIT_ARGS");
  if (env != nullptr) {
    // TODO(frankchn): Handles quotes properly if necessary.
    args = absl::StrSplit(env, ' ');
  }

  arg_ptrs.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    arg_ptrs.push_back(args[i].data());
  }

  return {std::move(args), std::move(arg_ptrs)};
}

}  // namespace tpu
}  // namespace tensorflow
