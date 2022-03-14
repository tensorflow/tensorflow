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
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/logging.h"

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
bool FindAndLogLibtpuProcess() {
  DIR* proc = opendir("/proc");

  if (proc == nullptr) {
    return false;
  }
  std::unique_ptr<DIR, int (*)(DIR*)> proc_dir(proc, closedir);
  struct dirent* ent;
  int64_t pid;
  while ((ent = readdir(proc))) {
    if (!isdigit(*ent->d_name)) continue;

    pid = strtol(ent->d_name, nullptr, 10);
    if (IsTpuUsed(pid)) {
      LOG(INFO) << "libtpu.so is already in use by process with pid " << pid
                << ". Not attempting to load libtpu.so in this process.";
      return true;
    }
  }
  return false;
}

bool TryAcquireTpuLock() {
  static absl::Mutex* mu = new absl::Mutex();
  absl::MutexLock l(mu);

  static bool attempted_file_open = false;
  static bool should_load_library = false;

  if (!attempted_file_open) {
    std::string load_library_override =
        absl::StrCat(getenv("TPU_LOAD_LIBRARY"));

    if (load_library_override == "1") {
      return true;
    } else if (load_library_override == "0") {
      return false;
    }
    should_load_library = true;

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
        if (!FindAndLogLibtpuProcess()) {
          LOG(INFO) << "libtpu.so already in use by another process probably"
                       " owned by another user. "
                       "Run \"$ sudo lsof -w /dev/accel0\" to figure out "
                       "which process is using the TPU. Not "
                       "attempting to load libtpu.so in this process.";
        }
        should_load_library = false;
      } else {
        should_load_library = true;
      }
    } else {
      VLOG(1) << "TPU_CHIPS_PER_PROCESS_BOUNDS is not empty or "
                 "ALLOW_MULTIPLE_LIBTPU_LOAD is set to True, "
                 "therefore allowing multiple libtpu.so loads.";
      should_load_library = true;
    }
  }

  return should_load_library;
}

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
