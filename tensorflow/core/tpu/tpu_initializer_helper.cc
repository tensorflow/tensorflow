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

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

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
        LOG(INFO) << "libtpu.so already in use by another process. "
                     "Run \"$ sudo lsof -w /dev/accel0\" to figure out "
                     "which process is using the TPU. Not "
                     "attempting to load libtpu.so in this process.";
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
