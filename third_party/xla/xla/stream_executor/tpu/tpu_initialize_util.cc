/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/tpu_initialize_util.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace tpu {
namespace {

static bool libtpu_acquired = false;
static int lock_fd = -1;

absl::Mutex& GetTpuLockMutex() {
  static absl::Mutex* const mu = new absl::Mutex();
  return *mu;
}

std::string GetEnvVar(const char* name) {
  // Constructing a std::string directly from nullptr is undefined behavior so
  // we can return empty string in that case
  const char* env_value = getenv(name);
  if (env_value == nullptr) {
    return "";
  }
  return std::string(env_value);
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

const char* GetTpuDriverFile() {
  static const char* tpu_dev_path = []() {
    struct stat sb;
    if (stat("/dev/accel0", &sb) == 0) {
      return "/dev/accel0";
    }
    return "/dev/vfio/0";
  }();
  return tpu_dev_path;
}

// This function (IsTpuUsed) gets pid of a process and checks if that process
// is using tpu. It is not able to check processes that are owned by another
// user.
bool IsTpuUsed(int64_t pid) {
  std::string fd_dir_path = absl::StrCat("/proc/", pid, "/fd");
  DIR* raw_fd_dir = opendir(fd_dir_path.c_str());
  if (raw_fd_dir == nullptr) {
    return false;
  }
  std::unique_ptr<DIR, int (*)(DIR*)> fd_dir(raw_fd_dir, closedir);
  struct dirent* ent;
  std::string link_target;
  std::string tpu_dev_path = GetTpuDriverFile();
  // Resize line to be one byte larger than tpu_dev_path.size() to detect
  // if the readlink result is longer than expected.
  link_target.resize(tpu_dev_path.size() + 1);
  std::string fd_file_path = absl::StrCat(fd_dir_path, "/");
  size_t base_len = fd_file_path.size();

  while ((ent = readdir(fd_dir.get()))) {
    absl::string_view d_name(ent->d_name);

    if (!absl::c_all_of(d_name, absl::ascii_isdigit)) {
      continue;
    }

    fd_file_path.resize(base_len);
    fd_file_path.append(d_name.data(), d_name.size());
    ssize_t len =
        readlink(fd_file_path.c_str(), link_target.data(), link_target.size());
    if (len < 0 || len != static_cast<ssize_t>(tpu_dev_path.size())) {
      continue;
    }
    if (absl::string_view(link_target.data(), len) != tpu_dev_path) {
      continue;
    }
    return true;
  }
  return false;
}

// This function iterates through all the processes in /proc and finds out if
// any process it was able to check is using the TPU. It does not have
// permission to processes owned by another user.
// TODO (shahrokhi) use tensorflow/core/platform/filesystem (GetChildren) for
// this.
absl::StatusOr<int64_t> FindLibtpuProcess() {
  DIR* proc = opendir("/proc");

  if (proc == nullptr) {
    return absl::UnavailableError("was not able to open /proc");
  }
  std::unique_ptr<DIR, int (*)(DIR*)> proc_dir(proc, closedir);
  struct dirent* ent;
  while ((ent = readdir(proc_dir.get()))) {
    if (!absl::ascii_isdigit(*ent->d_name)) {
      continue;
    }

    int64_t pid;
    if (!absl::SimpleAtoi(ent->d_name, &pid)) {
      continue;
    }
    if (IsTpuUsed(pid)) {
      return pid;
    }
  }
  return absl::NotFoundError("did not find which pid uses the libtpu.so");
}

// Attempts to open a lock file at the given path.
// We use O_EXCL to securely create the file if it doesn't exist,
// preventing symlink races, and O_CLOEXEC to prevent
// descriptor leaks to child processes. If the file already exists (EEXIST),
// we fall back to opening it with O_RDWR | O_NOFOLLOW | O_CLOEXEC.
int OpenLockFile(const char* path) {
  int fd = open(path, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  if (fd == -1 && errno == EEXIST) {
    fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
  }
  return fd;
}

// Returns a secure file descriptor for the multi-process TPU lockfile,
// or -1 if it was unable to open a lockfile in any safe location.
int OpenTpuLockFile() {
  static constexpr char libtpu_lockfn[] = "/tmp/libtpu_lockfile";
  // 1. Attempt to open the standard lock file.
  int fd = OpenLockFile(libtpu_lockfn);
  if (fd != -1) {
    return fd;
  }

  // 2. Fallback to a user-specific directory under /tmp with 0700 permissions
  // to prevent local DoS attacks from other users.
  uid_t uid = getuid();
  std::string user_lockdir = absl::StrCat("/tmp/libtpu_lock_", uid);
  bool dir_ok = false;
  if (mkdir(user_lockdir.c_str(), 0700) == 0 || errno == EEXIST) {
    // Validate directory ownership to prevent preemptive directory hijacking.
    struct stat st;
    if (lstat(user_lockdir.c_str(), &st) == 0) {
      if (S_ISDIR(st.st_mode) && st.st_uid == uid) {
        dir_ok = true;
      }
    }
  }

  if (dir_ok) {
    std::string user_lockfn = absl::StrCat(user_lockdir, "/libtpu_lockfile");
    fd = OpenLockFile(user_lockfn.c_str());
    if (fd != -1) {
      return fd;
    }
  }

  // 3. Fallback to the user's home directory as a last resort.
  const char* home_env = getenv("HOME");
  if (home_env != nullptr && home_env[0] != '\0') {
    std::string home_lockfn = absl::StrCat(home_env, "/.libtpu_lockfile");
    fd = OpenLockFile(home_lockfn.c_str());
    if (fd != -1) {
      return fd;
    }
  }

  return -1;
}

}  // namespace

absl::Status TryAcquireTpuLock() {
  absl::MutexLock l(GetTpuLockMutex());

  if (libtpu_acquired) {
    return absl::OkStatus();
  }

  std::string load_library_override = GetEnvVar("TPU_LOAD_LIBRARY");

  if (load_library_override == "1") {
    VLOG(1) << "TPU_LOAD_LIBRARY=1, force loading libtpu";
    libtpu_acquired = true;
    return absl::OkStatus();
  }
  if (load_library_override == "0") {
    return absl::FailedPreconditionError(
        "TPU_LOAD_LIBRARY=0, not loading libtpu");
  }

  bool allow_multiple_libtpu_load =
      GetEnvBool("ALLOW_MULTIPLE_LIBTPU_LOAD", false);

  if (allow_multiple_libtpu_load) {
    VLOG(1) << "ALLOW_MULTIPLE_LIBTPU_LOAD is set to True, "
               "allowing multiple concurrent libtpu.so loads.";
    libtpu_acquired = true;
    return absl::OkStatus();
  }

  std::string chips_per_process_bounds =
      GetEnvVar("TPU_CHIPS_PER_PROCESS_BOUNDS");
  if (chips_per_process_bounds.empty()) {
    // TODO(skyewm): remove this when TPU_CHIPS_PER_HOST_BOUNDS is fully
    // deprecated
    chips_per_process_bounds = GetEnvVar("TPU_CHIPS_PER_HOST_BOUNDS");
  }

  // TODO(b/291278826): make per-chip lock files and look at TPU_VISIBLE_DEVICES
  // to make TPU process mutex separation more accurate.
  bool use_all_tpus =
      chips_per_process_bounds.empty() || chips_per_process_bounds == "2,2,1";
  if (!use_all_tpus) {
    VLOG(1) << "TPU_CHIPS_PER_PROCESS_BOUNDS is a subset of host's TPU "
               "devices, allowing multiple libtpu.so loads.";
    libtpu_acquired = true;
    return absl::OkStatus();
  }

  int fd = OpenTpuLockFile();
  if (fd == -1) {
    // File open permission locks multi-user access by default.
    uid_t uid = getuid();
    std::string user_lockdir = absl::StrCat("/tmp/libtpu_lock_", uid);
    return absl::AbortedError(absl::StrCat(
        "The TPU is already in use or the lock file is stale. Run \"$ sudo ",
        "lsof -w ", GetTpuDriverFile(),
        "\" to figure out which process is using the TPU. If no process is "
        "listed, a stale lock file might exist. The potential lock file "
        "locations are '/tmp/libtpu_lockfile' and '",
        user_lockdir,
        "'. You can attempt to remove them with: \"$ rm /tmp/libtpu_lockfile\" "
        "and \"$ rm -rf ",
        user_lockdir,
        "\". Note: Only remove these files if you are certain no other "
        "legitimate process is using the TPU."));
  }

  // lockf()/fcntl() record locks are associated with [Process, Inode].
  // WARNING: POSIX locks established using lockf can be silently dropped
  // process-wide if secondary modules open and subsequently close a separate
  // fd referencing the identical path. Ensure strict lockfile name
  // exclusivity!
  if (lockf(fd, F_TLOCK, 0) != 0) {
    // Explicitly closing the FD on lock failure prevents descriptor leaks
    // if the caller retries or continues.
    close(fd);
    auto pid = FindLibtpuProcess();
    if (pid.ok()) {
      return absl::AbortedError(absl::StrCat(
          "The TPU is already in use by process with pid ", pid.value(),
          ". Not attempting to load libtpu.so in this process."));
    }
    uid_t uid = getuid();
    std::string user_lockdir = absl::StrCat("/tmp/libtpu_lock_", uid);
    return absl::AbortedError(absl::StrCat(
        "Internal error when accessing libtpu multi-process lockfile. "
        "The potential lock file locations are '/tmp/libtpu_lockfile' and '",
        user_lockdir,
        "'. You can attempt to remove them with: \"$ rm "
        "/tmp/libtpu_lockfile\" and \"$ rm -rf ",
        user_lockdir,
        "\". Note: Only remove these files if you are certain no other "
        "legitimate process is using the TPU."));
  }
  lock_fd = fd;  // Explicitly persist lock descriptor for process lifetime.
  libtpu_acquired = true;
  return absl::OkStatus();
}

void ResetTpuLockStateForTesting() {
  absl::MutexLock l(GetTpuLockMutex());
  if (lock_fd != -1) {
    close(lock_fd);
    lock_fd = -1;
  }
  libtpu_acquired = false;
}

std::pair<std::vector<std::string>, std::vector<const char*>>
GetLibTpuInitArguments() {
  // We make copies of the arguments returned by getenv because the memory
  // returned may be altered or invalidated by further calls to getenv.
  std::pair<std::vector<std::string>, std::vector<const char*>> result;

  // Retrieve arguments from environment if applicable.
  char* env = getenv("LIBTPU_INIT_ARGS");
  if (env != nullptr) {
    // TODO(frankchn): Handles quotes properly if necessary.
    result.first = absl::StrSplit(env, ' ');
  }

  result.second.reserve(result.first.size() + 1);
  for (const auto& arg : result.first) {
    result.second.push_back(arg.data());
  }
  result.second.push_back(nullptr);

  return result;
}

}  // namespace tpu
}  // namespace tensorflow
