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

#include "xla/tsl/platform/env.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifdef __FreeBSD__
#include <pthread_np.h>
#endif

#ifdef __APPLE__
#include <mach/thread_info.h>  // for MAXTHREADNAMESIZE
#endif

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <cstdint>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/default/posix_file_system.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/ram_file_system.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/mutex.h"

namespace tsl {

namespace {

#if defined(__APPLE__)
constexpr int kMaxThreadNameLen = MAXTHREADNAMESIZE - 1;
#elif defined(__FreeBSD__)
constexpr int kMaxThreadNameLen = MAXCOMLEN - 1;
#elif defined(__linux__)
// Per the man pages, the maximum length of a thread name is 15 characters.
constexpr int kMaxThreadNameLen = 15;
#else
constexpr int kMaxThreadNameLen = 0;
#endif

absl::Mutex name_mutex(absl::kConstInit);

absl::flat_hash_map<std::thread::id, std::string>& GetThreadNameRegistry()
    ABSL_SHARED_LOCKS_REQUIRED(name_mutex) {
  static auto* thread_name_registry =
      new absl::flat_hash_map<std::thread::id, std::string>();
  return *thread_name_registry;
}

int64_t GetCurrentThreadIdInternal() {
#ifdef __APPLE__
  uint64_t tid64;
  pthread_threadid_np(nullptr, &tid64);
  return static_cast<int64_t>(tid64);
#elif defined(__FreeBSD__)
  return pthread_getthreadid_np();
#elif defined(__NR_gettid)
  return static_cast<int64_t>(syscall(__NR_gettid));
#else
  return std::hash<std::thread::id>()(std::this_thread::get_id());
#endif
}

// We use the pthread API instead of std::thread so we can control stack sizes.
class PThread : public Thread {
 public:
  PThread(const ThreadOptions& thread_options, const std::string& name,
          absl::AnyInvocable<void()> fn) {
    ThreadParams* params = new ThreadParams;
    params->name = name;
    params->fn = std::move(fn);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    if (thread_options.stack_size != 0) {
      pthread_attr_setstacksize(&attributes, thread_options.stack_size);
    }
    int ret = pthread_create(&thread_, &attributes, &ThreadFn, params);
    // There is no mechanism for the thread creation API to fail, so we CHECK.
    CHECK_EQ(ret, 0) << "Thread " << name
                     << " creation via pthread_create() failed.";
    pthread_attr_destroy(&attributes);
  }

  ~PThread() override { pthread_join(thread_, nullptr); }

 private:
  struct ThreadParams {
    std::string name;
    absl::AnyInvocable<void()> fn;
  };
  static void* ThreadFn(void* params_arg) {
    std::unique_ptr<ThreadParams> params(
        reinterpret_cast<ThreadParams*>(params_arg));
    {
      absl::MutexLock l(&name_mutex);
      GetThreadNameRegistry().emplace(std::this_thread::get_id(), params->name);
    }
    if constexpr (kMaxThreadNameLen > 0) {
      std::array<char, kMaxThreadNameLen + 1> buf;
      absl::SNPrintF(buf.data(), buf.size(), "%s/%u", params->name.c_str(),
                     GetCurrentThreadIdInternal());
      buf[sizeof(buf) - 1] = '\0';
#if defined(__APPLE__) || defined(__FreeBSD__)
      pthread_set_name_np(pthread_self(), params->name.c_str());
#elif defined(__linux__)
      // NOLINTNEXTLINE: ABI requires unsigned long.
      prctl(PR_SET_NAME, reinterpret_cast<unsigned long>(buf));
#endif
    }
    params->fn();
    {
      absl::MutexLock l(&name_mutex);
      GetThreadNameRegistry().erase(std::this_thread::get_id());
    }
    return nullptr;
  }

  pthread_t thread_;
};

class PosixEnv : public Env {
 public:
  PosixEnv() {}

  ~PosixEnv() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }

  bool MatchPath(const string& path, const string& pattern) override {
    return fnmatch(pattern.c_str(), path.c_str(), FNM_PATHNAME) == 0;
  }

  void SleepForMicroseconds(int64 micros) override {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64_t>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64_t>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  Thread* StartThread(const ThreadOptions& thread_options,
                      const std::string& name,
                      absl::AnyInvocable<void()> fn) override {
    return new PThread(thread_options, name, std::move(fn));
  }

  int64_t GetCurrentThreadId() override {
    static thread_local int64_t current_thread_id =
        GetCurrentThreadIdInternal();
    return current_thread_id;
  }

  bool GetCurrentThreadName(std::string* name) override {
    {
      absl::ReaderMutexLock l(&name_mutex);
      auto thread_name =
          GetThreadNameRegistry().find(std::this_thread::get_id());
      if (thread_name != GetThreadNameRegistry().end()) {
        *name = absl::StrCat(thread_name->second, "/", GetCurrentThreadId());
        return true;
      }
    }

    if constexpr (kMaxThreadNameLen > 0) {
      int res = 0;
      char buf[kMaxThreadNameLen + 1];
#if defined(__FreeBSD__) || defined(__APPLE__)
      res = pthread_get_name_np(pthread_self(), buf, std::size(buf));
#elif defined(__linux__)
      // NOLINTNEXTLINE: ABI requires unsigned long.
      res = prctl(PR_GET_NAME, reinterpret_cast<unsigned long>(buf));
#endif
      if (res != 0) {
        return false;
      }
      *name = buf;
      return true;
    }

    return false;
  }

  void SchedClosure(absl::AnyInvocable<void()> closure) override {
    // TODO(b/27290852): Spawning a new thread here is wasteful, but
    // needed to deal with the fact that many `closure` functions are
    // blocking in the current codebase.
    std::thread closure_thread(std::move(closure));
    closure_thread.detach();
  }

  void SchedClosureAfter(int64 micros,
                         absl::AnyInvocable<void()> closure) override {
    // TODO(b/27290852): Consuming a thread here is wasteful, but this
    // code is (currently) only used in the case where a step fails
    // (AbortStep). This could be replaced by a timer thread
    SchedClosure([this, micros, closure = std::move(closure)]() mutable {
      SleepForMicroseconds(micros);
      closure();
    });
  }

  absl::Status LoadDynamicLibrary(const char* library_filename,
                                  void** handle) override {
    return internal::LoadDynamicLibrary(library_filename, handle);
  }

  absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                    void** symbol) override {
    return internal::GetSymbolFromLibrary(handle, symbol_name, symbol);
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
    return internal::FormatLibraryFileName(name, version);
  }

  string GetRunfilesDir() override {
    string bin_path = this->GetExecutablePath();
    string runfiles_suffix = ".runfiles/org_tensorflow";
    std::size_t pos = bin_path.find(runfiles_suffix);

    // Sometimes (when executing under python) bin_path returns the full path to
    // the python scripts under runfiles. Get the substring.
    if (pos != std::string::npos) {
      return bin_path.substr(0, pos + runfiles_suffix.length());
    }

    // See if we have the executable path. if executable.runfiles exists, return
    // that folder.
    string runfiles_path = bin_path + runfiles_suffix;
    absl::Status s = this->IsDirectory(runfiles_path);
    if (s.ok()) {
      return runfiles_path;
    }

    // If nothing can be found, return something close.
    return bin_path.substr(0, bin_path.find_last_of("/\\"));
  }

 private:
  void GetLocalTempDirectories(std::vector<string>* list) override;
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__APPLE__) || defined(__ANDROID__)
REGISTER_FILE_SYSTEM("", PosixFileSystem);
REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
REGISTER_FILE_SYSTEM("ram", RamFileSystem);

Env* Env::Default() {
  static Env* default_env = new PosixEnv;
  return default_env;
}
#endif

void PosixEnv::GetLocalTempDirectories(std::vector<string>* list) {
  list->clear();
  // Directories, in order of preference. If we find a dir that
  // exists, we stop adding other less-preferred dirs
  const char* candidates[] = {
    // Non-null only during unittest/regtest
    getenv("TEST_TMPDIR"),

    // Explicitly-supplied temp dirs
    getenv("TMPDIR"),
    getenv("TMP"),

#if defined(__ANDROID__)
    "/data/local/tmp",
#endif

    // If all else fails
    "/tmp",
  };

  std::vector<std::string> paths;  // Only in case of errors.
  for (const char* d : candidates) {
    if (!d || d[0] == '\0') continue;  // Empty env var
    paths.push_back(d);
    // Make sure we don't surprise anyone who's expecting a '/'
    string dstr = d;
    if (dstr[dstr.size() - 1] != '/') {
      dstr += "/";
    }

    struct stat statbuf;
    if (!stat(d, &statbuf) && S_ISDIR(statbuf.st_mode) &&
        !access(dstr.c_str(), 0)) {
      // We found a dir that exists and is accessible - we're done.
      list->push_back(dstr);
      return;
    }
  }
  LOG(WARNING) << "We are not able to find a directory for temporary files.\n"
               << "Verify the directory access and available space under: "
               << absl::StrJoin(paths, ",") << ". "
               << "You can also provide a directory for temporary files with"
               << " the environment variable TMP or TMPDIR. "
               << "Example under bash: `export TMP=/my_new_temp_directory;`";
}

int setenv(const char* name, const char* value, int overwrite) {
  return ::setenv(name, value, overwrite);
}

int unsetenv(const char* name) { return ::unsetenv(name); }

}  // namespace tsl
