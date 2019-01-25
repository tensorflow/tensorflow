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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <thread>
#include <vector>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"

namespace tensorflow {

namespace {

class StdThread : public Thread {
 public:
  // name and thread_options are both ignored.
  StdThread(const ThreadOptions& thread_options, const string& name,
            std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() override { thread_.join(); }

 private:
  std::thread thread_;
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
            std::min<int64>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64>(sleep_time.tv_sec) * 1e6;
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

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return new StdThread(thread_options, name, fn);
  }

  int32 GetCurrentThreadId() {
#ifdef __APPLE__
    pthread_threadid_np(nullptr, &tid64);
    return static_cast<int32>(tid64);
#elif defined(__FreeBSD__)
    // Has to be casted to long first, else this error appears:
    // static_cast from 'pthread_t' (aka 'pthread *') to 'int32' (aka 'int')
    // is not allowed
    return static_cast<int32>(static_cast<int64>(pthread_self()));
#else
    return static_cast<int32>(pthread_self());
#endif
  }

  bool GetCurrentThreadName(string* name) {
#ifdef __ANDROID__
    return false;
#else
    char buf[100];
    int res = pthread_getname_np(pthread_self(), buf, static_cast<size_t>(100));
    if (res != 0) {
      return false;
    }
    *name = buf;
    return true;
#endif
  }

  void SchedClosure(std::function<void()> closure) override {
    // TODO(b/27290852): Spawning a new thread here is wasteful, but
    // needed to deal with the fact that many `closure` functions are
    // blocking in the current codebase.
    std::thread closure_thread(closure);
    closure_thread.detach();
  }

  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
    // TODO(b/27290852): Consuming a thread here is wasteful, but this
    // code is (currently) only used in the case where a step fails
    // (AbortStep). This could be replaced by a timer thread
    SchedClosure([this, micros, closure]() {
      SleepForMicroseconds(micros);
      closure();
    });
  }

  Status LoadLibrary(const char* library_filename, void** handle) override {
    return tensorflow::internal::LoadLibrary(library_filename, handle);
  }

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
    return tensorflow::internal::GetSymbolFromLibrary(handle, symbol_name,
                                                      symbol);
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
    return tensorflow::internal::FormatLibraryFileName(name, version);
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
    Status s = this->IsDirectory(runfiles_path);
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

#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
REGISTER_FILE_SYSTEM("", PosixFileSystem);
REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
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

  for (const char* d : candidates) {
    if (!d || d[0] == '\0') continue;  // Empty env var

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
}

}  // namespace tensorflow
