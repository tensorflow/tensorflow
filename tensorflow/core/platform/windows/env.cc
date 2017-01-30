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

#include "tensorflow/core/platform/env.h"

#include <Shlwapi.h>
#include <Windows.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#undef LoadLibrary
#undef ERROR

#include <thread>
#include <vector>
#include <string>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/windows/windows_file_system.h"

#pragma comment(lib, "Shlwapi.lib")

namespace tensorflow {

namespace {

class StdThread : public Thread {
 public:
  // name and thread_options are both ignored.
  StdThread(const ThreadOptions& thread_options, const string& name,
            std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

class WindowsEnv : public Env {
 public:
  WindowsEnv()
      : GetSystemTimePreciseAsFileTime_(NULL) {
    // GetSystemTimePreciseAsFileTime function is only available in the latest
    // versions of Windows. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    HMODULE module = GetModuleHandle("kernel32.dll");
    if (module != NULL) {
      auto func = (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
          module, "GetSystemTimePreciseAsFileTime");
      GetSystemTimePreciseAsFileTime_ = func;
    }
  }

  ~WindowsEnv() override {
    LOG(FATAL) << "Env::Default() must not be destroyed";
  }

  bool MatchPath(const string& path, const string& pattern) override {
    return PathMatchSpec(path.c_str(), pattern.c_str()) == TRUE;
  }

  void SleepForMicroseconds(int64 micros) override { Sleep(micros / 1000); }

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return new StdThread(thread_options, name, fn);
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

  Status LoadLibrary(const char *library_filename, void** handle) override {
    std::string file_name = library_filename;
    std::replace(file_name.begin(), file_name.end(), '/', '\\');

    HMODULE hModule = LoadLibraryEx(file_name.c_str(), NULL,
      LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!hModule) {
      return errors::NotFound(file_name + " not found");
    }
    *handle = hModule;
    return Status::OK();
  }

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
    void** symbol) override {
    FARPROC found_symbol;

    found_symbol = GetProcAddress((HMODULE)handle, symbol_name);
    if (found_symbol == NULL) {
      return errors::NotFound(std::string(symbol_name) + " not found");
    }
    *symbol = (void **)found_symbol;
    return Status::OK();
  }

  string FormatLibraryFileName(const string& name, const string& version)
    override {
    string filename;
    if (version.size() == 0) {
      filename = name + ".dll";
    }
    else {
      filename = name + version + ".dll";
    }
    return filename;
  }

 private:
  typedef VOID(WINAPI * FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

REGISTER_FILE_SYSTEM("", WindowsFileSystem);
REGISTER_FILE_SYSTEM("file", LocalWinFileSystem);

Env* Env::Default() {
  static Env* default_env = new WindowsEnv;
  return default_env;
}

}  // namespace tensorflow
