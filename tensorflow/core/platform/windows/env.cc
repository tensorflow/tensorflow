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

#include <string>
#include <thread>
#include <vector>

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
  WindowsEnv() : GetSystemTimePreciseAsFileTime_(NULL) {
    // GetSystemTimePreciseAsFileTime function is only available in the latest
    // versions of Windows. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    HMODULE module = GetModuleHandleW(L"kernel32.dll");
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
    std::wstring ws_path(WindowsFileSystem::Utf8ToWideChar(path));
    std::wstring ws_pattern(WindowsFileSystem::Utf8ToWideChar(pattern));
    return PathMatchSpecW(ws_path.c_str(), ws_pattern.c_str()) == TRUE;
  }

  void SleepForMicroseconds(int64 micros) override { Sleep(micros / 1000); }

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return new StdThread(thread_options, name, fn);
  }

  static VOID CALLBACK SchedClosureCallback(PTP_CALLBACK_INSTANCE Instance,
                                            PVOID Context, PTP_WORK Work) {
    CloseThreadpoolWork(Work);
    std::function<void()>* f = (std::function<void()>*)Context;
    (*f)();
    delete f;
  }
  void SchedClosure(std::function<void()> closure) override {
    PTP_WORK work = CreateThreadpoolWork(
        SchedClosureCallback, new std::function<void()>(std::move(closure)),
        nullptr);
    SubmitThreadpoolWork(work);
  }

  static VOID CALLBACK SchedClosureAfterCallback(PTP_CALLBACK_INSTANCE Instance,
                                                 PVOID Context,
                                                 PTP_TIMER Timer) {
    CloseThreadpoolTimer(Timer);
    std::function<void()>* f = (std::function<void()>*)Context;
    (*f)();
    delete f;
  }

  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
    PTP_TIMER timer = CreateThreadpoolTimer(
        SchedClosureAfterCallback,
        new std::function<void()>(std::move(closure)), nullptr);
    // in 100 nanosecond units
    FILETIME FileDueTime;
    ULARGE_INTEGER ulDueTime;
    // Negative indicates the amount of time to wait is relative to the current
    // time.
    ulDueTime.QuadPart = (ULONGLONG) - (10 * micros);
    FileDueTime.dwHighDateTime = ulDueTime.HighPart;
    FileDueTime.dwLowDateTime = ulDueTime.LowPart;
    SetThreadpoolTimer(timer, &FileDueTime, 0, 0);
  }

  Status LoadLibrary(const char* library_filename, void** handle) override {
    std::string file_name = library_filename;
    std::replace(file_name.begin(), file_name.end(), '/', '\\');

    std::wstring ws_file_name(WindowsFileSystem::Utf8ToWideChar(file_name));

    HMODULE hModule = LoadLibraryExW(ws_file_name.c_str(), NULL,
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
    *symbol = (void**)found_symbol;
    return Status::OK();
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
    string filename;
    if (version.size() == 0) {
      filename = name + ".dll";
    } else {
      filename = name + version + ".dll";
    }
    return filename;
  }

 private:
  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

REGISTER_FILE_SYSTEM("", WindowsFileSystem);
REGISTER_FILE_SYSTEM("file", LocalWinFileSystem);

Env* Env::Default() {
  static Env* default_env = new WindowsEnv;
  return default_env;
}

void Env::GetLocalTempDirectories(std::vector<string>* list) {
  list->clear();
  // On windows we'll try to find a directory in this order:
  //   C:/Documents & Settings/whomever/TEMP (or whatever GetTempPath() is)
  //   C:/TMP/
  //   C:/TEMP/
  //   C:/WINDOWS/ or C:/WINNT/
  //   .
  char tmp[MAX_PATH];
  // GetTempPath can fail with either 0 or with a space requirement > bufsize.
  // See http://msdn.microsoft.com/en-us/library/aa364992(v=vs.85).aspx
  DWORD n = GetTempPathA(MAX_PATH, tmp);
  if (n > 0 && n <= MAX_PATH) list->push_back(tmp);
  list->push_back("C:\\tmp\\");
  list->push_back("C:\\temp\\");
}

}  // namespace tensorflow
