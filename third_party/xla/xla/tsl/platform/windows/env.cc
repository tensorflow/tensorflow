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

#include <Shlwapi.h>
#include <Windows.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstdint>
#undef ERROR

#include <string>
#include <thread>
#include <vector>

#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/ram_file_system.h"
#include "xla/tsl/platform/windows/wide_char.h"
#include "xla/tsl/platform/windows/windows_file_system.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/mutex.h"

#pragma comment(lib, "shlwapi.lib")

namespace tsl {

namespace {

mutex name_mutex(tsl::LINKER_INITIALIZED);

std::map<std::thread::id, string>& GetThreadNameRegistry()
    TF_EXCLUSIVE_LOCKS_REQUIRED(name_mutex) {
  static auto* const thread_name_registry =
      new std::map<std::thread::id, string>();
  return *thread_name_registry;
}

class StdThread : public Thread {
 public:
  // thread_options is ignored.
  StdThread(const ThreadOptions& thread_options, const string& name,
            absl::AnyInvocable<void()> fn)
      : thread_(std::move(fn)) {
    mutex_lock l(name_mutex);
    GetThreadNameRegistry().emplace(thread_.get_id(), name);
  }

  ~StdThread() override {
    std::thread::id thread_id = thread_.get_id();
    thread_.join();
    mutex_lock l(name_mutex);
    GetThreadNameRegistry().erase(thread_id);
  }

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
    std::wstring ws_path(Utf8ToWideChar(path));
    std::wstring ws_pattern(Utf8ToWideChar(pattern));
    return PathMatchSpecW(ws_path.c_str(), ws_pattern.c_str()) == TRUE;
  }

  void SleepForMicroseconds(int64 micros) override { Sleep(micros / 1000); }

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      absl::AnyInvocable<void()> fn) override {
    return new StdThread(thread_options, name, std::move(fn));
  }

  int64_t GetCurrentThreadId() override {
    return static_cast<int64_t>(::GetCurrentThreadId());
  }

  bool GetCurrentThreadName(string* name) override {
    mutex_lock l(name_mutex);
    auto thread_name = GetThreadNameRegistry().find(std::this_thread::get_id());
    if (thread_name != GetThreadNameRegistry().end()) {
      *name = thread_name->second;
      return true;
    } else {
      return false;
    }
  }

  static VOID CALLBACK SchedClosureCallback(PTP_CALLBACK_INSTANCE Instance,
                                            PVOID Context, PTP_WORK Work) {
    CloseThreadpoolWork(Work);
    absl::AnyInvocable<void()>* f = (absl::AnyInvocable<void()>*)Context;
    (*f)();
    delete f;
  }
  void SchedClosure(absl::AnyInvocable<void()> closure) override {
    PTP_WORK work = CreateThreadpoolWork(
        SchedClosureCallback,
        new absl::AnyInvocable<void()>(std::move(closure)), nullptr);
    SubmitThreadpoolWork(work);
  }

  static VOID CALLBACK SchedClosureAfterCallback(PTP_CALLBACK_INSTANCE Instance,
                                                 PVOID Context,
                                                 PTP_TIMER Timer) {
    CloseThreadpoolTimer(Timer);
    absl::AnyInvocable<void()>* f = (absl::AnyInvocable<void()>*)Context;
    (*f)();
    delete f;
  }

  void SchedClosureAfter(int64 micros,
                         absl::AnyInvocable<void()> closure) override {
    PTP_TIMER timer = CreateThreadpoolTimer(
        SchedClosureAfterCallback,
        new absl::AnyInvocable<void()>(std::move(closure)), nullptr);
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

  Status LoadDynamicLibrary(const char* library_filename,
                            void** handle) override {
    return internal::LoadDynamicLibrary(library_filename, handle);
  }

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
    return internal::GetSymbolFromLibrary(handle, symbol_name, symbol);
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
    return internal::FormatLibraryFileName(name, version);
  }

  string GetRunfilesDir() override {
    string bin_path = this->GetExecutablePath();
    string runfiles_path = bin_path + ".runfiles\\org_tensorflow";
    Status s = this->IsDirectory(runfiles_path);
    if (s.ok()) {
      return runfiles_path;
    } else {
      return bin_path.substr(0, bin_path.find_last_of("/\\"));
    }
  }

 private:
  void GetLocalTempDirectories(std::vector<string>* list) override;

  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

REGISTER_FILE_SYSTEM("", WindowsFileSystem);
REGISTER_FILE_SYSTEM("file", LocalWinFileSystem);
REGISTER_FILE_SYSTEM("ram", RamFileSystem);

Env* Env::Default() {
  static Env* const default_env = new WindowsEnv;
  return default_env;
}

void WindowsEnv::GetLocalTempDirectories(std::vector<string>* list) {
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

int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite) {
    char* env_val = getenv(name);
    if (env_val) {
      return 0;
    }
  }
  return _putenv_s(name, value);
}

int unsetenv(const char* name) { return _putenv_s(name, ""); }

}  // namespace tsl
