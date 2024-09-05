/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/windows/stacktrace.h"

// clang-format off
#include <windows.h>  // Windows.h must be declared above dgbhelp.
#include <dbghelp.h>
// clang-format on

#include <string>

#include "tsl/platform/mutex.h"

#pragma comment(lib, "dbghelp.lib")

namespace tsl {

// We initialize the Symbolizer on first call:
// https://docs.microsoft.com/en-us/windows/win32/debug/initializing-the-symbol-handler
static bool SymbolsAreAvailableInit() {
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  return SymInitialize(GetCurrentProcess(), NULL, true);
}

static bool SymbolsAreAvailable() {
  static bool kSymbolsAvailable = SymbolsAreAvailableInit();  // called once
  return kSymbolsAvailable;
}

// Generating stacktraces involve two steps:
// 1. Producing a list of pointers, where each pointer corresponds to the
//    function called at each stack frame (aka stack unwinding).
// 2. Converting each pointer into a human readable string corresponding to
//    the function's name (aka symbolization).
// Windows provides two APIs for stack unwinding: StackWalk
// (https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-stackwalk64)
// and CaptureStackBackTrace
// (https://docs.microsoft.com/en-us/windows/win32/debug/capturestackbacktrace).
// Although StackWalk is more flexible, it does not have any threadsafety
// guarantees. See https://stackoverflow.com/a/17890764
// Windows provides one symbolization API, SymFromAddr:
// https://docs.microsoft.com/en-us/windows/win32/debug/retrieving-symbol-information-by-address
// which is unfortunately not threadsafe. Therefore, we acquire a lock prior to
// calling it, making this function NOT async-signal-safe.
// FYI from m3b@ about signal safety:
// Functions that block when acquiring mutexes are not async-signal-safe
// primarily because the signal might have been delivered to a thread that holds
// the lock. That is, the thread could self-deadlock if a signal is delivered at
// the wrong moment; no other threads are needed.
std::string CurrentStackTrace() {
  // For reference, many stacktrace-related Windows APIs are documented here:
  // https://docs.microsoft.com/en-us/windows/win32/debug/about-dbghelp.
  HANDLE current_process = GetCurrentProcess();
  static constexpr int kMaxStackFrames = 64;
  void* trace[kMaxStackFrames];
  int num_frames = CaptureStackBackTrace(0, kMaxStackFrames, trace, NULL);

  static mutex mu(tsl::LINKER_INITIALIZED);

  std::string stacktrace;
  for (int i = 0; i < num_frames; ++i) {
    const char* symbol = "(unknown)";
    if (SymbolsAreAvailable()) {
      char symbol_info_buffer[sizeof(SYMBOL_INFO) +
                              MAX_SYM_NAME * sizeof(TCHAR)];
      SYMBOL_INFO* symbol_ptr =
          reinterpret_cast<SYMBOL_INFO*>(symbol_info_buffer);
      symbol_ptr->SizeOfStruct = sizeof(SYMBOL_INFO);
      symbol_ptr->MaxNameLen = MAX_SYM_NAME;

      // Because SymFromAddr is not threadsafe, we acquire a lock.
      mutex_lock lock(mu);
      if (SymFromAddr(current_process, reinterpret_cast<DWORD64>(trace[i]), 0,
                      symbol_ptr)) {
        symbol = symbol_ptr->Name;
      }
    }

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "0x%p\t%s", trace[i], symbol);
    stacktrace += buffer;
    stacktrace += "\n";
  }

  return stacktrace;
}

}  // namespace tsl
