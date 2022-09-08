/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_STACKTRACE_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_STACKTRACE_H_

// clang-format off
#include "tensorflow/tsl/platform/platform.h"
// clang-format on

#if !defined(IS_MOBILE_PLATFORM) && (defined(__clang__) || defined(__GNUC__))
#define TF_HAS_STACKTRACE
#endif

#if defined(TF_HAS_STACKTRACE)
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#endif  // defined(TF_GENERATE_BACKTRACE)

#include <sstream>
#include <string>
#include "tensorflow/core/platform/abi.h"

namespace tsl {

// Function to create a pretty stacktrace.
inline std::string CurrentStackTrace() {
#if defined(TF_HAS_STACKTRACE)
  std::stringstream ss("");
  ss << "*** Begin stack trace ***" << std::endl;

  // Get the mangled stack trace.
  int buffer_size = 128;
  void* trace[128];
  buffer_size = backtrace(trace, buffer_size);

  for (int i = 0; i < buffer_size; ++i) {
    const char* symbol = "";
    Dl_info info;
    if (dladdr(trace[i], &info)) {
      if (info.dli_sname != nullptr) {
        symbol = info.dli_sname;
      }
    }

    std::string demangled = tensorflow::port::MaybeAbiDemangle(symbol);
    if (demangled.length()) {
      ss << "\t" << demangled << std::endl;
    } else {
      ss << "\t" << symbol << std::endl;
    }
  }

  ss << "*** End stack trace ***" << std::endl;
  return ss.str();
#else
  return std::string();
#endif  // defined(TF_HAS_STACKTRACE)
}

inline void DebugWriteToString(const char* data, void* arg) {
  reinterpret_cast<std::string*>(arg)->append(data);
}

// A dummy class that does nothing.  Someday, add real support.
class SavedStackTrace {
 public:
  SavedStackTrace() {}

  void CreateCurrent(int skip_count) {}

  void Reset() {}

  typedef void DebugWriter(const char*, void*);
  void Dump(DebugWriter* writerfn, void* arg) const {}

  int depth() const { return 0; }
  void* const* stack() const { return stack_; }

 private:
  void* stack_[32];
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_STACKTRACE_H_
