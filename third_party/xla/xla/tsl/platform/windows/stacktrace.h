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

#ifndef XLA_TSL_PLATFORM_WINDOWS_STACKTRACE_H_
#define XLA_TSL_PLATFORM_WINDOWS_STACKTRACE_H_

#include <string>

#define TF_HAS_STACKTRACE

namespace tsl {

// Function to create a pretty stacktrace.
std::string CurrentStackTrace();

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

#endif  // XLA_TSL_PLATFORM_WINDOWS_STACKTRACE_H_
