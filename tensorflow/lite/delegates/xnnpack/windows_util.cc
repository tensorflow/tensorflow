/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/windows_util.h"

#if defined(_MSC_VER)

#include <windows.h>

#include <cctype>
#include <cstddef>
#include <string>

namespace tflite::xnnpack {

// Returns a string holding the error message corresponding to the code returned
// by `GetLastError()`.
std::string GetLastErrorString() {
  const DWORD error = GetLastError();
  LPSTR message_buffer = nullptr;
  const DWORD chars_written = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&message_buffer), 0, NULL);
  if (chars_written > 0 && message_buffer != nullptr) {
    std::string error_message = message_buffer;
    LocalFree(message_buffer);
    // Remove trailing whitespace
    while (!error_message.empty() && std::isspace(error_message.back())) {
      error_message.pop_back();
    }
    return error_message;
  }
  // https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes#system-error-codes
  return std::to_string(error);
}

}  // namespace tflite::xnnpack

#endif
