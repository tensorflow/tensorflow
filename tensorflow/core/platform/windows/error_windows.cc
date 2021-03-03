/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/windows/error_windows.h"

#include <Windows.h>
#include <Winsock2.h>

#include <string>

namespace tensorflow {
namespace internal {
namespace {

std::string GetWindowsErrorMessage(DWORD err) {
  LPSTR buffer = NULL;
  DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS;
  DWORD length = FormatMessageA(flags, NULL, err,
                                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                reinterpret_cast<LPSTR>(&buffer), 0, NULL);
  std::string message(buffer, length);
  LocalFree(buffer);
  if (length == 0) {
    message = "Failed to FormatMessage for error";
  }
  return message;
}

}  // namespace

std::string WindowsGetLastErrorMessage() {
  return GetWindowsErrorMessage(::GetLastError());
}

std::string WindowsWSAGetLastErrorMessage() {
  return GetWindowsErrorMessage(::WSAGetLastError());
}

}  // namespace internal
}  // namespace tensorflow
