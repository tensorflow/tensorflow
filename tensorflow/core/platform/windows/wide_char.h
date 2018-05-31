/* Copyright 2018 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_WINDOWS_WIDE_CHAR_H_
#define TENSORFLOW_CORE_PLATFORM_WINDOWS_WIDE_CHAR_H_

#include <Windows.h>
#include <string>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

inline std::wstring Utf8ToWideChar(const string& utf8str) {
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(),
                                          (int)utf8str.size(), NULL, 0);
  std::wstring ws_translated_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(),
                      &ws_translated_str[0], size_required);
  return ws_translated_str;
}

inline string WideCharToUtf8(const std::wstring& wstr) {
  if (wstr.empty()) return std::string();
  int size_required = WideCharToMultiByte(
      CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
  string utf8_translated_str(size_required, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(),
                      &utf8_translated_str[0], size_required, NULL, NULL);
  return utf8_translated_str;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_WINDOWS_WIDE_CHAR_H_
