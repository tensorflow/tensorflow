/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/windows/wide_char.h"

#include <Windows.h>

#include <string>

namespace tensorflow {

std::wstring Utf8ToWideChar(const std::string& utf8str) {
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(),
                                          (int)utf8str.size(), NULL, 0);
  std::wstring ws_translated_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(),
                      &ws_translated_str[0], size_required);
  return ws_translated_str;
}

std::string WideCharToUtf8(const std::wstring& wstr) {
  if (wstr.empty()) return std::string();
  int size_required = WideCharToMultiByte(
      CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
  std::string utf8_translated_str(size_required, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(),
                      &utf8_translated_str[0], size_required, NULL, NULL);
  return utf8_translated_str;
}

}  // namespace tensorflow
