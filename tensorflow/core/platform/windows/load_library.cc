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

#include "tensorflow/core/platform/load_library.h"

#include <Shlwapi.h>
#undef StrCat  // Don't let StrCat be renamed to lstrcatA
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#undef ERROR

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/windows/wide_char.h"

#pragma comment(lib, "Shlwapi.lib")

namespace tensorflow {

namespace internal {

Status LoadDynamicLibrary(const char* library_filename, void** handle) {
  string file_name = library_filename;
  std::replace(file_name.begin(), file_name.end(), '/', '\\');

  std::wstring ws_file_name(Utf8ToWideChar(file_name));

  HMODULE hModule =
      LoadLibraryExW(ws_file_name.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!hModule) {
    return errors::NotFound(file_name + " not found");
  }
  *handle = hModule;
  return Status::OK();
}

Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol) {
  FARPROC found_symbol;

  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);
  if (found_symbol == NULL) {
    return errors::NotFound(std::string(symbol_name) + " not found");
  }
  *symbol = (void**)found_symbol;
  return Status::OK();
}

string FormatLibraryFileName(const string& name, const string& version) {
  string filename;
  if (version.size() == 0) {
    filename = name + ".dll";
  } else {
    filename = name + version + ".dll";
  }
  return filename;
}

}  // namespace internal

}  // namespace tensorflow
