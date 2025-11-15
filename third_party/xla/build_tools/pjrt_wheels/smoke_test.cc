/* Copyright 2025 The OpenXLA Authors.

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

#include <dlfcn.h>

#include <iostream>

#include "xla/pjrt/c/pjrt_c_api.h"

typedef const PJRT_Api* (*GetPjrtApi_Func)();

int main() {
  // 1. Open the shared object
  const char* so_path = std::getenv("PJRT_PLUGIN_PATH");
  std::cout << "so_path: " << so_path << std::endl;
  void* handle = dlopen(so_path, RTLD_LAZY);
  if (!handle) {
    std::cerr << "Error: Could not open shared object." << std::endl;
    std::cerr << "Reason: " << dlerror() << std::endl;
    return 1;
  }

  // 2. Load the symbol (the function)
  GetPjrtApi_Func get_pjrt_api = (GetPjrtApi_Func)dlsym(handle, "GetPjrtApi");
  const char* dlsym_error = dlerror();
  if (dlsym_error) {
    std::cerr << "Error: Could not find symbol 'GetPjrtApi'." << std::endl;
    std::cerr << "Reason: " << dlsym_error << std::endl;
    dlclose(handle);
    return 1;
  }

  // 3. Call the function
  std::cout << "Successfully loaded symbol. Calling GetPjrtApi()..."
            << std::endl;
  const PJRT_Api* api = get_pjrt_api();
  if (api) {
    std::cout << "Success! Received PjrtApi struct pointer." << std::endl;
  } else {
    std::cerr << "Error: GetPjrtApi() returned a null pointer." << std::endl;
    return 1;
  }

  return 0;
}
