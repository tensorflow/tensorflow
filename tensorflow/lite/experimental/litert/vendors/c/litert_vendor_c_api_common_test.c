// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// This file exists to verify that the below header files can build, link,
// and run as C code.
#ifdef __cplusplus
#error "This file should be compiled as C code, not as C++."
#endif

// Include all the header files in the litert/c directory.
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"  // NOLINT
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin_api.h"  // NOLINT
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"  // NOLINT
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch_api.h"  // NOLINT

int main(void) {
  return 0;
}
