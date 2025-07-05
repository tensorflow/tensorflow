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

#include "xla/pjrt/extensions/example/example_extension.h"

#include <iostream>

#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/extensions/example/example_extension_cpp.h"

PJRT_Error* PJRT_ExampleExtension_ExampleMethod(
    PJRT_ExampleExtension_ExampleMethod_Args* args) {
  auto status = args->extension->extension_cpp->ExampleMethod(args->value);
  if (!status.ok()) {
    std::cout << "ExampleMethod failed: " << status << "\n";
  }
  return nullptr;
}

PJRT_Example_Extension pjrt::CreateExampleExtension(
    PJRT_Extension_Base* next, xla::ExampleExtensionCpp* extension_cpp) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Example_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Unknown,
          /*next=*/next,
      },
      /*extension_cpp=*/
      extension_cpp,
      /*PJRT_ExampleExtension_ExampleMethod=*/
      PJRT_ExampleExtension_ExampleMethod,
  };
}
