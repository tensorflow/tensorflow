/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/plugin/example_plugin/myplugin_c_pjrt.h"

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/extensions/example/example_extension.h"
#include "xla/pjrt/extensions/example/example_extension_cpp.h"
#include "xla/pjrt/extensions/example/example_extension_private.h"
#include "xla/pjrt/plugin/example_plugin/example_extension_impl.h"
#include "xla/pjrt/plugin/testing/testing_c_pjrt_internal.h"

namespace myplugin_pjrt {

PJRT_Error* PJRT_MyPlugin_CreateExampleExtensionCpp(
    PJRT_ExampleExtension_CreateExampleExtensionCpp_Args* args) {
  xla::ExampleExtensionCpp* example_extension_cpp =
      new xla::ExampleExtensionImpl("standard_prefix: ", "myplugin_prefix: ");
  PJRT_ExampleExtensionCpp* opaque_extension_cpp = new PJRT_ExampleExtensionCpp{
      example_extension_cpp,
  };
  args->extension_cpp = opaque_extension_cpp;
  return nullptr;
}

PJRT_Error* PJRT_MyPlugin_DestroyExampleExtensionCpp(
    PJRT_ExampleExtension_DestroyExampleExtensionCpp_Args* args) {
  delete args->extension_cpp->extension_cpp;
  delete args->extension_cpp;
  return nullptr;
}

}  // namespace myplugin_pjrt

const PJRT_Api* GetPjrtApi() {
  static PJRT_Example_Extension example_extension =
      pjrt::CreateExampleExtension(
          nullptr, myplugin_pjrt::PJRT_MyPlugin_CreateExampleExtensionCpp,
          myplugin_pjrt::PJRT_MyPlugin_DestroyExampleExtensionCpp);

  return testing::GetTestingPjrtApi(&example_extension.base);
}
