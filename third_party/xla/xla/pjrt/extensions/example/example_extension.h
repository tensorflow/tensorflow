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

#ifndef XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_H_

#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"

#define PJRT_API_EXAMPLE_EXTENSION_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

struct PJRT_ExampleExtension_ExampleMethod_Args;
struct PJRT_ExampleExtension_CreateExampleExtensionCpp_Args;

// Create and destroy share the same args struct in this case.
typedef struct PJRT_ExampleExtension_CreateExampleExtensionCpp_Args
    PJRT_ExampleExtension_DestroyExampleExtensionCpp_Args;

typedef struct PJRT_ExampleExtensionCpp PJRT_ExampleExtensionCpp;

typedef struct PJRT_Example_Extension {
  PJRT_Extension_Base base;

  // Example method.
  PJRT_Error* (*example_method)(PJRT_ExampleExtension_ExampleMethod_Args* args);
  PJRT_Error* (*create)(
      PJRT_ExampleExtension_CreateExampleExtensionCpp_Args* args);
  PJRT_Error* (*destroy)(
      PJRT_ExampleExtension_DestroyExampleExtensionCpp_Args* args);
} PJRT_Example_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Example_Extension, destroy);

typedef struct PJRT_ExampleExtension_ExampleMethod_Args {
  PJRT_ExampleExtensionCpp*
      extension_cpp;  // Input: nullptr if not using CPP impl
  int64_t value;      // Input: Value to pass to ExampleMethod.
} PJRT_ExampleExtension_ExampleMethod_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExampleExtension_ExampleMethod_Args, value);

// Create and destroy are only necessary if the plugin author is using the CPP
// backed implementation. If the plugin author is only using the C API, then
// they can ignore these, and the extension author can provide a simpler C API.

// Create an object of the CPP extension, wrap it in the opaque C type
typedef struct PJRT_ExampleExtension_CreateExampleExtensionCpp_Args {
  PJRT_ExampleExtensionCpp* extension_cpp;
} PJRT_ExampleExtension_CreateExampleExtensionCpp_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExampleExtension_CreateExampleExtensionCpp_Args,
                          extension_cpp);

typedef PJRT_Error*(
    PJRT_CreateExampleExtensionCpp_Fn)(PJRT_ExampleExtension_CreateExampleExtensionCpp_Args*);  // NOLINT(whitespace/line_length)

typedef PJRT_Error*(
    PJRT_DestroyExampleExtensionCpp_Fn)(PJRT_ExampleExtension_DestroyExampleExtensionCpp_Args*);  // NOLINT(whitespace/line_length)

#ifdef __cplusplus
}
#endif

namespace pjrt {

PJRT_Example_Extension CreateExampleExtension(
    PJRT_Extension_Base* next, PJRT_CreateExampleExtensionCpp_Fn create,
    PJRT_DestroyExampleExtensionCpp_Fn destroy);

}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_H_
