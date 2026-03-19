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

#ifndef XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_PRIVATE_H_
#define XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_PRIVATE_H_

#include "xla/pjrt/extensions/example/example_extension_cpp.h"

// Note: this is a private header, and should not be included by any caller
// code. This is intended to be opaque type to callers, since this pointer *can
// not* be dereferenced directly - it is on the other side of the C API.

typedef struct PJRT_ExampleExtensionCpp {
  // C++ implementation of the extension.
  xla::ExampleExtensionCpp* extension_cpp;
} PJRT_ExampleExtensionCpp;

#endif  // XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_PRIVATE_H_
