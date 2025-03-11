// Copyright 2025 Google LLC.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_COMPILATION_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_COMPILATION_OPTIONS_H_

#include <string>

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"

// This must be the very first field (or base) of every accelerator option
// object in order to make the option objects part of a list.
//
// NOTE: fields can only be appended to the end of the struct and existing
// fields cannot be removed, only renamed or deprecated.
struct LiteRtAcceleratorCompilationOptionsHeader {
  // NOTE: THIS SHOULD BE INCREMENTED EVERY TIME A NEW FIELD IS ADDED TO THIS
  // STRUCT.
  static constexpr int kVersion = 1;

  // Note: this should be always set to kVersion.
  const int version = kVersion;

  // Pointer to the next link structure.
  LiteRtAcceleratorCompilationOptionsHeader* next = nullptr;

  // Identifier for the configuration structure. Used by the accelerator
  // implementation to go through the list and reinterpret the link to its
  // actual type.
  std::string identifier;

  // NOLINTBEGIN(*-readability-class-member-naming)

  // A destructor for this link's data. Releases the memory stored in this link
  // AS WELL AS THE LINK ITSELF.
  //
  // We need this for option objects that may own some of their data. The most
  // common use case here being helper functions that build a path from other
  // program inputs. If the options structure doesn't own the data, then the
  // user must ensure that the string outlives the compiled model, which may be
  // tricky. This let's the user define a function that will be called to clean
  // up the data.
  void (*destructor)(LiteRtAcceleratorCompilationOptionsHeader*) = nullptr;

  // NOLINTEND(*-readability-class-member-naming)
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_COMPILATION_OPTIONS_H_
