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

#ifndef TENSORFLOW_TSL_PLATFORM_DYNAMIC_ANNOTATIONS_H_
#define TENSORFLOW_TSL_PLATFORM_DYNAMIC_ANNOTATIONS_H_

#include "absl/base/dynamic_annotations.h"

#define TF_ANNOTATE_MEMORY_IS_INITIALIZED(ptr, bytes) \
  ANNOTATE_MEMORY_IS_INITIALIZED(ptr, bytes)

#define TF_ANNOTATE_BENIGN_RACE(ptr, description) \
  ANNOTATE_BENIGN_RACE(ptr, description)

// Tell MemorySanitizer to relax the handling of a given function. All "Use of
// uninitialized value" warnings from such functions will be suppressed, and
// all values loaded from memory will be considered fully initialized.
#ifdef MEMORY_SANITIZER
#define TF_ATTRIBUTE_NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))
#else
#define TF_ATTRIBUTE_NO_SANITIZE_MEMORY
#endif

#endif  // TENSORFLOW_TSL_PLATFORM_DYNAMIC_ANNOTATIONS_H_
