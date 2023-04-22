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

// Printf variants that place their output in a C++ string.
//
// Usage:
//      string result = strings::Printf("%d %s\n", 10, "hello");
//      strings::SPrintf(&result, "%d %s\n", 10, "hello");
//      strings::Appendf(&result, "%d %s\n", 20, "there");

#ifndef TENSORFLOW_CORE_LIB_STRINGS_STRINGPRINTF_H_
#define TENSORFLOW_CORE_LIB_STRINGS_STRINGPRINTF_H_

#include "tensorflow/core/platform/stringprintf.h"

#endif  // TENSORFLOW_CORE_LIB_STRINGS_STRINGPRINTF_H_
