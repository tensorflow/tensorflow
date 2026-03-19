/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_HAS_KEYWORD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_HAS_KEYWORD_H_

// CAUTION: __is_identifier behaves opposite how you would expect!
// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by
// the compiler and '1' otherwise.  borrowed from LLVM __config header under
// Apache license 2.
// (https://www.mend.io/blog/top-10-apache-license-questions-answered/)

#ifndef __is_identifier       // Optional of course.
#define __is_identifier(x) 1  // Compatibility with non-clang compilers.
#endif

// More sensible macro for keyword detection
#define __has_keyword(__x) !(__is_identifier(__x))

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_HAS_KEYWORD_H_
