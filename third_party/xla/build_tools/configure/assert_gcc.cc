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

// Notably, clang will define `__GNUC__`, so need to make sure __clang__ is not
// defined to detect GCC (or, most correctly, some compiler that supports GNU
// extensions that is not clang).
#if !defined(__GNUC__) || defined(__clang__)
#error "__GNUC__ is not defined independently of __clang__!"
#endif  // #if !defined(__GNUC__) || defined(__clang__)
