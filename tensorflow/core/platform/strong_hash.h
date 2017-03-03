/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_STRONG_HASH_H_
#define TENSORFLOW_CORE_PLATFORM_STRONG_HASH_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// This is a strong keyed hash function interface for strings.
// The hash function is deterministic on the content of the string within the
// process. The key of the hash is an array of 2 uint64 elements.
// A strong hash make it dificult, if not infeasible, to compute inputs that
// hash to the same bucket.
//
// Usage:
//   uint64 key[2] = {123, 456};
//   string input = "input string";
//   uint64 hash_value = StrongKeyedHash(key, input);
//
uint64 StrongKeyedHash(const uint64 (&)[2], const string&);

}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/strong_hash.h"
#else
#include "tensorflow/core/platform/default/strong_hash.h"
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_STRONG_HASH_H_
