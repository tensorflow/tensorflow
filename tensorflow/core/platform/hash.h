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

// Simple hash functions used for internal data structures

#ifndef TENSORFLOW_CORE_PLATFORM_HASH_H_
#define TENSORFLOW_CORE_PLATFORM_HASH_H_

#include "tsl/platform/hash.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::hash;
using ::tsl::Hash32;
using ::tsl::Hash64;
using ::tsl::Hash64Combine;
using ::tsl::Hash64CombineUnordered;
using ::tsl::StringPieceHasher;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow


#endif  // TENSORFLOW_CORE_PLATFORM_HASH_H_
