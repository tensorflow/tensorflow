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

// This file provides utility functions for use with STL map-like data
// structures, such as std::map and hash_map. Some functions will also work with
// sets, such as ContainsKey().

#ifndef TENSORFLOW_CORE_LIB_GTL_MAP_UTIL_H_
#define TENSORFLOW_CORE_LIB_GTL_MAP_UTIL_H_

#include "tensorflow/tsl/lib/gtl/map_util.h"

namespace tensorflow {
namespace gtl {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::gtl::EraseKeyReturnValuePtr;
using ::tsl::gtl::FindOrNull;
using ::tsl::gtl::FindPtrOrNull;
using ::tsl::gtl::FindWithDefault;
using ::tsl::gtl::InsertIfNotPresent;
using ::tsl::gtl::InsertOrUpdate;
using ::tsl::gtl::LookupOrInsert;
using ::tsl::gtl::ReverseMap;
// NOLINTEND(misc-unused-using-decls)
}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_MAP_UTIL_H_
