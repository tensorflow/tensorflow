/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Traits classes for performing uniform lookup on different map value types.
//
// The access is computed as follows:
//
//   1. If T has a `first` or `second` field, use them.
//   2. Otherwise if it has `key()` or `value()` methods, use them.
//   3. Otherwise the program is ill-formed.
#ifndef TENSORFLOW_CORE_LIB_GTL_SUBTLE_MAP_TRAITS_H_
#define TENSORFLOW_CORE_LIB_GTL_SUBTLE_MAP_TRAITS_H_

#include "xla/tsl/lib/gtl/subtle/map_traits.h"

namespace tensorflow {
namespace gtl {
namespace subtle {
namespace internal_map_traits {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::gtl::subtle::internal_map_traits::GetKey;
using ::tsl::gtl::subtle::internal_map_traits::GetMapped;
using ::tsl::gtl::subtle::internal_map_traits::Rank0;
using ::tsl::gtl::subtle::internal_map_traits::Rank1;
// NOLINTEND(misc-unused-using-decls)

}  // namespace internal_map_traits
}  // namespace subtle
}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_SUBTLE_MAP_TRAITS_H_
