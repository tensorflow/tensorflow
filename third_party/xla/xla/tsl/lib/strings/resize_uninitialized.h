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

#ifndef XLA_TSL_LIB_STRINGS_RESIZE_UNINITIALIZED_H_
#define XLA_TSL_LIB_STRINGS_RESIZE_UNINITIALIZED_H_

#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/meta/type_traits.h"

namespace tsl {

// Helpers for STLStringResizeUninitialized
// HasMember is true_type or false_type, depending on whether or not
// T has a __resize_default_init member. Resize will call the
// __resize_default_init member if it exists, and will call the resize
// member otherwise.
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type* s, size_t new_size) { s->resize(new_size); }
};

// __resize_default_init is provided by libc++ >= 8.0.
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type&>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type* s, size_t new_size) {
    s->__resize_default_init(new_size);
  }
};

// Resize string `s` to `new_size`, leaving the data uninitialized.
inline void STLStringResizeUninitialized(std::string* s, size_t new_size) {
  ResizeUninitializedTraits<std::string>::Resize(s, new_size);
}

}  // namespace tsl

#endif  // XLA_TSL_LIB_STRINGS_RESIZE_UNINITIALIZED_H_
