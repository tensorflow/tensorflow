/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_VALUE_UTIL_H_
#define XLA_PYTHON_IFRT_VALUE_UTIL_H_

#include <vector>

#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/value.h"

namespace xla {
namespace ifrt {

// Eagerly converts a span of `ValueRef` to a vector of `ArrayRef` by copying
// references.
//
// REQUIRES: All values are arrays.
std::vector<ArrayRef> ToArrays(absl::Span<const ValueRef> values);

// Eagerly converts a span of `ValueRef` to a vector of `ArrayRef` by moving
// references.
//
// REQUIRES: All values are arrays.
std::vector<ArrayRef> ToArrays(absl::Span<ValueRef> values);

// Upcasts `ArrayRef`s to `ValueRef`s by copying references.
std::vector<ValueRef> ToValues(absl::Span<const ArrayRef> arrays);

// Upcasts `ArrayRef`s to `ValueRef`s by moving references.
std::vector<ValueRef> ToValues(absl::Span<ArrayRef> arrays);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_VALUE_UTIL_H_
