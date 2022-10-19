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

// This provides a very simple, boring adaptor for a begin and end iterator
// into a range type. This should be used to build range views that work well
// with range based for loops and range based constructors.
//
// Note that code here follows more standards-based coding conventions as it
// is mirroring proposed interfaces for standardization.
//
// Converted from chandlerc@'s code to Google style by joshl@.

#ifndef TENSORFLOW_CORE_LIB_GTL_ITERATOR_RANGE_H_
#define TENSORFLOW_CORE_LIB_GTL_ITERATOR_RANGE_H_

#include "tensorflow/tsl/lib/gtl/iterator_range.h"

namespace tensorflow {
namespace gtl {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::gtl::iterator_range;
using ::tsl::gtl::make_range;
// NOLINTEND(misc-unused-using-decls)
}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_ITERATOR_RANGE_H_
