/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_GTL_OPTIONAL_H_
#define TENSORFLOW_CORE_LIB_GTL_OPTIONAL_H_

#include "absl/types/optional.h"

namespace tensorflow {
namespace gtl {

// Deprecated: please use absl::optional directly.
using absl::make_optional;
using absl::nullopt;
template <typename T>
using optional = absl::optional<T>;

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_OPTIONAL_H_
