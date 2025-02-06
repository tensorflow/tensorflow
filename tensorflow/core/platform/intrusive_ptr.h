/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PLATFORM_INTRUSIVE_PTR_H_
#define TENSORFLOW_CORE_PLATFORM_INTRUSIVE_PTR_H_

#include <algorithm>

#include "xla/tsl/platform/intrusive_ptr.h"

namespace tensorflow {
namespace core {

template <class T>
using IntrusivePtr = tsl::core::IntrusivePtr<T>;

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_INTRUSIVE_PTR_H_
