/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
namespace shlo_ref {
using F16 = ::std::float16_t;
}  // namespace shlo_ref

#else
namespace shlo_ref {
using F16 = _Float16;
}  // namespace shlo_ref

#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_F16_H_
