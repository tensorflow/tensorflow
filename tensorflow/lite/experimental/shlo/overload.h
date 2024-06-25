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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_

namespace shlo_ref {

// Returns a functor that provides overloads based on the
// functors passed to it.
//
// Useful when used in conjunction with `std::visit`.
//
// Use absl version when we know for sure the version we can use.
template <class... Ts>
class Overload : public Ts... {
 public:
  explicit Overload(Ts&&... ts) : Ts(static_cast<Ts&&>(ts))... {}
  using Ts::operator()...;
};

template <class... Ts>
Overload(Ts&&...) -> Overload<Ts...>;

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_
