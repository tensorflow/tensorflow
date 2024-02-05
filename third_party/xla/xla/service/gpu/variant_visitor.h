/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_VARIANT_VISITOR_H_
#define XLA_SERVICE_GPU_VARIANT_VISITOR_H_

namespace xla::gpu {
// This structure is used to support C++17 overload pattern as described in
// https://en.cppreference.com/w/cpp/utility/variant/visit
//
// TODO(b/319202112): Replace with absl::Overload once abs lts_2024_XXX is
// tagged.
template <class... Ts>
struct VariantVisitor : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
VariantVisitor(Ts...) -> VariantVisitor<Ts...>;

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_VARIANT_VISITOR_H_
