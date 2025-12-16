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

#ifndef XLA_SERVICE_HLO_CSE_CONSTANT_KEY_H_
#define XLA_SERVICE_HLO_CSE_CONSTANT_KEY_H_

#include <cstdint>
#include <utility>

#include "absl/base/attributes.h"
#include "xla/literal.h"
#include "xla/shape.h"

namespace xla {
template <bool kIsLayoutSensitive>
struct CseConstantKey {
  template <typename H>
  friend H AbslHashValue(H h, const CseConstantKey& key) {
    h = H::combine(std::move(h), key.domain);
    h = Shape::Hash<H, kIsLayoutSensitive>(std::move(h), key.shape);
    return Literal::Hash<H, kIsLayoutSensitive, /*kByteLimit=*/64>(std::move(h),
                                                                   key.literal);
  }
  friend bool operator==(const CseConstantKey& lhs, const CseConstantKey& rhs) {
    return lhs.domain == rhs.domain &&
           (kIsLayoutSensitive
                ? Shape::Equal()
                : Shape::Equal().IgnoreLayout())(lhs.shape, rhs.shape) &&
           lhs.literal.Equal(rhs.literal, kIsLayoutSensitive);
  }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CseConstantKey& key) {
    absl::Format(&sink, "literal: %s, shape: %s, domain: %d",
                 key.literal.ToString(), key.shape.ToString(kIsLayoutSensitive),
                 key.domain);
  }

  const Literal& literal ABSL_REQUIRE_EXPLICIT_INIT;
  const Shape& shape ABSL_REQUIRE_EXPLICIT_INIT;
  int64_t domain ABSL_REQUIRE_EXPLICIT_INIT;
};
}  // namespace xla

#endif  // XLA_SERVICE_HLO_CSE_CONSTANT_KEY_H_
