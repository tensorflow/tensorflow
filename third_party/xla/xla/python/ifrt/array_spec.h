/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_ARRAY_SPEC_H_
#define XLA_PYTHON_IFRT_ARRAY_SPEC_H_

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

class Client;

// Specification of an array that groups the static properties of an `Array`
// together. Typically used for describing expected or requested static
// properties of an input/output array of an operation.
struct ArraySpec {
  DType dtype;
  Shape shape;
  ShardingRef sharding;
  absl_nullable std::shared_ptr<const xla::PjRtLayout> layout;

  bool operator==(const ArraySpec& other) const {
    auto are_pointees_equal = [](auto* lhs, auto* rhs) {
      if (lhs == nullptr || rhs == nullptr) {
        return lhs == nullptr && rhs == nullptr;
      }
      return lhs == rhs || *lhs == *rhs;
    };
    return dtype == other.dtype && shape == other.shape &&
           are_pointees_equal(sharding.get(), other.sharding.get()) &&
           are_pointees_equal(layout.get(), other.layout.get());
  }

  bool operator!=(const ArraySpec& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const ArraySpec& value) {
    h = H::combine(std::move(h), value.dtype, value.shape);
    // The current implementation gracefully handles null sharding even if it's
    // invalid (see `absl_nonnull` annotation) since we don't enforce such
    // properties at ArraySpec creation time. Once we have a constructor that
    // crashes with a null sharding, we can remove this null check.
    if (value.sharding != nullptr) {
      h = H::combine(std::move(h), *value.sharding);
    }
    if (value.layout != nullptr) {
      h = H::combine(std::move(h), *value.layout);
    }
    return h;
  }

  // Constructs `ArraySpec` from `ArraySpecProto`.
  static absl::StatusOr<ArraySpec> FromProto(Client* client,
                                             const ArraySpecProto& proto);

  // Returns a `ArraySpecProto` representation.
  absl::StatusOr<ArraySpecProto> ToProto() const;

  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  std::string DebugString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ArraySpec& array_spec) {
    sink.Append(array_spec.DebugString());
  }
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ARRAY_SPEC_H_
