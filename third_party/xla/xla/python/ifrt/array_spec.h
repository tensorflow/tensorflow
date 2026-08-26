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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

class Client;
class AbstractArraySpec;

// Specification of an array that groups the static properties of an `Array`
// together. Typically used for describing expected or requested static
// properties of an input/output array of an operation.
class ArraySpec {
 public:
  ArraySpec(DType dtype, Shape shape, ShardingRef sharding,
            absl_nullable std::shared_ptr<const xla::PjRtLayout> layout);

  ArraySpec(const ArraySpec&) = default;
  ArraySpec& operator=(const ArraySpec&) = default;
  ArraySpec(ArraySpec&&) = default;
  ArraySpec& operator=(ArraySpec&&) = default;

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const ShardingRef& sharding() const { return sharding_; }
  const absl_nullable std::shared_ptr<const xla::PjRtLayout>& layout() const {
    return layout_;
  }

  bool operator==(const ArraySpec& other) const {
    auto are_pointees_equal = [](auto* lhs, auto* rhs) {
      if (lhs == nullptr || rhs == nullptr) {
        return lhs == nullptr && rhs == nullptr;
      }
      return lhs == rhs || *lhs == *rhs;
    };
    return dtype_ == other.dtype_ && shape_ == other.shape_ &&
           are_pointees_equal(sharding_.get(), other.sharding_.get()) &&
           are_pointees_equal(layout_.get(), other.layout_.get());
  }

  bool operator!=(const ArraySpec& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const ArraySpec& value) {
    h = H::combine(std::move(h), value.dtype_, value.shape_);
    // The current implementation gracefully handles null sharding even if it's
    // invalid (see `absl_nonnull` annotation) since we don't enforce such
    // properties at ArraySpec creation time. Once we have a constructor that
    // crashes with a null sharding, we can remove this null check.
    if (value.sharding_ != nullptr) {
      h = H::combine(std::move(h), *value.sharding_);
    }
    if (value.layout_ != nullptr) {
      h = H::combine(std::move(h), *value.layout_);
    }
    return h;
  }

  // Constructs `ArraySpec` from `ArraySpecProto`.
  static absl::StatusOr<ArraySpec> FromProto(Client* client,
                                             const ArraySpecProto& proto);

  // Converts the array spec to a protobuf.
  absl::Status ToProto(
      ArraySpecProto& proto,
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  // Returns a `ArraySpecProto` representation.
  absl::StatusOr<ArraySpecProto> ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const {
    ArraySpecProto proto;
    ABSL_RETURN_IF_ERROR(ToProto(proto, version));
    return proto;
  }

  // Converts this array spec to an `AbstractArraySpec`.
  absl::StatusOr<AbstractArraySpec> ToAbstractArraySpec() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ArraySpec& array_spec) {
    sink.Append(absl::StrCat(
        "ArraySpec(dtype=", array_spec.dtype_, ",shape=", array_spec.shape_,
        ",sharding=", array_spec.sharding_, ",layout=",
        (array_spec.layout_ != nullptr ? array_spec.layout_->ToString()
                                       : "<nullptr>"),
        ")"));
  }

 private:
  DType dtype_;
  Shape shape_;
  ShardingRef sharding_;
  absl_nullable std::shared_ptr<const xla::PjRtLayout> layout_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ARRAY_SPEC_H_
