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

#ifndef XLA_PYTHON_IFRT_ABSTRACT_ARRAY_SPEC_H_
#define XLA_PYTHON_IFRT_ABSTRACT_ARRAY_SPEC_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/abstract_array_spec.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"

namespace xla {
namespace ifrt {

struct ArraySpec;
class ArraySpecProto;

class AbstractArraySpec {
 public:
  // Creates an `AbstractArraySpec` from the given properties.
  static absl::StatusOr<AbstractArraySpec> Create(
      DType dtype, Shape shape, ShardingSpecRef sharding_spec,
      MemoryKind memory_kind,
      absl_nullable std::shared_ptr<const xla::PjRtLayout> layout);

  AbstractArraySpec() = delete;
  AbstractArraySpec(const AbstractArraySpec&) = default;
  AbstractArraySpec& operator=(const AbstractArraySpec&) = default;
  AbstractArraySpec(AbstractArraySpec&&) = default;
  AbstractArraySpec& operator=(AbstractArraySpec&&) = default;

  DType dtype() const { return rep_->dtype; }
  const Shape& shape() const { return rep_->shape; }
  const ShardingSpecRef& sharding_spec() const { return rep_->sharding_spec; }
  MemoryKind memory_kind() const { return rep_->memory_kind; }
  const absl_nullable std::shared_ptr<const xla::PjRtLayout>& layout() const {
    return rep_->layout;
  }

  bool operator==(const AbstractArraySpec& other) const;
  bool operator!=(const AbstractArraySpec& other) const {
    return !(*this == other);
  }

  template <typename H>
  friend H AbslHashValue(H h, const AbstractArraySpec& value) {
    return H::combine(std::move(h), value.Hash());
  }

  // Converts this `AbstractArraySpec` into an `ArraySpec` with the given
  // devices.
  absl::StatusOr<ArraySpec> ToArraySpec(DeviceListRef devices) const;

  // Deserializes an `AbstractArraySpecProto` into an `AbstractArraySpec`.
  static absl::StatusOr<AbstractArraySpec> FromProto(
      const AbstractArraySpecProto& proto);

  // Serializes this `AbstractArraySpec` into an `AbstractArraySpecProto`.
  absl::Status ToProto(AbstractArraySpecProto& proto,
                       SerDesVersion version = SerDesVersion::current()) const;

  // Serializes this `AbstractArraySpec` into an `AbstractArraySpecProto`.
  absl::StatusOr<AbstractArraySpecProto> ToProto(
      SerDesVersion version = SerDesVersion::current()) const {
    AbstractArraySpecProto proto;
    ABSL_RETURN_IF_ERROR(ToProto(proto, version));
    return proto;
  }

  std::string DebugString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const AbstractArraySpec& spec) {
    sink.Append(spec.DebugString());
  }

 private:
  struct Rep {
    DType dtype;
    Shape shape;
    ShardingSpecRef sharding_spec;
    MemoryKind memory_kind;
    absl_nullable std::shared_ptr<const xla::PjRtLayout> layout;

    static constexpr uint64_t kUnsetHash = 0;
    mutable std::atomic<uint64_t> hash{kUnsetHash};

    Rep(DType dtype, Shape shape, ShardingSpecRef sharding_spec,
        MemoryKind memory_kind,
        absl_nullable std::shared_ptr<const xla::PjRtLayout> layout);
    Rep(const Rep&) = delete;
    Rep& operator=(const Rep&) = delete;
  };

  explicit AbstractArraySpec(
      DType dtype, Shape shape, ShardingSpecRef sharding_spec,
      MemoryKind memory_kind,
      absl_nullable std::shared_ptr<const xla::PjRtLayout> layout);

  uint64_t Hash() const;

  absl_nonnull std::shared_ptr<const Rep> rep_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ABSTRACT_ARRAY_SPEC_H_
