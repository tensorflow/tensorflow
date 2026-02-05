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

#ifndef XLA_PYTHON_IFRT_SHARDING_SPEC_H_
#define XLA_PYTHON_IFRT_SHARDING_SPEC_H_

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.pb.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ifrt {

class ShardingSpec;

using ShardingSpecRef = absl_nonnull std::shared_ptr<const ShardingSpec>;

// `ShardingSpec` represents partitioning of a logical array into a certain
// ordered list of shards.
//
// Shards may contain duplicate or exclusive data, depending on the partitioning
// scheme. One well-known case of "fully replicated" sharding spec indicates
// that every shard contains the entire data of the logical array.
//
// Shards are order-sensitive. The ordering of individual shards in a sharding
// spec must be respected, in particular when the list of the shards is used in
// parallel to another order-sensitive list, such as `DeviceList`. For example,
// the first shard from `ShardingSpec` would be placed on the first device from
// `DeviceList`, the second shard from `ShardingSpec` would be placed on the
// second device from `DeviceList`, and so on.
//
// Depending on the type of a sharding spec, the partitioning of the sharding
// spec is applicable only to a particular array shape, but that of some others
// may be applicable to a broad set of shapes.
class ShardingSpec : public llvm::RTTIExtends<ShardingSpec, Serializable> {
 public:
  // Returns the number of shards.
  int num_shards() const { return num_shards_; }

  // Returns if this sharding spec is fully replicated. A fully replicated
  // sharding spec means that the logical shape and shard shapes are identical
  // (`GetShardShape(shape) == shape`), and every shard of the array contains
  // the entire data of the logical array.
  bool IsFullyReplicated() const { return is_fully_replicated_; }

  // Returns if this sharding spec is equal to `other`.
  bool operator==(const ShardingSpec& other) const;
  bool operator!=(const ShardingSpec& other) const { return !(*this == other); }

  // Returns a shard shape if the sharding spec always has the equal shape for
  // all shards. Returns an error if the sharding spec may not have a single
  // shard shape, or `shape` is not a valid shape for this sharding spec.
  virtual absl::StatusOr<Shape> GetShardShape(const Shape& shape) const = 0;

  // Returns if this sharding spec has the same logical partitioning as `other`.
  // By the same logical partitioning, we mean that `ShardingSpec` type is the
  // same, and the partitioning scheme within the sharding spec is equivalent.
  // It does not need to check if `Disassemble()` would return the same result.
  virtual bool HasSamePartitioning(const ShardingSpec& other) const = 0;

  // Breaks a shape up into per-device shapes and sharding specs. See
  // Array::DisassembleIntoSingleDeviceArrays(). It may return an error if
  // disassembly is unsupported.
  virtual absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
  Disassemble(const Shape& shape) const = 0;

  // Variant of `Disassemble` that takes a dynamic shape.
  virtual absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const = 0;

  // Maps each shard to an `IndexDomain` over `shape`. The result is a list of
  // `index_domain_i` such that `array[index_domain_i] = disassembled_array_i`.
  // Note that multiple shards may map onto equal `IndexDomain`. For instance, a
  // fully replicated sharding spec would return a vector of
  // `[IndexDomain(shape)] * num_shards()`.
  virtual absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const = 0;

  template <typename H>
  friend H AbslHashValue(H h, const ShardingSpec& value) {
    value.Hash(absl::HashState::Create(&h));
    return std::move(h);
  }

  // Deserializes `ShardingSpecProto` into `ShardingSpec`.
  // Note that `ShardingSpec` serialization uses `SerDes` to handle an open set
  // of `ShardingSpec` subclasses. See `serdes.h`.
  static absl::StatusOr<ShardingSpecRef> FromProto(
      const ShardingSpecProto& sharding_spec_proto);

  // Converts `Sharding` into a protobuf.
  absl::Status ToProto(
      ShardingSpecProto& sharding_spec_proto,
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  // Serializes `ShardingSpec` into `ShardingSpecProto`.
  // Note that `ShardingSpec` serialization uses `SerDes` to handle an open set
  // of `ShardingSpec` subclasses. See `serdes.h`.
  absl::StatusOr<ShardingSpecProto> ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const {
    ShardingSpecProto proto;
    TF_RETURN_IF_ERROR(ToProto(proto, version));
    return proto;
  }

  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  virtual std::string DebugString() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ShardingSpec& sharding_spec) {
    sink.Append(sharding_spec.DebugString());
  }

  // TODO(hyeontaek): Remove this template definition. In theory,
  // `std::shared_ptr<>` is responsible for defining it. Consider introducing a
  // `std::shared_ptr<>` version of `RCReferenceWrapper` to own this template
  // definition.
  template <class Sink>
  friend void AbslStringify(
      Sink& sink, std::shared_ptr<const ShardingSpec>& sharding_spec) {
    if (sharding_spec == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(sharding_spec->DebugString());
    }
  }

  static char ID;  // NOLINT

 protected:
  ShardingSpec(int num_shards, bool is_fully_replicated);

  virtual void Hash(absl::HashState state) const = 0;

  int num_shards_;
  bool is_fully_replicated_;
};

std::ostream& operator<<(std::ostream& os, const ShardingSpec& sharding_spec);

// TODO(hyeontaek): Move the subclasses of `ShardingSpec` to a seperate file,
// making this sharding_spec.{h,cc} only define interface and common functions.

// Single-device sharding spec.
//
// TODO(hyeontaek): `SingleDeviceShardingSpec` tends to be created or consumed
// in a large quantity. It may be useful for performance optimization to
// special-case this sharding type rather than expressing it as a general
// `ShardingSpec`.
class SingleDeviceShardingSpec final
    : public llvm::RTTIExtends<SingleDeviceShardingSpec, ShardingSpec> {
 public:
  // Creates a single-device sharding spec.
  static std::unique_ptr<SingleDeviceShardingSpec> Create();

  // ShardingSpec implementation.

  ~SingleDeviceShardingSpec() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const ShardingSpec& other) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  SingleDeviceShardingSpec();

  void Hash(absl::HashState state) const override;
};

// Opaque sharding spec that does not define a fixed semantics for conversion
// between a logical shape and per-device shapes, and device placements.
class OpaqueShardingSpec
    : public llvm::RTTIExtends<OpaqueShardingSpec, ShardingSpec> {
 public:
  // Creates an opaque sharding spec. `Disassemble()` will fail.
  static std::unique_ptr<OpaqueShardingSpec> Create(int num_shards);

  // ShardingSpec implementation.

  ~OpaqueShardingSpec() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const ShardingSpec& other) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  explicit OpaqueShardingSpec(int num_shards);

  void Hash(absl::HashState state) const override;
};

// Opaque sharding spec that does not define a fixed semantics for conversion
// between a logical shape and shard shapes, and device placements. It can
// disassemble a certain shape into shard shapes that may not be identical. It
// is advised to use `ConcreteEvenShardingSpec` if all shard shapes are
// identical.
class ConcreteShardingSpec
    : public llvm::RTTIExtends<ConcreteShardingSpec, ShardingSpec> {
 public:
  // Creates a concrete sharding spec that may contain non-identical shard
  // shapes.
  static std::unique_ptr<ConcreteShardingSpec> Create(
      Shape shape, std::vector<Shape> shard_shapes,
      std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains =
          std::nullopt);

  // Creates a concrete sharding spec that may contain non-identical shard
  // dynamic shapes.
  static std::unique_ptr<ConcreteShardingSpec> Create(
      DynamicShape dynamic_shape,
      std::vector<DynamicShape> shard_dynamic_shapes);

  bool has_dynamic_shape() const {
    DCHECK(this);
    return std::holds_alternative<DynamicShape>(shape_) &&
           std::holds_alternative<std::vector<DynamicShape>>(shard_shapes_);
  }

  bool has_static_shape() const {
    DCHECK(this);
    return std::holds_alternative<Shape>(shape_) &&
           std::holds_alternative<std::vector<Shape>>(shard_shapes_);
  }

  const Shape& shape() const {
    DCHECK(has_static_shape());
    return std::get<Shape>(shape_);
  }

  const DynamicShape& dynamic_shape() const {
    DCHECK(has_dynamic_shape());
    return std::get<DynamicShape>(shape_);
  }

  const std::vector<Shape>& shard_shapes() const {
    DCHECK(this);
    DCHECK(std::holds_alternative<std::vector<Shape>>(shard_shapes_));
    return std::get<std::vector<Shape>>(shard_shapes_);
  }

  const std::vector<DynamicShape>& shard_dynamic_shapes() const {
    DCHECK(this);
    DCHECK(std::holds_alternative<std::vector<DynamicShape>>(shard_shapes_));
    return std::get<std::vector<DynamicShape>>(shard_shapes_);
  }

  const std::optional<std::vector<xla::ifrt::IndexDomain>>& index_domains()
      const {
    return index_domains_;
  }

  // ShardingSpec implementation.

  ~ConcreteShardingSpec() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const ShardingSpec& other) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  ConcreteShardingSpec(
      int num_shards, Shape shape, std::vector<Shape> shard_shapes,
      std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains);

  ConcreteShardingSpec(int num_shards, DynamicShape dynamic_shape,
                       std::vector<DynamicShape> shard_dynamic_shapes);

  void Hash(absl::HashState state) const override;

  std::variant<Shape, DynamicShape> shape_;
  std::variant<std::vector<Shape>, std::vector<DynamicShape>> shard_shapes_;
  std::optional<Shape> shard_shape_;
  std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains_;
};

// Opaque sharding spec that does not define a fixed semantics for conversion
// between a logical shape and shard shapes, and device placements. It can
// disassemble a certain shape into shard shapes that are identical.
class ConcreteEvenShardingSpec
    : public llvm::RTTIExtends<ConcreteEvenShardingSpec, ShardingSpec> {
 public:
  // Creates a concrete even sharding spec.
  // TODO(hyeontaek): Remove the default value of `is_fully_replicated` once all
  // callers are updated to provide it explicitly.
  static std::unique_ptr<ConcreteEvenShardingSpec> Create(
      int num_shards, Shape shape, Shape shard_shape,
      bool is_fully_replicated = false);

  Shape shape() const {
    DCHECK(this);
    return shape_;
  }
  const Shape& shard_shape() const {
    DCHECK(this);
    return shard_shape_;
  }

  // ShardingSpec implementation.

  ~ConcreteEvenShardingSpec() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const ShardingSpec& other) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  ConcreteEvenShardingSpec(int num_shards, Shape shape, Shape shard_shape,
                           bool is_fully_replicated);

  void Hash(absl::HashState state) const override;

  Shape shape_;
  Shape shard_shape_;
};

// Sharding spec derived from an IR ShardingParam.
class ShardingParamShardingSpec
    : public llvm::RTTIExtends<ShardingParamShardingSpec, ShardingSpec> {
 public:
  static std::unique_ptr<ShardingParamShardingSpec> Create(
      ShardingParam sharding_param);

  const ShardingParam& sharding_param() const { return sharding_param_; }

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const ShardingSpec& other) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  ShardingParamShardingSpec(int num_shards, ShardingParam sharding_param);

  void Hash(absl::HashState state) const override;

  ShardingParam sharding_param_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SHARDING_SPEC_H_
