/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_SHARDING_H_
#define XLA_PYTHON_IFRT_SHARDING_H_

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.pb.h"

namespace xla {
namespace ifrt {

// TODO(hyeontaek): Unify sharding types with jax::Sharding.

struct DeserializeShardingOptions;

// Abstract sharding type.
//
// TODO(hyeontaek): There is an indication that we may prefer to split logical
// partitioning and device assignment into two separate data structures. It is
// common that an operation preserves the logical partitioning and only updates
// devices (e.g., "copy to devices" and portable execution). This fine-grained
// sharding design may help reduce overhead around these operations.
class Sharding : public llvm::RTTIExtends<Sharding, Serializable> {
 public:
  using DeserializeOptions = DeserializeShardingOptions;

  // All devices in this sharding. Devices may appear more than once.
  const DeviceList& devices() const { return devices_; }

  // Returns the memory kind for all shards in this sharding.
  MemoryKind memory_kind() const { return memory_kind_; }

  // Returns if this sharding is fully replicated. A fully replicated sharding
  // means that the logical shape and shard shapes are identical
  // (`GetShardShape(shape) == shape`), and every shard of the array contains
  // the entire data of the logical array.
  bool IsFullyReplicated() const { return is_fully_replicated_; }

  // Returns if this sharding is equal to `other`.
  bool operator==(const Sharding& other) const;
  bool operator!=(const Sharding& other) const { return !(*this == other); }

  // Returns a shard shape if the sharding always has the equal shape for all
  // shards. Returns an error if the sharding may not have a single shard
  // shape, or `shape` is not a valid shape for this sharding.
  virtual absl::StatusOr<Shape> GetShardShape(const Shape& shape) const = 0;

  // Returns if this sharding has the same logical partitioning as `other`. By
  // the same logical partitioning, we mean that `Sharding` type is the same,
  // and the partitioning scheme within the sharding is equivalent. It does not
  // need to check if `Disassemble()` would return the same result.
  virtual bool HasSamePartitioning(const Sharding& other) const = 0;

  // Returns a new sharding with the same logical partitioning as this sharding,
  // but with different devices and/or a different memory kind. If `devices` is
  // provided, the number of devices must be the same as the number of devices
  // in this sharding. If `memory_kind` is provided, it must be a valid memory
  // kind for the devices used.
  virtual absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceList> devices,
      std::optional<MemoryKind> memory_kind) const = 0;

  // Breaks a shape up into per-device shapes and shardings. See
  // Array::DisassembleIntoSingleDeviceArrays(). It may return an error if
  // disassembly is unsupported.
  virtual absl::StatusOr<
      std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const = 0;

  // Variant of `Disassemble` that takes a dynamic shape.
  virtual absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const = 0;

  // Maps each shard to an `IndexDomain` over `shape`. The result is a list of
  // `index_domain_i` such that `array[index_domain_i] = disassembled_array_i`.
  // Note that multiple shards may map onto equal `IndexDomain`. For instance, a
  // fully replicated sharding would return a vector of `[IndexDomain(shape)] *
  // devices().size()`.
  virtual absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const = 0;

  // Deserializes `ShardingProto` into `Sharding`.
  // Note that `Sharding` serialization uses `SerDes` to handle an open set of
  // `Sharding` subclasses. See `serdes.h`.
  static absl::StatusOr<std::unique_ptr<Sharding>> FromProto(
      DeviceList::LookupDeviceFunc lookup_device,
      const ShardingProto& sharding_proto);

  // Serializes `Sharding` into `ShardingProto`.
  // Note that `Sharding` serialization uses `SerDes` to handle an open set of
  // `Sharding` subclasses. See `serdes.h`.
  absl::StatusOr<ShardingProto> ToProto() const;

  virtual std::string DebugString() const = 0;

  static char ID;  // NOLINT

 protected:
  Sharding(DeviceList devices, MemoryKind memory_kind, bool is_fully_replicated)
      : devices_(devices),
        memory_kind_(memory_kind),
        is_fully_replicated_(is_fully_replicated) {}

  DeviceList devices_;
  MemoryKind memory_kind_;
  bool is_fully_replicated_;
};

std::ostream& operator<<(std::ostream& os, const Sharding& sharding);

// TODO(hyeontaek): Move the subclasses of `Sharding` to a seperate file,
// making this sharding.{h,cc} only define interface and common functions.

// Single-device sharding.
//
// TODO(hyeontaek): `SingleDeviceSharding` tends to be created or consumed in a
// large quantity. It may be useful for performance optimization to special-case
// this sharding type rather than expressing it as a general `Sharding`.
class SingleDeviceSharding final
    : public llvm::RTTIExtends<SingleDeviceSharding, Sharding> {
 public:
  // Creates a single-device sharding.
  static std::unique_ptr<SingleDeviceSharding> Create(Device* device,
                                                      MemoryKind memory_kind);

  // Sharding implementation.

  ~SingleDeviceSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceList> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const override;

  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  explicit SingleDeviceSharding(Device* device, MemoryKind memory_kind)
      : llvm::RTTIExtends<SingleDeviceSharding, Sharding>(
            DeviceList({device}), memory_kind, /*is_fully_replicated=*/true) {}
};

// Opaque sharding that does not define a fixed semantics for conversion between
// a logical shape and per-device shapes, and device placements.
class OpaqueSharding : public llvm::RTTIExtends<OpaqueSharding, Sharding> {
 public:
  // Creates an opaque sharding. `Disassemble()` will fail.
  static std::unique_ptr<OpaqueSharding> Create(DeviceList devices,
                                                MemoryKind memory_kind);

  // Sharding implementation.

  ~OpaqueSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceList> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const override;

  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  explicit OpaqueSharding(DeviceList devices, MemoryKind memory_kind);
};

// Opaque sharding that does not define a fixed semantics for conversion between
// a logical shape and shard shapes, and device placements. It can disassemble a
// certain shape into shard shapes that may not be identical. It is advised to
// use `ConcreteEvenSharding` if all shard shapes are identical.
class ConcreteSharding : public llvm::RTTIExtends<ConcreteSharding, Sharding> {
 public:
  // Creates a concrete sharding that may contain non-identical shard shapes.
  // REQUIRES: `devices`.size() == `shard_shapes`.size()
  static std::unique_ptr<ConcreteSharding> Create(
      DeviceList devices, MemoryKind memory_kind, Shape shape,
      std::vector<Shape> shard_shapes);

  // Creates a concrete sharding that may contain non-identical shard dynamic
  // shapes.
  // REQUIRES: `devices`.size() == `shard_dynamic_shapes`.size()
  static std::unique_ptr<ConcreteSharding> Create(
      DeviceList devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
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

  // Sharding implementation.

  ~ConcreteSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceList> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const override;
  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  ConcreteSharding(DeviceList devices, MemoryKind memory_kind, Shape shape,
                   std::vector<Shape> shard_shapes);

  ConcreteSharding(DeviceList devices, MemoryKind memory_kind,
                   DynamicShape dynamic_shape,
                   std::vector<DynamicShape> shard_dynamic_shapes);

  std::variant<Shape, DynamicShape> shape_;
  std::variant<std::vector<Shape>, std::vector<DynamicShape>> shard_shapes_;
};

// Opaque sharding that does not define a fixed semantics for conversion between
// a logical shape and shard shapes, and device placements. It can disassemble a
// certain shape into shard shapes that are identical.
class ConcreteEvenSharding
    : public llvm::RTTIExtends<ConcreteEvenSharding, Sharding> {
 public:
  // Creates a concrete even sharding.
  // TODO(hyeontaek): Remove the default value of `is_fully_replicated` once all
  // callers are updated to provide it explicitly.
  static std::unique_ptr<ConcreteEvenSharding> Create(
      DeviceList devices, MemoryKind memory_kind, Shape shape,
      Shape shard_shape, bool is_fully_replicated = false);

  Shape shape() const {
    DCHECK(this);
    return shape_;
  }
  const Shape& shard_shape() const {
    DCHECK(this);
    return shard_shape_;
  }

  // Sharding implementation.

  ~ConcreteEvenSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceList> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const override;
  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  ConcreteEvenSharding(DeviceList devices, MemoryKind memory_kind, Shape shape,
                       Shape shard_shape, bool is_fully_replicated);

  Shape shape_;
  Shape shard_shape_;
};

// Sharding derived from an IR ShardingParam.
class ShardingParamSharding
    : public llvm::RTTIExtends<ShardingParamSharding, Sharding> {
 public:
  static absl::StatusOr<std::unique_ptr<ShardingParamSharding>> Create(
      ShardingParam sharding_param, DeviceList devices, MemoryKind memory_kind);

  const ShardingParam& sharding_param() const { return sharding_param_; }

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceList> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const override;
  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  ShardingParamSharding(ShardingParam sharding_param, DeviceList devices,
                        MemoryKind memory_kind);

  ShardingParam sharding_param_;
};

// Options for deserializing shardings. Function referenced by `lookup_device`
// must remain valid during deserialization.
struct DeserializeShardingOptions
    : llvm::RTTIExtends<DeserializeShardingOptions, DeserializeOptions> {
  explicit DeserializeShardingOptions(
      DeviceList::LookupDeviceFunc lookup_device)
      : lookup_device(lookup_device) {}

  static char ID;  // NOLINT

  // Function that converts device ids to devices.
  DeviceList::LookupDeviceFunc lookup_device;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SHARDING_H_
