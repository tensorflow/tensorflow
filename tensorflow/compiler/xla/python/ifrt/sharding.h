/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHARDING_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHARDING_H_

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace ifrt {

// TODO(hyeontaek): Unify sharding types with jax::Sharding.

// Abstract sharding type.
//
// TODO(hyeontaek): There is an indication that we may prefer to split logical
// partitioning and device assignment into two separate data structures. It is
// common that an operation preserves the logical partitioning and only updates
// devices (e.g., "copy to devices" and portable execution). This fine-grained
// sharding design may help reduce overhead around these operations.
class Sharding : public llvm::RTTIExtends<Sharding, llvm::RTTIRoot> {
 public:
  // All devices in this sharding. Devices may appear more than once.
  const DeviceList& devices() const { return devices_; }

  // Breaks a shape up into per-device shapes and shardings. See
  // Array::Explode(). It may return an error if explosion is unsupported.
  virtual StatusOr<
      std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Explode(const Shape& shape) const = 0;

  virtual std::string DebugString() const = 0;

  static char ID;  // NOLINT

 protected:
  explicit Sharding(DeviceList devices) : devices_(devices) {}

  DeviceList devices_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

// Single-device sharding. It does not support per-device explosion.
//
// TODO(hyeontaek): `SingleDeviceSharding` tends to be created or consumed in a
// large quantity. It may be useful for performance optimization to special-case
// this sharding type rather than expressing it as a general `Sharding`.
class SingleDeviceSharding final
    : public llvm::RTTIExtends<SingleDeviceSharding, Sharding> {
 public:
  // Creates a single-device sharding.
  static std::shared_ptr<const Sharding> Create(Device* device);

  // Sharding implementation.

  ~SingleDeviceSharding() override = default;

  StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Explode(const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  explicit SingleDeviceSharding(Device* device)
      : llvm::RTTIExtends<SingleDeviceSharding, Sharding>(
            DeviceList({device})) {}
};

// Opaque sharding that does not define a fixed semantics for conversion between
// a logical shape and per-device shapes, and device placements.
//
// TODO(hyeontaek): In most cases, we have the same shape on each device. Make
// an OpaqueEqualSharding to save time to construct an explode function.
// TODO(hyeontaek): Make a separate type to explore non-explodable sharding.
class OpaqueSharding : public llvm::RTTIExtends<OpaqueSharding, Sharding> {
 public:
  using ExplodeFunc = std::function<StatusOr<std::vector<Shape>>(
      const OpaqueSharding&, const Shape&)>;

  // Creates an opaque sharding. `Explode()` will fail.
  static std::shared_ptr<const Sharding> Create(DeviceList devices);

  // Creates an opaque sharding with a custom shape exploding function.
  static std::shared_ptr<const Sharding> Create(DeviceList devices,
                                                ExplodeFunc explode_func);

  // Creates an ExplodeFunc from a list of shapes. The ExplodeFunc would ignore
  // sharding and shape arguments.
  static ExplodeFunc MakeExplodeFuncFromShapes(std::vector<Shape> shapes);

  ExplodeFunc explode_func() const {
    DCHECK(this);
    return explode_func_;
  }

  // Sharding implementation.

  ~OpaqueSharding() override = default;

  StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Explode(const Shape& shape) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  explicit OpaqueSharding(DeviceList devices, ExplodeFunc explode_func);

  ExplodeFunc explode_func_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHARDING_H_
