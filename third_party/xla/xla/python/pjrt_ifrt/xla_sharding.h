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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// XLA-compatible sharding types.
class XlaCompatibleSharding
    : public llvm::RTTIExtends<XlaCompatibleSharding, Sharding> {
 public:
  using llvm::RTTIExtends<XlaCompatibleSharding, Sharding>::RTTIExtends;

  static char ID;  // NOLINT
};

// XLA `HloSharding` wrapper. `HloSharding` is the main sharding representation
// in XLA. This class holds an `HloSharding` to be used with IFRT.
class HloSharding final
    : public llvm::RTTIExtends<HloSharding, XlaCompatibleSharding> {
 public:
  // Creates an `HloSharding` wrapper. This bypasses consistency checks against
  // devices to optimize the common path of passing it to the user or to a
  // lower-level runtime. It is instead validated when the information in the
  // sharding is used within IFRT, e.g., in `Disassemble()`.
  static std::unique_ptr<HloSharding> Create(
      tsl::RCReference<DeviceList> devices, MemoryKind memory_kind,
      xla::HloSharding xla_hlo_sharding);

  // Returns the wrapped XLA `HloSharding`.
  const xla::HloSharding& xla_hlo_sharding() const { return xla_hlo_sharding_; }

  // Sharding implementation.

  ~HloSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<tsl::RCReference<DeviceList>> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
  Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<
      std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
  Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  HloSharding(tsl::RCReference<DeviceList> devices, MemoryKind memory_kind,
              xla::HloSharding xla_hlo_sharding);

  xla::HloSharding xla_hlo_sharding_;
};

// Test only: returns `HloSharding::IndexDomains()`, using `xla::HloSharding`
// APIs internally.
std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& sharding, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_H_
