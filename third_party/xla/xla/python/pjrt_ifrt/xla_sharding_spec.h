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

#ifndef XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_SPEC_H_
#define XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_SPEC_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"

namespace xla {
namespace ifrt {

// XLA-compatible sharding spec types.
class XlaCompatibleShardingSpec
    : public llvm::RTTIExtends<XlaCompatibleShardingSpec, ShardingSpec> {
 public:
  using llvm::RTTIExtends<XlaCompatibleShardingSpec, ShardingSpec>::RTTIExtends;

  static char ID;  // NOLINT
};

// XLA `HloSharding` wrapper. `HloSharding` is the main sharding representation
// in XLA. This class holds an `HloSharding` to be used with IFRT as a
// `ShardingSpec`.
class HloShardingSpec final
    : public llvm::RTTIExtends<HloShardingSpec, XlaCompatibleShardingSpec> {
 public:
  // Creates an `HloShardingSpec` wrapper.
  static std::unique_ptr<HloShardingSpec> Create(
      int num_shards, xla::HloSharding xla_hlo_sharding);

  // Returns the wrapped XLA `HloSharding`.
  const xla::HloSharding& xla_hlo_sharding() const { return xla_hlo_sharding_; }

  // ShardingSpec implementation.

  ~HloShardingSpec() override = default;

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
  HloShardingSpec(int num_shards, xla::HloSharding xla_hlo_sharding);

  void Hash(absl::HashState state) const override;

  xla::HloSharding xla_hlo_sharding_;

  // Cached hash. 0 indicates the hash needs to be computed and cached.
  // May be written multiple times with the same non-zero value.
  static constexpr uint64_t kUnsetHash = 0;
  mutable std::atomic<uint64_t> hash_ = kUnsetHash;
};

// Test only: returns `HloShardingSpec::IndexDomains()`, using
// `xla::HloSharding` APIs internally.
std::vector<IndexDomain> TEST_HloShardingSpecIndexDomainsSlowPath(
    const HloShardingSpec& sharding_spec, const Shape& shape);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_XLA_SHARDING_SPEC_H_
