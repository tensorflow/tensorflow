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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

// Wraps around `xla::PjRtLayout` as an IFRT Layout.
//
// Compatibility note: While `xla::PjRtLayout` may accept take an arbitrary
// `xla::Layout`, we strongly suggest using only a small subset of `xla::Layout`
// features (`minor_to_major`, `tiles`, and `element_size_in_bits`) that are
// approved for use in PjRt C API and less commonly used features.
class PjRtLayout final
    : public llvm::RTTIExtends<xla::ifrt::PjRtLayout, Layout> {
 public:
  // Creates a PjRtLayout.
  //
  // TODO(hyeontaek): Consider accepting only `xla::PjRtLayout` whose
  // `xla::Layout` uses supported features by PjRt.
  static absl_nonnull std::unique_ptr<PjRtLayout> Create(
      absl_nonnull std::shared_ptr<const xla::PjRtLayout> pjrt_layout);

  ~PjRtLayout() override = default;

  absl_nonnull const std::shared_ptr<const xla::PjRtLayout>& pjrt_layout()
      const {
    return pjrt_layout_;
  }

  // Layout implementation.

  absl::StatusOr<std::optional<int64_t>> ByteSize(
      DType dtype, const Shape& shard_shape) const override;

  static char ID;  // NOLINT

 private:
  explicit PjRtLayout(std::shared_ptr<const xla::PjRtLayout> pjrt_layout)
      : pjrt_layout_(std::move(pjrt_layout)) {}

  bool operator==(const Layout& other) const override;
  std::string ToString() const override;

  absl_nonnull std::shared_ptr<const xla::PjRtLayout> pjrt_layout_;
};

// Converts IFRT `CustomLayoutRef` into `xla::PjRtLayout`. Only supports a
// reference to IFRT `PjRtLayout` and `CompactLayout`, as input.
absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape,
             const CustomLayoutRef& layout);

// Converts IFRT `LayoutRef` into `xla::PjRtLayout`. Only supports `nullptr`
// (default layout), a reference to IFRT `PjRtLayout` and `CompactLayout` as
// input. `nullptr` will be resolved to a concrete layout.
//
// Do not use this API to check the equivalence of two `LayoutRef`s. Use
// `EquivalentLayouts` instead for a formal layout comparison logic.
absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape, Device* device,
             MemoryKind memory_kind, const LayoutRef& layout);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_H_
