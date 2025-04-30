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

#ifndef XLA_PYTHON_IFRT_LAYOUT_H_
#define XLA_PYTHON_IFRT_LAYOUT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

class Client;
class Device;
class Sharding;
class Layout;

// Reference to a layout.
//
// If `nullptr`, it represents a default layout; since the concrete layout of a
// default layout is context-sensitive, the user must not treat two `nullptr`
// values as the same layout.
//
// If not `nullptr`, it represents a custom layout.
using LayoutRef = absl_nullable std::shared_ptr<const Layout>;

// Reference to a custom layout that is not a default layout.
using CustomLayoutRef = absl_nonnull std::shared_ptr<const Layout>;

// Abstract layout type.
//
// `Layout` describes how the elements of a single shard in an array are
// arranged in memory. A layout may contain transpose, padding, bit packing,
// sparsity, indirection, and so forth. All shards of a single array use the
// same layout.
//
// Note that within-element layouts such as big/little endian are not expressed
// through `Layout`. They may be expressed through `DType`.
class Layout : public llvm::RTTIExtends<Layout, Serializable> {
 public:
  Layout(const Layout&) = delete;
  Layout& operator=(const Layout&) = delete;
  Layout(Layout&&) = delete;
  Layout& operator=(Layout&&) = delete;

  // Computes the byte size of a shard shape using the layout. If `dtype`
  // represents non-fixed-size (e.g., `kString`), size-less (e.g., `kToken`), or
  // opaque (`kOpaque`) data, returns `std::nullopt`.
  virtual absl::StatusOr<std::optional<int64_t>> ByteSize(
      DType dtype, const Shape& shard_shape) const = 0;

  // Constructs `Layout` from `LayoutProto`.
  static absl::StatusOr<CustomLayoutRef> FromProto(const LayoutProto& proto);

  // Returns a `LayoutProto` representation.
  absl::StatusOr<LayoutProto> ToProto() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Layout& layout) {
    sink.Append(layout.ToString());
  }

  static char ID;  // NOLINT

 protected:
  Layout() = default;

 private:
  // `operator==` is expected to be used only by `EquivalentLayouts()`.
  friend absl::StatusOr<bool> EquivalentLayouts(
      DType dtype1, const Shape& shape1,
      const std::shared_ptr<const Sharding>& sharding1,
      const LayoutRef& layout1, DType dtype2, const Shape& shape2,
      const std::shared_ptr<const Sharding>& sharding2,
      const LayoutRef& layout2);
  virtual bool operator==(const Layout& other) const = 0;

  // Returns a string representation of the layout.
  virtual std::string ToString() const = 0;
};

// Concrete layout that expresses a compact layout using major-to-minor order of
// dimensions. There is no padding or gaps between elements. Sub-byte `DType`s
// such as `DType::kS4` use a packed layout.
class CompactLayout final : public llvm::RTTIExtends<CompactLayout, Layout> {
 public:
  static absl::StatusOr<absl_nonnull std::unique_ptr<CompactLayout>> Create(
      absl::Span<const int> major_to_minor);

  // Creates a compact layout that represents a C order (row-major order) layout
  // for a shard shape with `num_shard_shape_dims` dimensions.
  static absl_nonnull std::unique_ptr<CompactLayout> CreateCOrder(
      int num_shard_shape_dims);

  absl::Span<const int> major_to_minor() const { return major_to_minor_; }

  // Layout implementation.

  absl::StatusOr<std::optional<int64_t>> ByteSize(
      DType dtype, const Shape& shard_shape) const override;

  static char ID;  // NOLINT

 private:
  using MajorToMinor = absl::InlinedVector<int, 6>;

  explicit CompactLayout(MajorToMinor major_to_minor)
      : major_to_minor_(std::move(major_to_minor)) {}

  bool operator==(const Layout& other) const override;
  std::string ToString() const override;

  MajorToMinor major_to_minor_;
};

// Returns true if two array specs have equivalent layouts.
//
// Caution: It is not well-defined what it should return if two dtypes or shapes
// do not match. Typically, the caller should perform separate equivalence
// checks on dtype and shape as required.
//
// TODO(hyeontaek): Consider taking `ArraySpec` once `ArraySpec::layout` becomes
// a `LayoutRef`.
absl::StatusOr<bool> EquivalentLayouts(
    DType dtype1, const Shape& shape1,
    const std::shared_ptr<const Sharding>& sharding1, const LayoutRef& layout1,
    DType dtype2, const Shape& shape2,
    const std::shared_ptr<const Sharding>& sharding2, const LayoutRef& layout2);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_LAYOUT_H_
