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

#ifndef XLA_PYTHON_IFRT_BUNDLE_H_
#define XLA_PYTHON_IFRT_BUNDLE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

class Bundle;
using BundleRef = tsl::RCReference<Bundle>;

// A runtime-managed data structure that represents an ordered list of
// `ValueRef`s.
class Bundle : public tsl::ReferenceCounted<Bundle>,
               public llvm::RTTIExtends<Bundle, llvm::RTTIRoot> {
 public:
  Bundle() = default;

  //// `Value`-like methods.

  virtual Client* client() const = 0;

  // Returns the user context associated with the creation of this `Bundle`.
  virtual UserContextRef user_context() const = 0;

  // Returns a future that becomes ready once all values in this `Bundle` are
  // ready.
  virtual tsl::Future<> GetReadyFuture() const = 0;

  // Deletes all values in this `Bundle`.
  virtual tsl::Future<> Delete() = 0;

  // Returns true if this `Bundle` has been deleted.
  virtual bool IsDeleted() const = 0;

  // Returns a string representation of this `Bundle`.
  virtual std::string DebugString() const = 0;

  //// Bundle-specific methods.

  // Returns the number of values in this `Bundle`. This is an inexpensive
  // operation.
  virtual int num_values() const = 0;

  // Expands this `Bundle` into `ValueRef`s.
  //
  // `semantics` must not be `kAlwaysCopy`.
  virtual absl::StatusOr<std::vector<ValueRef>> GetValues(
      ArrayCopySemantics semantics) = 0;

  // Returns the `ArraySpec` for each value in this `Bundle` without expanding
  // the `Bundle`.
  //
  // If any value is not an array, returns an error. The returned span is valid
  // while this `Bundle` is alive.
  //
  // TODO(hyeontaek): This is a transient API and will be updated with a more
  // general API that can handle "ValueSpec" (TBD) and not just `ArraySpec`s.
  virtual absl::StatusOr<absl::Span<const ArraySpec>> GetArraySpecs() const = 0;

  // Slices a `Bundle` into `Bundle`s.
  //
  // Each output `Bundle` will contain contiguous values of the specified size
  // from the input `Bundle`.
  //
  // The sum of `slice_sizes` must be equal to `num_values()`.
  // `semantics` must not be `kAlwaysCopy`.
  virtual absl::StatusOr<std::vector<BundleRef>> Slice(
      absl::Span<const int> slice_sizes, ArrayCopySemantics semantics) = 0;

  // Specification for copying a slice of the bundle.
  struct CopySpec {
    // New devices and memory kind.
    //
    // TODO(hyeontaek): Take `layout` once `CopyArrays` API is migrated to
    // strict custom layout handling where the user must provide a custom layout
    // explicitly whenever the destination array should use the custom layout.
    std::optional<DeviceListRef> devices;
    std::optional<MemoryKind> memory_kind;

    // Array copy semantics. Aliasing is default.
    ArrayCopySemantics semantics = ArrayCopySemantics::kReuseInput;
  };

  // Copies the arrays in this `Bundle` to create a new `Bundle`.
  //
  // Fails if this `Bundle` contains any non-`Array` value.
  //
  // This `Bundle` is logically sliced according to `slice_sizes`, with each
  // sliced `Bundle` copied using the corresponding `copy_specs`. Then,
  // the copied `Bundle`s are logically concatenated to a single result
  // `Bundle`.
  //
  // It is also semantically equivalent to applying a sequence of
  // `Bundle::GetValues()`, `Client::CopyArrays()`s, and `Client::Bundle()`, but
  // it does not create intermediate `Array`s.
  //
  // The size of `slice_sizes` must be equal to the size of `copy_specs`. The
  // sum of `slice_sizes` must be equal to `num_values()`.
  virtual absl::StatusOr<BundleRef> CopyArrays(
      absl::Span<const int> slice_sizes,
      absl::Span<const CopySpec> copy_specs) = 0;

  // Specification for resharding a slice of the bundle.
  struct ReshardSpec {
    // New array specs for the values in this slice.
    //
    // The size of the span must match the number of values in this slice.
    //
    // TODO(hyeontaek): Generalize it when we support non-array values.
    std::vector<ArraySpec> array_specs;

    // Array copy semantics for this slice. Aliasing is default.
    ArrayCopySemantics semantics = ArrayCopySemantics::kReuseInput;
  };

  // Reshards the arrays in this `Bundle` to create a new `Bundle`.
  //
  // Fails if this `Bundle` contains any non-`Array` value.
  //
  // This `Bundle` is logically sliced according to `slice_sizes`, with each
  // sliced `Bundle` resharded using the corresponding `reshard_specs`. Then,
  // the resharded `Bundle`s are logically concatenated to a single result
  // `Bundle`.
  //
  // It is also semantically equivalent to applying a sequence of
  // `Bundle::GetValues()`, `Client::ReshardArrays()`s, and `Client::Bundle()`,
  // but it does not create intermediate `Array`s.
  //
  // The size of `slice_sizes` must be equal to the size of `reshard_specs`. The
  // sum of `slice_sizes` must be equal to `num_values()`.
  virtual absl::StatusOr<BundleRef> ReshardArrays(
      absl::Span<const int> slice_sizes,
      absl::Span<const ReshardSpec> reshard_specs) = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_BUNDLE_H_
