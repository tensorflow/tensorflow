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

#ifndef XLA_PYTHON_IFRT_BASIC_BUNDLE_H_
#define XLA_PYTHON_IFRT_BASIC_BUNDLE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/bundle.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace ifrt {

class BasicBundle final : public llvm::RTTIExtends<BasicBundle, Bundle> {
 public:
  // Creates a new `BundleRef` from `ValueRef`s.
  static absl::StatusOr<BundleRef> Create(absl::Span<ValueRef> values,
                                          ArrayCopySemantics semantics);

  // Concatenates the given `Bundle`s into a single `Bundle`.
  static absl::StatusOr<BundleRef> ConcatBundles(absl::Span<BundleRef> bundles,
                                                 ArrayCopySemantics semantics);

  ~BasicBundle() final;

  Client* client() const final { return client_; }

  UserContextRef user_context() const final { return user_context_; }

  tsl::Future<> GetReadyFuture() const final;

  tsl::Future<> Delete() final;

  bool IsDeleted() const final;

  std::string DebugString() const final;

  int num_values() const final { return values_.size(); }

  absl::StatusOr<std::vector<ValueRef>> GetValues(
      ArrayCopySemantics semantics) final;

  absl::StatusOr<absl::Span<const ArraySpec>> GetArraySpecs() const final;

  absl::StatusOr<std::vector<BundleRef>> Slice(
      absl::Span<const int> sizes, ArrayCopySemantics semantics) final;

  absl::StatusOr<BundleRef> CopyArrays(absl::Span<const int> slice_sizes,
                                       absl::Span<const CopySpec> copy_specs,
                                       ArrayCopySemantics semantics) final;

  absl::StatusOr<BundleRef> ReshardArrays(
      absl::Span<const xla::ifrt::ArraySpec> array_specs,
      ArrayCopySemantics semantics) final;

  static char ID;  // NOLINT

 private:
  BasicBundle(Client* client, std::vector<ValueRef> values);

  Client* const client_;
  // We avoid moving the elements of `values_` regardless of
  // `ArrayCopySemantics` so that `value_` is const and needs not be protected
  // by a mutex.
  const std::vector<ValueRef> values_;
  const UserContextRef user_context_;

  mutable absl::Mutex mu_;
  mutable std::optional<absl::StatusOr<std::vector<ArraySpec>>> array_specs_;
  mutable std::optional<tsl::Future<>> ready_future_ ABSL_GUARDED_BY(mu_);
  tsl::Future<> delete_future_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_BASIC_BUNDLE_H_
