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

#include "xla/python/ifrt/basic_bundle.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/bundle.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt/value_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

namespace {

// Checks if the given `ArrayCopySemantics` is supported by `BasicBundle`.
absl::Status ValidateArrayCopySemantics(ArrayCopySemantics semantics) {
  if (semantics == ArrayCopySemantics::kAlwaysCopy) {
    return absl::InvalidArgumentError(
        "`ArrayCopySemantics::kAlwaysCopy` is not supported");
  }
  return absl::OkStatus();
}

// Checks if the given `slice_sizes` are valid for the given `num_values`.
absl::Status ValidateSliceSizes(int num_values,
                                absl::Span<const int> slice_sizes) {
  int total_size = 0;
  for (int size : slice_sizes) {
    total_size += size;
  }
  if (total_size != num_values) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Sum of slice_sizes does not match number of values in the "
        "bundle: ",
        total_size, " vs ", num_values));
  }
  return absl::OkStatus();
}

}  // namespace

char BasicBundle::ID = 0;

absl::StatusOr<BundleRef> BasicBundle::Create(absl::Span<ValueRef> values,
                                              ArrayCopySemantics semantics) {
  RETURN_IF_ERROR(ValidateArrayCopySemantics(semantics));

  if (values.empty()) {
    return tsl::TakeRef<BasicBundle>(new BasicBundle(nullptr, {}));
  }

  Client* const client = values.front()->client();
  for (int i = 1; i < values.size(); ++i) {
    if (values[i]->client() != client) {
      return absl::InvalidArgumentError("Values must use the same client");
    }
  }

  std::vector<ValueRef> values_copy;
  values_copy.reserve(values.size());
  if (semantics == ArrayCopySemantics::kDonateInput) {
    absl::c_move(values, std::back_inserter(values_copy));
  } else {
    absl::c_copy(values, std::back_inserter(values_copy));
  }
  return tsl::TakeRef<BasicBundle>(
      new BasicBundle(client, std::move(values_copy)));
}

absl::StatusOr<BundleRef> BasicBundle::ConcatBundles(
    absl::Span<BundleRef> bundles, ArrayCopySemantics semantics) {
  RETURN_IF_ERROR(ValidateArrayCopySemantics(semantics));

  if (bundles.empty()) {
    return tsl::TakeRef<BasicBundle>(new BasicBundle(nullptr, {}));
  }

  Client* const client = bundles.front()->client();
  int total_size = 0;
  for (const BundleRef& bundle : bundles) {
    auto* basic_bundle = llvm::dyn_cast_or_null<BasicBundle>(bundle.get());
    if (basic_bundle == nullptr) {
      return absl::InvalidArgumentError(
          "`Bundle::ConcatBundles()` expects `BasicBundle`s");
    }
    if (basic_bundle->client_ != client) {
      return absl::InvalidArgumentError("Bundles must use the same client");
    }
    total_size += basic_bundle->values_.size();
  }
  std::vector<ValueRef> new_values;
  new_values.reserve(total_size);
  for (const BundleRef& bundle : bundles) {
    absl::c_copy(llvm::cast<BasicBundle>(bundle.get())->values_,
                 std::back_inserter(new_values));
  }
  return tsl::TakeRef<BasicBundle>(
      new BasicBundle(client, std::move(new_values)));
}

BasicBundle::BasicBundle(Client* client, std::vector<ValueRef> values)
    : client_(client),
      values_(std::move(values)),
      user_context_(UserContextScope::current()) {}

BasicBundle::~BasicBundle() {
  std::vector<ValueRef> to_delete;
  for (const auto& value : values_) {
    if (value->IsUnique()) {
      to_delete.push_back(value);
    }
  }
  client_->DeleteValues(absl::MakeSpan(to_delete));
}

tsl::Future<> BasicBundle::GetReadyFuture() const {
  if (values_.empty()) {
    return absl::OkStatus();
  }

  absl::MutexLock lock(mu_);
  if (!ready_future_.has_value()) {
    ready_future_ = client_->GetReadyFuture(values_);
  }
  return *ready_future_;
}

tsl::Future<> BasicBundle::Delete() {
  absl::MutexLock lock(mu_);
  if (!delete_future_.IsValid()) {
    std::vector<ValueRef> to_delete = values_;
    delete_future_ = client_->DeleteValues(absl::MakeSpan(to_delete));
  }
  return delete_future_;
}

bool BasicBundle::IsDeleted() const {
  absl::MutexLock lock(mu_);
  return delete_future_.IsValid();
}

std::string BasicBundle::DebugString() const {
  return absl::StrCat("BasicBundle(num_values=", values_.size(), ")");
}

absl::StatusOr<std::vector<ValueRef>> BasicBundle::GetValues(
    ArrayCopySemantics semantics) {
  RETURN_IF_ERROR(ValidateArrayCopySemantics(semantics));
  return values_;
}

absl::StatusOr<absl::Span<const ArraySpec>> BasicBundle::GetArraySpecs() const {
  absl::MutexLock lock(mu_);
  if (!array_specs_.has_value()) {
    array_specs_ = [&]() -> absl::StatusOr<std::vector<ArraySpec>> {
      std::vector<ArraySpec> array_specs;
      array_specs.reserve(values_.size());
      for (const ValueRef& value : values_) {
        if (auto* array = llvm::dyn_cast_or_null<Array>(value.get())) {
          ASSIGN_OR_RETURN(std::shared_ptr<const xla::PjRtLayout> layout,
                           array->pjrt_layout());
          array_specs.push_back(ArraySpec{
              /*dtype=*/array->dtype(),
              /*shape=*/array->shape(),
              /*sharding=*/array->shared_ptr_sharding(),
              /*layout=*/std::move(layout),
          });
        } else {
          return absl::InvalidArgumentError(
              "`Bundle::GetArraySpecs()` requires all values to be `Array`s");
        }
      }
      return array_specs;
    }();
  }
  return *array_specs_;
}

absl::StatusOr<std::vector<BundleRef>> BasicBundle::Slice(
    absl::Span<const int> slice_sizes, ArrayCopySemantics semantics) {
  RETURN_IF_ERROR(ValidateSliceSizes(values_.size(), slice_sizes));
  RETURN_IF_ERROR(ValidateArrayCopySemantics(semantics));

  std::vector<BundleRef> slices;
  slices.reserve(slice_sizes.size());
  int offset = 0;
  for (int size : slice_sizes) {
    std::vector<ValueRef> slice_values;
    slice_values.reserve(size);
    for (int j = 0; j < size; ++j) {
      slice_values.push_back(values_[offset]);
      ++offset;
    }

    slices.push_back(
        tsl::TakeRef(new BasicBundle(client_, std::move(slice_values))));
  }
  return slices;
}

absl::StatusOr<BundleRef> BasicBundle::CopyArrays(
    absl::Span<const int> slice_sizes, absl::Span<const CopySpec> copy_specs,
    ArrayCopySemantics semantics) {
  RETURN_IF_ERROR(ValidateSliceSizes(values_.size(), slice_sizes));

  std::vector<ValueRef> new_values;
  new_values.reserve(values_.size());
  int offset = 0;
  for (int i = 0; i < slice_sizes.size(); ++i) {
    std::vector<ArrayRef> arrays;
    arrays.reserve(slice_sizes[i]);
    for (int j = 0; j < slice_sizes[i]; ++j) {
      if (auto* array = llvm::dyn_cast_or_null<Array>(values_[offset].get())) {
        arrays.push_back(tsl::FormRef(array));
      } else {
        return absl::InvalidArgumentError(
            "`Bundle:CopyArrays()` requires all values to be `Array`s");
      }
      ++offset;
    }

    ASSIGN_OR_RETURN(
        std::vector<ArrayRef> copied_arrays,
        client_->CopyArrays(absl::MakeSpan(arrays), copy_specs[i].devices,
                            copy_specs[i].memory_kind, semantics));
    absl::c_move(ToValues(std::move(copied_arrays)),
                 std::back_inserter(new_values));
  }
  if (semantics == ArrayCopySemantics::kDonateInput) {
    absl::MutexLock lock(mu_);
    if (!delete_future_.IsValid()) {
      delete_future_ = absl::OkStatus();
    }
  }
  return tsl::TakeRef<BasicBundle>(
      new BasicBundle(client_, std::move(new_values)));
}

absl::StatusOr<BundleRef> BasicBundle::ReshardArrays(
    absl::Span<const int> slice_sizes,
    absl::Span<const ReshardSpec> reshard_specs, ArrayCopySemantics semantics) {
  RETURN_IF_ERROR(ValidateSliceSizes(values_.size(), slice_sizes));

  std::vector<ValueRef> new_values;
  new_values.reserve(values_.size());
  int offset = 0;
  for (int i = 0; i < slice_sizes.size(); ++i) {
    std::vector<ArrayRef> arrays;
    arrays.reserve(slice_sizes[i]);
    for (int j = 0; j < slice_sizes[i]; ++j) {
      if (auto* array = llvm::dyn_cast_or_null<Array>(values_[offset].get())) {
        arrays.push_back(tsl::FormRef(array));
      } else {
        return absl::InvalidArgumentError(
            "`Bundle::ReshardArrays()` requires all values to be `Array`s");
      }
      ++offset;
    }

    ASSIGN_OR_RETURN(
        std::vector<ArrayRef> resharded_arrays,
        client_->ReshardArrays(absl::MakeSpan(arrays),
                               reshard_specs[i].array_specs, semantics));
    absl::c_move(ToValues(std::move(resharded_arrays)),
                 std::back_inserter(new_values));
  }
  if (semantics == ArrayCopySemantics::kDonateInput) {
    absl::MutexLock lock(mu_);
    if (!delete_future_.IsValid()) {
      delete_future_ = absl::OkStatus();
    }
  }
  return tsl::TakeRef<BasicBundle>(
      new BasicBundle(client_, std::move(new_values)));
}

}  // namespace ifrt
}  // namespace xla
