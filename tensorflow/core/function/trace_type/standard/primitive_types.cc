/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/function/trace_type/standard/primitive_types.h"

#include <string>
#include <utility>

const int kNoneHash = 42;
const int kAnyHash = 4242;

namespace tensorflow {
namespace trace_type {

None::None() {}

std::unique_ptr<TraceType> None::clone() const {
  return std::unique_ptr<TraceType>(new None());
}

bool None::is_subtype_of(const TraceType& other) const {
  return *this == other;
}

std::unique_ptr<TraceType> None::most_specific_common_supertype(
    const std::vector<const TraceType*>& others) const {
  for (const auto& other : others) {
    if (*this != *other) return nullptr;
  }

  return std::unique_ptr<TraceType>(new None());
}

std::string None::to_string() const { return "None<>"; }

std::size_t None::hash() const { return kNoneHash; }

bool None::operator==(const TraceType& other) const {
  const None* casted_other = dynamic_cast<const None*>(&other);
  return (casted_other != nullptr);
}

Any::Any(absl::optional<std::unique_ptr<TraceType>> base)
    : base_(std::move(base)) {}

std::unique_ptr<TraceType> Any::clone() const {
  if (base_.has_value()) {
    return std::unique_ptr<TraceType>(new Any(base_.value()->clone()));
  }

  return std::unique_ptr<TraceType>(new Any(absl::nullopt));
}

absl::optional<const TraceType*> Any::base() const {
  if (base_.has_value()) {
    return base_.value().get();
  }

  return absl::nullopt;
}

bool Any::is_subtype_of(const TraceType& other) const {
  const Any* casted_other = dynamic_cast<const Any*>(&other);
  if (casted_other == nullptr) {
    return false;
  }

  if (!casted_other->base().has_value()) {
    return true;
  }

  if (!base_.has_value()) {
    return false;
  }

  return base_.value()->is_subtype_of(*casted_other->base().value());
}

std::unique_ptr<TraceType> Any::most_specific_common_supertype(
    const std::vector<const TraceType*>& others) const {
  std::vector<const Any*> casted_others;
  for (const auto& other : others) {
    const Any* casted_other = dynamic_cast<const Any*>(other);
    if (casted_other == nullptr) {
      return nullptr;
    }
    casted_others.push_back(casted_other);
  }

  if (!base_.has_value()) {
    return std::unique_ptr<TraceType>(new Any(absl::nullopt));
  }

  std::vector<const TraceType*> raw_ptrs;
  for (const auto& casted_other : casted_others) {
    if (!casted_other->base().has_value()) {
      return std::unique_ptr<TraceType>(new Any(absl::nullopt));
    }
    raw_ptrs.push_back(casted_other->base().value());
  }

  std::unique_ptr<TraceType> result(
      base_.value()->most_specific_common_supertype(raw_ptrs));

  if (result == nullptr) {
    return std::unique_ptr<TraceType>(new Any(absl::nullopt));
  } else {
    return std::unique_ptr<TraceType>(new Any(std::move(result)));
  }
}

std::string Any::to_string() const {
  return "Any<" + (base_.has_value() ? base_.value()->to_string() : "Any") +
         ">";
}

std::size_t Any::hash() const {
  return kAnyHash + (base_.has_value() ? base_.value()->hash() : 0);
}

bool Any::operator==(const TraceType& other) const {
  const Any* casted_other = dynamic_cast<const Any*>(&other);

  if (casted_other == nullptr ||
      base_.has_value() != casted_other->base().has_value()) {
    return false;
  }

  if (base_.has_value()) {
    return *base_.value() == *casted_other->base().value();
  }

  return true;
}

}  // namespace trace_type
}  // namespace tensorflow
