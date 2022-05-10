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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"

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

Product::Product(std::vector<std::unique_ptr<TraceType>> elements)
    : elements_(std::move(elements)) {}

std::unique_ptr<TraceType> Product::clone() const {
  std::vector<std::unique_ptr<TraceType>> clones;
  clones.reserve(elements_.size());
  for (const auto& element : elements_) {
    clones.push_back(std::unique_ptr<TraceType>(element->clone()));
  }
  return std::unique_ptr<TraceType>(new Product(std::move(clones)));
}

const std::vector<const TraceType*> Product::elements() const {
  std::vector<const TraceType*> raw_ptrs;
  std::transform(elements_.begin(), elements_.end(),
                 std::back_inserter(raw_ptrs), [](auto& c) { return c.get(); });
  return raw_ptrs;
}

bool Product::is_subtype_of(const TraceType& other) const {
  const Product* collection_other = dynamic_cast<const Product*>(&other);
  if (collection_other == nullptr ||
      collection_other->elements().size() != elements_.size()) {
    return false;
  }

  return std::equal(elements_.begin(), elements_.end(),
                    collection_other->elements().begin(),
                    [](const std::unique_ptr<TraceType>& l,
                       const TraceType* r) { return l->is_subtype_of(*r); });
}

std::unique_ptr<TraceType> Product::most_specific_common_supertype(
    const std::vector<const TraceType*>& others) const {
  std::vector<const Product*> collection_others;
  for (const auto& other : others) {
    const Product* collection_other = dynamic_cast<const Product*>(other);
    if (collection_other == nullptr ||
        collection_other->elements().size() != elements_.size()) {
      return nullptr;
    }
    collection_others.push_back(collection_other);
  }

  std::vector<std::unique_ptr<TraceType>> element_supertypes;

  for (int i = 0; i < elements_.size(); i++) {
    std::vector<const TraceType*> raw_ptrs;
    raw_ptrs.reserve(collection_others.size());
    for (const auto& collection_other : collection_others) {
      raw_ptrs.push_back(collection_other->elements()[i]);
    }
    std::unique_ptr<TraceType> supertype =
        elements_[i]->most_specific_common_supertype(raw_ptrs);
    if (supertype == nullptr) return nullptr;
    element_supertypes.push_back(std::move(supertype));
  }

  return std::unique_ptr<TraceType>(new Product(std::move(element_supertypes)));
}

std::string Product::to_string() const {
  std::ostringstream ss;
  ss << "Product<";

  bool first = true;
  for (int i = 0; i < elements_.size(); i++) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << elements_[i]->to_string();
  }
  ss << ">";

  return std::string(ss.str());
}

std::size_t Product::hash() const {
  std::vector<std::size_t> hashes;
  std::for_each(elements_.begin(), elements_.end(),
                [&hashes](auto& c) { hashes.push_back(c->hash()); });

  return absl::HashOf(hashes);
}

bool Product::operator==(const TraceType& other) const {
  const Product* collection_other = dynamic_cast<const Product*>(&other);
  if (collection_other == nullptr ||
      collection_other->elements().size() != elements_.size()) {
    return false;
  }

  return std::equal(elements_.begin(), elements_.end(),
                    collection_other->elements().begin(),
                    [](const std::unique_ptr<TraceType>& l,
                       const TraceType* r) { return *l == *r; });
}

}  // namespace trace_type
}  // namespace tensorflow
