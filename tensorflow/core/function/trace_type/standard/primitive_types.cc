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
#include <iostream>
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

Record::Record(std::vector<std::unique_ptr<TraceType>> keys,
               std::vector<std::unique_ptr<TraceType>> values)
    : owned_keys_(std::move(keys)), owned_values_(std::move(values)) {
  assert(owned_keys_.size() == owned_values_.size());

  for (int i = 0; i < owned_keys_.size(); i++) {
    fields_.insert(
        std::make_pair(owned_keys_[i].get(), owned_values_[i].get()));
  }
}

std::unique_ptr<TraceType> Record::clone() const {
  std::vector<std::unique_ptr<TraceType>> new_keys;
  std::vector<std::unique_ptr<TraceType>> new_values;

  for (const auto& key_val_pair : fields_) {
    new_keys.push_back(std::unique_ptr<TraceType>(key_val_pair.first->clone()));
    new_values.push_back(
        std::unique_ptr<TraceType>(key_val_pair.second->clone()));
  }

  return std::unique_ptr<TraceType>(
      new Record(std::move(new_keys), std::move(new_values)));
}

const RecordMap& Record::fields() const { return fields_; }

bool Record::is_subtype_of(const TraceType& other) const {
  const Record* record_other = dynamic_cast<const Record*>(&other);
  if (record_other == nullptr ||
      record_other->fields().size() != fields_.size()) {
    return false;
  }

  for (const auto& key_val_pair : fields_) {
    const TraceType* key = key_val_pair.first;
    const TraceType* value = key_val_pair.second;
    if (record_other->fields().contains(key)) {
      const TraceType* other_value = record_other->fields().at(key);
      if (!value->is_subtype_of(*other_value)) return false;
    } else {
      return false;
    }
  }

  return true;
}

std::unique_ptr<TraceType> Record::most_specific_common_supertype(
    const std::vector<const TraceType*>& others) const {
  std::vector<const Record*> record_others;
  for (const auto& other : others) {
    const Record* record_other = dynamic_cast<const Record*>(other);
    if (record_other == nullptr ||
        record_other->fields().size() != fields_.size()) {
      return nullptr;
    }
    record_others.push_back(record_other);
  }

  std::vector<std::unique_ptr<TraceType>> keys;
  std::vector<std::unique_ptr<TraceType>> value_supertypes;

  for (const auto& key_val_pair : fields_) {
    const TraceType* key = key_val_pair.first;
    const TraceType* value = key_val_pair.second;
    keys.push_back(key->clone());

    std::vector<const TraceType*> raw_ptrs;
    raw_ptrs.reserve(record_others.size());
    for (const auto& record_other : record_others) {
      if (record_other->fields().contains(key)) {
        raw_ptrs.push_back(record_other->fields().at(key));
      } else {
        return nullptr;
      }
    }
    std::unique_ptr<TraceType> supertype =
        value->most_specific_common_supertype(raw_ptrs);
    if (supertype == nullptr) return nullptr;
    value_supertypes.push_back(std::move(supertype));
  }

  return std::unique_ptr<TraceType>(
      new Record(std::move(keys), std::move(value_supertypes)));
}

std::string Record::to_string() const {
  std::ostringstream ss;
  ss << "Record<";

  bool first = true;
  for (const auto& key_val_pair : fields_) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << key_val_pair.first->to_string();
    ss << ":";
    ss << key_val_pair.second->to_string();
  }

  ss << ">";
  return std::string(ss.str());
}

std::size_t Record::hash() const {
  std::set<std::size_t> hashes;
  for (const auto& key_val_pair : fields_) {
    hashes.insert(key_val_pair.first->hash());
  }
  return absl::HashOf(hashes);
}

bool Record::operator==(const TraceType& other) const {
  const Record* record_other = dynamic_cast<const Record*>(&other);
  if (record_other == nullptr ||
      record_other->fields().size() != fields_.size()) {
    return false;
  }

  for (const auto& key_val_pair : fields_) {
    const TraceType* key = key_val_pair.first;
    const TraceType* value = key_val_pair.second;
    if (record_other->fields().contains(key)) {
      const TraceType* other_value = record_other->fields().at(key);
      if (*value != *other_value) return false;
    } else {
      return false;
    }
  }

  return true;
}

UserDefinedType::UserDefinedType(std::string name,
                                 std::unique_ptr<TraceType> base)
    : name_(name), base_(std::move(base)) {}

std::unique_ptr<TraceType> UserDefinedType::clone() const {
  return std::unique_ptr<TraceType>(new UserDefinedType(name_, base_->clone()));
}

const std::string& UserDefinedType::name() const { return name_; }

const TraceType* UserDefinedType::base() const { return base_.get(); }

bool UserDefinedType::is_subtype_of(const TraceType& other) const {
  const UserDefinedType* casted_other =
      dynamic_cast<const UserDefinedType*>(&other);
  if (casted_other == nullptr || casted_other->name() != name_) {
    return false;
  }
  return base_->is_subtype_of(*casted_other->base());
}

std::unique_ptr<TraceType> UserDefinedType::most_specific_common_supertype(
    const std::vector<const TraceType*>& others) const {
  std::vector<const TraceType*> base_others;
  for (const auto& other : others) {
    const UserDefinedType* casted_other =
        dynamic_cast<const UserDefinedType*>(other);
    if (casted_other == nullptr || casted_other->name() != name_) {
      return nullptr;
    }
    base_others.push_back(casted_other->base());
  }

  std::unique_ptr<TraceType> result(
      base_->most_specific_common_supertype(base_others));

  if (result == nullptr) {
    return nullptr;
  } else {
    return std::unique_ptr<TraceType>(
        new UserDefinedType(name_, std::move(result)));
  }
}

std::string UserDefinedType::to_string() const {
  return name_ + "<" + base_->to_string() + ">";
}

std::size_t UserDefinedType::hash() const {
  return absl::HashOf(name_, base_->hash());
}

bool UserDefinedType::operator==(const TraceType& other) const {
  const UserDefinedType* casted_other =
      dynamic_cast<const UserDefinedType*>(&other);
  return casted_other != nullptr && casted_other->name() == name_ &&
         *casted_other->base() == *base_;
}

}  // namespace trace_type
}  // namespace tensorflow
